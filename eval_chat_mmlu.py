import os
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from .utils import load, TASK_NAME_MAPPING, SUBJECTS, choices
from transformers.trainer_utils import set_seed
from transformers.generation import GenerationConfig

import logging

'''
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar
mkdir data/mmlu
mv data.tar data/mmlu
cd data/mmlu; tar xf data.tar
cd ../../

pip install thefuzz
python eval/evaluate_chat_mmlu.py -d data/mmlu/data/

'''


def format_example(line, is_cot=True):
    cot_prompt = "For each multiple-choice question, follow this specific response structure: Begin by carefully analyzing the question and its options, labeled A, B, C, and D. Start your response with the header 'Analysis:', under which you will delve into a detailed examination of each option. Discuss the logic, relevance, and factual accuracy of each option, considering how they relate to the question. Evaluate why some options may be incorrect and why others could be correct, based on your understanding of the topic. Once you have completed this in-depth analysis, proceed to the next section with the header 'Response:', where you will clearly state your final answer, choosing from options A, B, C, or D. This structured approach is designed to prioritize a comprehensive analysis before arriving at a conclusion, thereby enhancing the depth and clarity of your reasoning.\n\n" if is_cot else ""
        
    example = cot_prompt + line["question"] + "\n\n"
    
    for choice in choices:
        example += f'{choice}. {line[f"{choice}"]}\n'
        
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    # answer is A | choice is A | choose A
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
        gen,
    )

    # A is correct | A is right
    if res is None:
        res = re.search(
            r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    # straight answer: A
    if res is None:
        res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)

    # simply extract the first appearred letter
    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(response, {choice: row[choice] for choice in choices})
    pred = extract_choice(gen, [row[choice] for choice in choices])
    return pred


@torch.no_grad()
def answer_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    generated_ids = [pred_token_idx.item()]
    pos = 0
    
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = (
            tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
                spaces_between_special_tokens=False,
            )
            .strip()
            .split(" ")
        )
        now = len(generated_text) - 1
        if now > pos:
            pos = now

        if pred_token_idx == tokenizer.eos_token_id:
            break
    
    return past_key_values, " ".join(generated_text)


@torch.no_grad()
def eval_subject(
    model,
    tokenizer,
    subject_name,
    test_df,
    save_result_dir=None,
    overwrite=False,
    is_cot=True,
    **kwargs
):
    result_path = os.path.join(save_result_dir, f"{subject_name}_result.csv")

    if not overwrite and os.path.exists(result_path):
        print(f"{result_path} already exists, skip!")
        score = []
        for (_, datarow), (_, resultrow) in zip(
            test_df.iterrows(), pd.read_csv(result_path).iterrows()
        ):
            pred = resultrow["model_output"]
            correct = 1 if pred == datarow["answer"] else 0
            score.append(correct)
        return score

    result = []
    score = []
    responses = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        question = format_example(row, is_cot)
        question += "\n Answer:" if not is_cot else  "\n Analysis:"
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        past_key_values, response = answer_generate(
            model, tokenizer, input_ids, None, max_gen_len=2048
        )
        
        pred = extract_answer(response, row)
        
        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            if args.debug:
                print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)
        responses.append(response)

    if save_result_dir:
        test_df["model_output"] = result
        test_df["model_response"] = responses
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return score


def cal_mmlu(res):
    acc_sub_dict = dict()
    sub_cnt_dict = dict()
    acc_sum_dict = dict()
    acc_norm_sum_dict = dict()
    cnt_dict = dict()
    acc_sum = 0.0
    cnt = 0

    for class_ in TASK_NAME_MAPPING.keys():
        acc_sum_dict[class_] = 0.0
        acc_norm_sum_dict[class_] = 0.0
        cnt_dict[class_] = 0.0

        for tt in TASK_NAME_MAPPING[class_]:
            acc_sum += sum(res[tt])
            cnt += len(res[tt])

            acc_sum_dict[class_] += sum(res[tt])
            cnt_dict[class_] += len(res[tt])
            
            acc_sub_dict[tt] = sum(res[tt])
            sub_cnt_dict[tt] = len(res[tt])

    print("\n\n\n")
    for k in TASK_NAME_MAPPING.keys():
        if k in cnt_dict:
            print("%s ACC: %.2f " % (k, acc_sum_dict[k] * 100 / cnt_dict[k]))
        for sub in TASK_NAME_MAPPING[k]:
            print("%s ACC: %.2f " % (sub, acc_sub_dict[sub] * 100 / sub_cnt_dict[sub]))
    print("AVERAGE ACC:%.2f " % (acc_sum * 100 / cnt))
    

def main(args):
    print("loading model weights")
    if args.checkpoint_path is not None:
        model, tokenizer = load(args.checkpoint_path)

        model.generation_config = GenerationConfig.from_pretrained(args.checkpoint_path, trust_remote_code=True)
        model.generation_config.do_sample = False  # use greedy decoding
        model.generation_config.repetition_penalty = 1.0  # disable repetition penalty
    else:
        model, tokenizer = None, None
    print("model loaded")
    print(f"enable CoT: {args.enable_cot}")

    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        test_file_path = os.path.join(
            args.eval_data_path, "test", f"{subject_name}_test.csv"
        )
        
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )

        score = eval_subject(
            model,
            tokenizer,
            subject_name,
            test_df,
            save_result_dir=f"outs_chat/mmlu_eval_result",
            overwrite=args.overwrite,
            is_cot=args.enable_cot,
        )
        dev_result[subject_name] = score

    cal_mmlu(dev_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="./Llama-2-7b-chat-hf/",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data", default="./data/MMLU")
    group.add_argument(
        "--debug", action="store_true", default=False, help="Print infos."
    )
    group.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existed results",
    )
    parser.add_argument("--enable_cot", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    main(args)
