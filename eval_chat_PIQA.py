import os
import argparse
import re
import torch
import pandas as pd
from tqdm import tqdm
from thefuzz import process
from utils import load, TASK_NAME_MAPPING, SUBJECTS, choices, SIQA_choices, PIQA_choices
from transformers.trainer_utils import set_seed
from transformers.generation import GenerationConfig

import logging
import json


def format_example(line, is_cot=True):
    cot_prompt = "For each multiple-choice question, follow this specific response structure: Begin by carefully analyzing the question and its options, labeled A and B. Start your response with the header 'Analysis:', under which you will delve into a detailed examination of each option. Discuss the logic, relevance, and factual accuracy of each option, considering how they relate to the question. Evaluate why some options may be incorrect and why others could be correct, based on your understanding of the topic. Once you have completed this in-depth analysis, proceed to the next section with the header 'Response:', where you will clearly state your final answer, choosing from options A, B, or C. This structured approach is designed to prioritize a comprehensive analysis before arriving at a conclusion, thereby enhancing the depth and clarity of your reasoning.\n\n" if is_cot else ""
        
    example = cot_prompt + line["goal"] + "\n\n"
    
    for idx, choice in enumerate(PIQA_choices):
        example += f'{choice}. {line[f"sol{idx + 1}"]}\n'
        
    return example


def process_before_extraction(gen, choice_dict):
    # replace the choice by letter in the generated sentence
    # from longest one to shortest one
    for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
        pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
        gen = pattern.sub(key, gen)
    return gen


def extract_choice(gen, choice_list):
    res = re.search(
        r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^AB]{0,20}?(?:n't|not))[^AB]{0,10}?\b(?:|is|:|be))\b)[^AB]{0,20}?\b(A|B)\b",
        gen,
    )

    if res is None:
        res = re.search(
            r"\b(A|B)\b(?![^AB]{0,8}?(?:n't|not)[^AB]{0,5}?(?:correct|right))[^AB]{0,10}?\b(?:correct|right)\b",
            gen,
        )

    if res is None:
        res = re.search(r"^(A|B)(?:\.|,|:|$)", gen)

    if res is None:
        res = re.search(r"(?<![a-zA-Z])(A|B)(?![a-zA-Z=])", gen)

    if res is None:
        return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
    return res.group(1)


def extract_answer(response, row):
    gen = process_before_extraction(response, {choice: row[f"sol{idx + 1}"] for idx, choice in enumerate(PIQA_choices)})
    pred = extract_choice(gen, [row[f"sol{idx + 1}"] for idx, choice in enumerate(PIQA_choices)])
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
def eval_piqa(
    model,
    tokenizer,
    test_data,
    save_result_dir=None,
    is_cot=True,
    **kwargs
):
    if save_result_dir:
        os.makedirs(save_result_dir, exist_ok=True)  # Ensure the directory exists
        result_path = os.path.join(save_result_dir, "llama7b_chat_cot_attenmod_piqa_predictions.csv")
        responses_path = os.path.join(save_result_dir, "llama7b_chat_cot_attenmod_piqa_responses.csv")
    else:
        result_path = None
        responses_path = None

    result = []
    responses = []

    for _, row in tqdm(test_data.iterrows(), total=len(test_data)):
        question = format_example(row, is_cot)
        question += "\n Answer:" if not is_cot else  "\n Analysis:"
        input_ids = tokenizer(question, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        past_key_values, response = answer_generate(
            model, tokenizer, input_ids, None, max_gen_len=2048
        )
        
        pred = extract_answer(response, row)
        result.append(pred)
        responses.append(response)
    
    if save_result_dir:
        dataset_name = "physicaliqa"
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write("dataset,prediction\n")
            for pred in result:
                pred_numeric = PIQA_choices.index(pred) + 1  # Convert A/B to 1/2
                f.write(f"{dataset_name},{pred_numeric}\n")
        
        response_data = pd.DataFrame({"response": responses})
        response_data.to_csv(responses_path, index=False)

    return result
    

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

    test_data = []
    with open(args.eval_data_path, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    test_df = pd.DataFrame(test_data)

    eval_piqa(
        model,
        tokenizer,
        test_df,
        save_result_dir="outs_chat/piqa_eval_result",
        is_cot=args.enable_cot,
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test HF checkpoint.")
    parser.add_argument(
        "-c",
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default="/sata/data/LLMs/Llama-2-7b-chat-hf/",
    )
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")

    # Provide extra arguments required for tasks
    group = parser.add_argument_group(title="Evaluation options")
    group.add_argument("-d", "--eval_data_path", type=str, help="Path to eval data")
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