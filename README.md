# Extending Token Computation for LLM Reasoning 
[[paper](http://arxiv.org/abs/2403.14932)]

## Abstract
Large Language Models (LLMs) are pivotal in advancing natural language processing but often struggle with complex reasoning tasks due to inefficient attention distributions. In this paper, we explore the effect of increased computed tokens on LLM performance and introduce a novel method for extending computed tokens in the Chain-of-Thought (CoT) process, utilizing attention mechanism optimization. By fine-tuning an LLM on a domain-specific, highly structured dataset, we analyze attention patterns across layers, identifying inefficiencies caused by non-semantic tokens with outlier high attention scores. To address this, we propose an algorithm that emulates early layer attention patterns across downstream layers to re-balance skewed attention distributions and enhance knowledge abstraction. Our findings demonstrate that our approach not only facilitates a deeper understanding of the internal dynamics of LLMs but also significantly improves their reasoning capabilities, particularly in non-STEM domains. Our study lays the groundwork for further innovations in LLM design, aiming to create more powerful, versatile, and responsible models capable of tackling a broad range of real-world applications.

## Usage

### Environment Setup

```bash
conda create -n attllm python=3.8
conda activate attllm

pip install torch torchvision torchaudio
pip install transformers==4.33.0 accelerate datasets scipy sentencepiece
```
### Prepare Weights
Download the Llama-2-7B-chat-hf weights (.bin files) into the /Llama-2-7B-chat-hf folder.

### Test MMLU

```bash
CUDA_VISIBLE_DEVICES=0 python eval_chat_mmlu.py  --enable_cot
```
Please note that the MMLU test script uses regex to extract the answer generated by the LLM. When using Chain-of-Thought (CoT) reasoning, this can cause inaccuracies. In our paper, we manually checked the generated answers to ensure accuracy. The manually checked results are also uploaded in the /outs_chat folders.

## Citation

If you find our works useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{liao2024extending,
        title={Extending Token Computation for LLM Reasoning},
        author={Bingli Liao and Danilo Vasconcellos Vargas},
        journal={arXiv},
        year={2024}
        }
```
