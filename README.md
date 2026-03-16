# LLM Fine-Tuning with LoRA

Parameter-efficient fine-tuning of a large language model using **LoRA (Low-Rank Adaptation)**. This project adapts the pretrained **LiquidAI LFM2-1.2B** causal language model into a style-aware conversational chatbot while training only a very small subset of parameters.

Built in a **Kaggle notebook environment** with **PyTorch**, **Hugging Face Transformers**, and **PEFT**, the project shows how billion-parameter LLMs can be customized on limited hardware without full-model fine-tuning.

## Overview

Traditional fine-tuning updates all model parameters, which is expensive in terms of GPU memory, training time, and storage. LoRA avoids that cost by freezing the pretrained model and injecting small trainable low-rank matrices into attention layers:

```math
W = W_0 + BA
```

Where:

- `W0` is the frozen pretrained weight matrix
- `A` and `B` are low-rank trainable matrices

This enables adaptation with **less than 1% trainable parameters** while preserving most of the original model weights.

## Problem Statement

Fine-tuning large language models with full parameter updates is often impractical for students, researchers, and developers working with limited compute. The goal of this project is to demonstrate that **LoRA makes fine-tuning feasible even for a 1.2B-parameter model trained on a relatively small dataset**.

The target task is **conversational style adaptation**: teaching the model to respond in a specific character or dialogue style rather than changing the full architecture or retraining from scratch.

## Base Model

The foundation model used in this project is:

**LiquidAI LFM2-1.2B**

| Property | Value |
| --- | --- |
| Model type | Causal Language Model |
| Parameters | ~1.17B |
| Tokenizer vocab size | ~65K |
| Framework | Hugging Face Transformers |

The model is loaded directly from Hugging Face and used for both baseline generation and LoRA-based fine-tuning.

## Why LoRA

LoRA is a practical choice for LLM experimentation because it:

- reduces GPU memory requirements
- decreases training time
- lowers the number of trainable parameters
- minimizes catastrophic forgetting
- makes it easier to save and share only adapter weights

Instead of storing a second full model checkpoint, only the learned adapter parameters are saved.

## Dataset

The training data contains approximately **2048 conversational examples**. Each example is formatted as a structured chat exchange:

```text
<|startoftext|><|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
{answer}
<|im_end|>
```

This prompt format teaches the model the structure of multi-turn dialogue and clearly separates user input from assistant output.

The project focuses on **style tuning**

## Training Strategy

The fine-tuning pipeline follows a standard LLM adaptation workflow:

```text
Dataset
   ↓
Prompt Template
   ↓
Tokenization
   ↓
LoRA Injection
   ↓
Fine-Tuning
```

### Training Steps

1. Load the pretrained tokenizer and model
2. Format each example using the chat template
3. Tokenize questions and answers
4. Inject LoRA adapters using the PEFT library
5. Train only the LoRA parameters

### Forward Pass and Loss

During training:

- the model outputs logits for next-token prediction
- predictions are compared with the ground-truth next tokens
- cross-entropy loss is used for optimization

An **answer mask** is applied so that the loss is computed only on the assistant response tokens, not on the user question tokens. This focuses learning on response generation behavior.

### Training Loop

The loop follows the standard deep learning pattern:

1. Sample a batch from the dataset
2. Format the batch into question-answer prompts
3. Tokenize the text
4. Run the forward pass
5. Compute loss
6. Backpropagate gradients
7. Update LoRA parameters

## Training Configuration

| Parameter | Value |
| --- | --- |
| Base Model | LFM2-1.2B |
| Training Method | LoRA |
| Dataset Size | ~2048 samples |
| Task | Conversational style generation |
| Framework | PyTorch + Hugging Face |

Only the **LoRA adapter weights** are saved after training, not the full model checkpoint.

## Inference

A simple chatbot function is used to test the fine-tuned model:

```python
def chat(question, max_new_tokens=32, temperature=0.7):
```

The inference process:

1. Formats the input with the chat template
2. Tokenizes the prompt
3. Generates output tokens with the fine-tuned model
4. Decodes tokens back into readable text

This provides a lightweight way to compare model behavior before and after fine-tuning.

## Results

The fine-tuned model shows:

- improved stylistic consistency
- better adherence to conversational formatting
- stronger alignment with the intended response style

Despite the small training dataset, LoRA enables meaningful adaptation and demonstrates that useful LLM customization is possible without expensive full fine-tuning.

## Evaluation Extension

This project also explores a second stage: **evaluating style-tuned outputs with an LLM judge**.

### Yoda-Style Task

In the evaluation section, the model is further adapted to produce **Yoda-style responses**, which are harder to emulate because they involve:

- unusual word order
- distinct grammar patterns
- stylistic variability

### LLM-as-a-Judge

Evaluating style transfer in language generation is difficult because the quality of an answer is often subjective. To address this, the project uses an **LLM-as-a-Judge** approach, where a stronger external model evaluates the quality of generated responses.

The judge LLM receives:

- a system prompt describing the evaluation criteria
- the generated response
- instructions to score style adherence and explain its reasoning

Suggested judge model:

- **Gemini 2.5 Flash** via OpenRouter

## Key Libraries

| Library | Purpose |
| --- | --- |
| Transformers | Load and run the pretrained LLM |
| PEFT | LoRA-based parameter-efficient fine-tuning |
| PyTorch | Training framework |
| Datasets | Dataset loading and preprocessing |
| MIT Deep Learning utilities | Training helpers |

## Key Takeaways

- Parameter-efficient fine-tuning makes billion-parameter LLM adaptation practical on limited hardware
- LoRA dramatically reduces the number of trainable parameters
- Small datasets can still produce useful style specialization
- Chat-template formatting matters for conversational behavior
- LLM-as-a-Judge provides a scalable way to assess stylistic quality

## Repository Structure

```text
.
├── llm-fine-tuning-using-lora.ipynb
└── README.md
```  
