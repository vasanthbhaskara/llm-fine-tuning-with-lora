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

The project focuses on **style tuning**, including examples such as:

- leprechaun-style responses
- Yoda-style responses for evaluation experiments

## Tokenization

Language models do not work directly on raw text. They operate on tokens, so tokenization is a critical part of the pipeline.

Common tokenization approaches include:

- **Word-based tokenization**: splits text into words, but creates large vocabularies and handles unseen words poorly
- **Character-based tokenization**: uses a very small vocabulary, but produces long sequences and loses semantic structure
- **Subword tokenization**: breaks text into frequent subword units and balances vocabulary size with sequence length

This project uses the **LFM2 tokenizer**, which is based on **Byte Pair Encoding (BPE)**. BPE is widely used in modern LLMs because it balances:

- vocabulary size
- sequence length
- unknown word handling

The tokenizer supports:

- `encode()` to convert text into tokens
- `decode()` to convert tokens back into text

Together with the chat template, the tokenizer builds prompts that the model can process for both training and inference.

## How the Model Generates Text

LFM2 is trained using **causal language modeling**, also called **autoregressive next-token prediction**.

At each step, the model estimates:

```text
P(token_t | token_1, token_2, ..., token_t-1)
```

Text generation works as follows:

1. Feed a prompt into the model
2. Predict the probability distribution for the next token
3. Select or sample the next token
4. Append it to the sequence
5. Repeat until the response is complete

Rather than manually implementing that loop, this project uses:

```python
model.generate()
```

to automatically generate multiple tokens during inference.

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
   ↓
Adapter Export
```

### Training Steps

1. Load the pretrained tokenizer and model
2. Format each example using the chat template
3. Tokenize questions and answers
4. Inject LoRA adapters using the PEFT library
5. Train only the LoRA parameters
6. Save the trained adapter weights

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

## Saving the Adapter

After training, the LoRA adapter can be exported with:

```python
model.save_pretrained("/path/to/lora_adapter")
```

These saved adapters can later be:

- reloaded on top of the same base model
- merged into the base model for inference
- shared without distributing a full fine-tuned checkpoint

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

### Evaluation Stack

The evaluation workflow uses **Comet Opik** to define and track custom metrics. A custom metric scores outputs based on how closely they match Yoda-style language patterns.

The project compares:

- negative control: standard English text
- positive control: real Yoda-style dataset text
- model outputs: generated responses from the fine-tuned model

This allows analysis of:

- average style score
- score distributions
- relative quality versus controls

## Key Libraries

| Library | Purpose |
| --- | --- |
| Transformers | Load and run the pretrained LLM |
| PEFT | LoRA-based parameter-efficient fine-tuning |
| PyTorch | Training framework |
| Datasets | Dataset loading and preprocessing |
| MIT Deep Learning utilities | Training helpers |
| Comet Opik | Evaluation and experiment tracking |

## Key Takeaways

- Parameter-efficient fine-tuning makes billion-parameter LLM adaptation practical on limited hardware
- LoRA dramatically reduces the number of trainable parameters
- Small datasets can still produce useful style specialization
- Chat-template formatting matters for conversational behavior
- LLM-as-a-Judge provides a scalable way to assess stylistic quality

## Future Improvements

Potential next steps include:

- Retrieval-Augmented Generation (RAG)
- larger and more diverse conversational datasets
- automated evaluation with BERTScore or stronger judge prompts
- deployment as a Streamlit or Gradio chatbot
- adapter merging for faster inference
- hyperparameter sweeps for LoRA rank, alpha, and dropout

## Repository Structure

```text
.
├── llm-fine-tuning-using-lora.ipynb
├── LLM_Finetuning_Solution_mkd.ipynb
└── README.md
```

If adapter weights are exported locally, the project can also include:

```text
lora_adapter/
```

## Portfolio Value

This project demonstrates practical experience with:

- LLM fine-tuning workflows
- parameter-efficient training methods
- prompt formatting for conversational models
- Hugging Face and PEFT tooling
- evaluation design for generative AI systems

It is a strong portfolio project because it combines **model adaptation, inference, and evaluation** in a compact but realistic applied LLM pipeline.

## References

- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*
- Hugging Face Transformers
- PEFT Library
- Liquid AI LFM2 model
- OpenRouter
- Comet Opik
