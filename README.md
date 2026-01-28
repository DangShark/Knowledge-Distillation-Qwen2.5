# Knowledge Distillation for Mathematical Reasoning
## Distilling Qwen2.5-7B into Qwen2.5-0.5B on GSM8K

## Overview
This project studies knowledge distillation (KD) as a practical method to transfer
mathematical reasoning capability from a large language model (LLM) to a compact model.
The goal is to improve reasoning performance while significantly reducing computational
and memory requirements.

We distill Qwen2.5-7B-Instruct (teacher) into Qwen2.5-0.5B-Instruct (student) and evaluate
the distilled model on the GSM8K benchmark for grade-school mathematical reasoning.

The proposed approach combines Chain-of-Thought (CoT) supervision with token-level
Top-K logit distillation, leading to clear improvements over standard instruction tuning.

---

## Objectives
- Compress a large reasoning-capable LLM into a smaller student model
- Preserve structured, multi-step mathematical reasoning
- Improve GSM8K accuracy under strict model capacity constraints
- Demonstrate the effectiveness of token-level knowledge transfer

---

## Key Contributions
- A two-stage distillation framework for reasoning-centric LLMs
- Integration of CoT-based black-box distillation and white-box Top-K logit KD
- Empirical evidence that transferring token-level uncertainty improves reasoning
- Efficient distillation using only Top-K teacher logits instead of full vocabulary

---

## Methodology

### Stage 0: Chain-of-Thought Generation
- Use the teacher model to generate multiple CoT solutions per GSM8K problem
- Enforce a strict output format with a final numeric answer marked by `####`
- Filter solutions by exact numeric correctness
- Retain one verified CoT per problem for training

### Stage 1: Black-Box Distillation (Supervised Fine-Tuning)
- Train the student using supervised fine-tuning on verified teacher CoT solutions
- Apply loss only on completion tokens (prompt tokens are masked)
- Objective: learn structured reasoning behavior from the teacher

### Stage 2: White-Box Distillation (Top-K Logit Distillation)
- Extract teacher Top-M logits offline for each decoding step
- Align teacher and student vocabularies at token level
- Retain Top-K aligned tokens and discard the rest
- Minimize KL divergence between teacher and student Top-K distributions
- Combine SFT loss and KD loss with a linear warm-up schedule

The final training objective is:
L = L_SFT + α · L_KD

---

## Experimental Setup

### Dataset
- GSM8K (grade-school mathematical reasoning)

### Models
- Teacher: Qwen2.5-7B-Instruct
- Student: Qwen2.5-0.5B-Instruct

### Training Configuration
Stage 1 (SFT):
- Epochs: 3
- Learning rate: 2e-5
- Effective batch size: 32

Stage 2 (KD):
- Epochs: 3
- Learning rate: 5e-6
- Effective batch size: 32
- Top-M = 25, Top-K = 10
- Distillation temperature T = 2.0
- KD weight α = 0.35 (linear warm-up)

Evaluation is performed using greedy decoding on the GSM8K test set.

---

## Results

| Model                         | GSM8K Accuracy |
|------------------------------|---------------|
| Qwen2.5-0.5B (Base)          | 41.6%         |
| Qwen2.5-0.5B (Instruction)   | 49.6%         |
| Qwen2.5-0.5B (Distilled)     | 52.2%         |
| Qwen2.5-7B (Teacher)         | 85.4%         |

The distilled student model outperforms instruction tuning alone while being
significantly smaller than the teacher model.

---

## Analysis
- CoT supervision stabilizes multi-step reasoning trajectories
- Top-K token-level distillation transfers the teacher’s uncertainty structure
- Most teacher probability mass is concentrated in a small set of tokens
- Selective Top-K distillation preserves informative signals efficiently
- Improvements reflect genuine reasoning gains, not just formatting effects

---

## Limitations
- Evaluation focuses on final-answer accuracy, not reasoning faithfulness
- Filtering by numeric correctness may bias training toward easier examples
- Optimal Top-K value may vary across tasks and domains

---

## Future Work
- Ablation studies on CoT supervision and Top-K distillation
- Evaluation on additional reasoning benchmarks
- Adaptive selection of K for high-entropy decoding steps
- Improved vocabulary alignment strategies

---

## Suggested Project Structure
project-root/
├── data/
│ ├── gsm8k_train.jsonl
│ └── gsm8k_test.jsonl
├── cot_generation/
├── stage1_sft/
├── stage2_kd/
├── evaluation/
├── scripts/
└── README.md

---

## References
- Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
- Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", 2023
- Gu et al., "MiniLLM: Knowledge Distillation of Large Language Models", 2025
- Qwen Team, "Qwen2.5 Technical Report", 2025

---

## Author
Bui Hai Dang  
Vietnam National University – University of Engineering and Technology  
Institute for Artificial Intelligence
