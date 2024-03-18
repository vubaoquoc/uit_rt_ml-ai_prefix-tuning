# Prefix-Tuning for GPT-2 Model Tuning with WebNLG Data
- **Model training Colab Demo**. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1aAkIFPGuwlTbpuW1XpVH4NipG6guQ7mj/view?usp=sharing)
- **Evaluation Colab Demo**. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1aAkIFPGuwlTbpuW1XpVH4NipG6guQ7mj/view?usp=sharing)

This repository serves as a demonstration of utilizing prefix-tuning to fine-tune GPT-2 models using WebNLG data. The project is developed as part of the coursework for the "Machine Learning and Artificial Intelligence" course at the University of Information Technology, Vietnam National University Ho Chi Minh City.

## Overview

Prefix-tuning is a method proposed for efficient and effective fine-tuning of large language models like GPT-2. It involves adding task-specific prefixes to the input during fine-tuning, which guides the model to produce more contextually relevant outputs for the given task. In this project, we leverage prefix-tuning to fine-tune a GPT-2 model on the WebNLG dataset.

## Repository Structure

The repository structure is as follows:

- ``: Contains the WebNLG dataset for training and evaluation.
- `demo-notebook/`: Source code for fine-tuning the GPT-2 model using prefix-tuning.
- `gpt2/` and `transformers/`: Code and resources referenced from Transformer, GPT-2, and Prefix-tuning repositories.
- `gpt2/webnlg_models/`: Model tuning (1 epoch, 5 epochs and 3, 5 prefix length)
- `output/contrast_LM/transformers/examples/text-generation/`: Output prediction and evaluation

## Set up env python < 3.7

``cd transformer; pip install -e .``

-----------------------------------------------------
## Train via prefix-tuning:

```python
cd gpt2;

python train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```


```python
cd seq2seq; 

python train_bart.py --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 16  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```


Other baseline approaches 

```
cd gpt2;

python train_e2e.py --tuning_mode {finetune/adaptertune} --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```

```
cd seq2seq;

python train_e2e.py --tuning_mode finetune --epoch 5 --learning_rate 0.00005 --mode webnlg --bsz 5 --seed 101
```
-----------------------------------------------------

## Decode:

```python
cd gpt2;

python gen.py {data2text/webnlg/...} yes test {checkpoint_path} no
```


```python
cd seq2seq; 

python train_bart.py --mode xsum --do_train no --prefix_model_path {checkpoint_path} --preseqlen {same as training} --mid_dim {same as training}
```

-----------------------------------------------------

## References

- [Transformer](https://github.com/huggingface/transformers)
- [GPT-2](https://github.com/openai/gpt-2)
- [Prefix-tuning](https://github.com/microsoft/PrefixTuning)

