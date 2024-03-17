# Prefix Tuning
- **Model training Colab Demo**. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UbPCspeZQ9zZ6uAIC5-3gN_rApiN4c1C)
- **Evaluation Colab Demo**. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1M_zZke_MRaaeDS_UzEv4Do282i0rpiKv)

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

For details of the methods and results, please refer to our paper. 

```bibtex
@misc{li2021prefixtuning,
      title={Prefix-Tuning: Optimizing Continuous Prompts for Generation}, 
      author={Xiang Lisa Li and Percy Liang},
      year={2021},
      eprint={2101.00190},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
