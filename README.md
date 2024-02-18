<img src="images/InBedder.jpeg" width="300" height="300">

<small>Credit DALL·E 3</small>

# 🛌InBedder: Instruction-following Text Embedder

This repository contains the code, dataset and pre-trained models for our paper [Answer is All You Need: Instruction-following Text Embedding via Answering the Question]().

We introduce **InBedder**🛌, a text embedder that is designed to follow instructions. Instruction-following text embedder can capture characteristics of texts specified by user instructions. InBedder offers a novel viewpoint that treats the instruction as a _question_ about the input text and encodes the _expected answers_ to obtain the representation accordingly. We show that InBedder is aware of instructions with different evaluation tasks.

**************************** **Updates** ****************************

* 02/15/2024: We released [our paper](https://arxiv.org/abs/2402.09642), [code](https://github.com/zhang-yu-wei/InBedder/), [pre-training dataset](https://huggingface.co/datasets/KomeijiForce/Inbedder-Pretrain-Data), [project page]() and [checkpoint](https://huggingface.co/BrandonZYW). Check them out!

## ⚡ Quick Links

## 📦 Installation
Follow the following steps to set up InBedder.
```bash
conda create -n inbedder python=3.9
conda activate inbedder
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

## 🚀 Getting Started

### Load Model

```python
from lm_encoders_hf import CausalLMEncoder

model = CausalLMEncoder(
    model_name_or_path="BrandonZYW/llama-2-7b-InBedder",
    generation_configs={
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 3,
        "do_sample": true
    }
)
```

### Add instructions

```python
pattern = "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"
corpus = [pattern.replace('{input}', s).replace('{instruction}', instruction) for s in corpus]
```

### The `encode` function

```python
embeddins, generations = model.encode(
    corpus,
    batch_size=32,
    cache_dir=None, # useful when you want reuse the embeddings
    return_generations=True # useful if you want to look at your generations
)
```

## 📊 Model List

We released a series of InBedder checkpoints with different sizes. You can easily load these models with huggingface. 
|              Model              | Avg. Score |
|:-------------------------------|:--------:|
| [llama-2-7b-InBedder](https://huggingface.co/BrandonZYW/llama-2-7b-InBedder) | **58.80** |
| [opt-2.7b-InBedder](https://huggingface.co/BrandonZYW/opt-2.7b-InBedder)   | 56.57 |
| [opt-1.3b-InBedder](https://huggingface.co/BrandonZYW/opt-1.3b-InBedder)   | 54.99 |
| [roberta-large-InBedder](https://huggingface.co/BrandonZYW/roberta-large-InBedder)   | 53.06 |

## 💡 Use Case
We show how to use InBedder for personalized clustering.

## 🏋️‍♂️ Training
### Data

### Train InBedder

## ✅ Evaluation
### Data

### Evaluation Code

## 🐞 Bugs or questions?
If you have any question related to the code or the paper, feel free to email Yuwei (`yuz163@ucsd.edu`) and Letian (`lepeng@ucsd.edu`).

## 📑 Citation
If you find our work helpful, please cite us:

```bibtex
@article{DBLP:journals/corr/abs-2402.09642,
  author       = {Letian Peng and
                  Yuwei Zhang and
                  Zilong Wang and
                  Jayanth Srinivasa and
                  Gaowen Liu and
                  Zihan Wang and
                  Jingbo Shang},
  title        = {Answer is All You Need: Instruction-following Text Embedding via Answering the Question},
  journal      = {CoRR},
  volume       = {abs/2402.09642},
  year         = {2023},
  url          = {https://arxiv.org/abs/2402.09642},
  eprinttype    = {arXiv},
  eprint       = {2402.09642},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2402.09642.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
