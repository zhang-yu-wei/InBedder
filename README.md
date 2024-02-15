# InBedder: Instruction-following Text Embedder

This repository contains the code and pre-trained models for our paper [Answer is All You Need: Instruction-following Text Embedding via Answering the Question]().

We introduce **InBedder**🛌, a text embedder that is designed to follow instructions. Instruction-following text embedder can capture characteristics of texts specified by user instructions. InBedder offers a novel viewpoint that treats the instruction as a _question_ about the input text and encodes the _expected answers_ to obtain the representation accordingly. We show that InBedder is aware of instructions with different evaluation tasks.

**************************** **Updates** ****************************

* 02/14/2024: We released [our paper](), [code](), [project page]() and [checkpoint](). Check them out!

## Quick Links

## Installation
Follow the following steps to install InBedder.
```bash
conda create -n inbedder python=3.9
conda activate inbedder
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
python -m pip install flash-attn --no-build-isolation
```

## Getting Started

### Load Model

### The `encode` function

## Model List

We released a series of InBedder checkpoints with different sizes. You can easily load these models with huggingface. 
|              Model              | Avg. Score |
|:-------------------------------|:--------:|
| [Llama-2-7b-InBedder]() |  |
| [OPT-2.7b-InBedder]()   |  |
| [OPT-1.3b-InBedder]()   |  |
| [OPT-350m-InBedder]()   |  |

## Use Case
We show how to use InBedder for personalized clustering.

## Training
### Data

### Train InBedder

## Evaluation
### Data

### Evaluation Code

## Bugs or questions?
If you have any question related to the code or the paper, feel free to email Yuwei (`yuz163@ucsd.edu`) and Letian (`lepeng@ucsd.edu`).

## Citation
If you find our work helpful, please cite us:

```bibtex

```
