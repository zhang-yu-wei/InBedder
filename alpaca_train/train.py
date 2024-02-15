"""
AR train script. Support causal LM, encoder-decoder LM and encoder LM
https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py

TODO:
1. load_model, specify different model types
2. preprocess, customize different preprocessing procedures according to model
"""
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer

os.environ["WANDB_PROJECT"]="LLMEmbAPI"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# Why do we need the prefix?
# PROMPT_DICT = {
#     "prompt_input": (
#         "{instruction}\n\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "{instruction}\n\n### Response:"
#     ),
# }
ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
# immediate answer after the question will help?
QA_PROMPT_DICT = {
    "prompt_input": (
        "### Input:\n{input}\n\n### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_flash_attention_2: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    model_name = tokenizer.name_or_path.lower()
    if 'gpt' in model_name or \
    'opt' in model_name or \
    'llama' in model_name:
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
            label[:source_len] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=labels)
    elif 'bert' in model_name:
        outputs = [s + t for s, t in zip(sources, targets)]
        outputs_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (outputs, sources)]
        output_ids = outputs_tokenized['input_ids']
        input_ids = copy.deepcopy(output_ids)
        for inp_tok, out_tok, source_len in zip(input_ids, output_ids, sources_tokenized["input_ids_lens"]):
            inp_tok[source_len-1:-1] = tokenizer.mask_token_id
            # out_tok[:source_len-1] = IGNORE_INDEX
            # out_tok[-1] = IGNORE_INDEX
            out_tok[inp_tok.ne(tokenizer.mask_token_id)] = IGNORE_INDEX
        return dict(input_ids=input_ids, labels=output_ids)
    elif 't5' in model_name or 'bart' in model_name:
        sources_tokenized, targets_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, targets)]
        return dict(input_ids=sources_tokenized['input_ids'], labels=targets_tokenized['input_ids'])
    else:
        raise NotImplementedError(f"{model_name} not implemented")


def truncate_inputs(data_dict: Dict[str,str], tokenizer: transformers.PreTrainedTokenizer):
    """a proper trunaction is needed for the input
    
    always truncate longest one
    """
    # there will be extra tokens in the prompt + special tokens, here is just a randomly picked number
    num_extra_tokens = 20
    for i in range(len(data_dict)):
        if "input" in data_dict[i] and data_dict[i]["input"] != "":
            ist_toks = tokenizer.tokenize(data_dict[i]['instruction'])
            inp_toks = tokenizer.tokenize(data_dict[i]['input'])
            res_toks = tokenizer.tokenize(data_dict[i]['output'])
            if len(ist_toks + inp_toks + res_toks) > tokenizer.model_max_length - num_extra_tokens:
                max_len_in_trip = max((len(ist_toks), len(inp_toks), len(res_toks)))
                if max_len_in_trip == len(inp_toks):
                    inp_toks = inp_toks[:tokenizer.model_max_length - num_extra_tokens - len(ist_toks + res_toks)]
                    data_dict[i]['input'] = tokenizer.convert_tokens_to_string(inp_toks)
                elif max_len_in_trip == len(res_toks):
                    res_toks = res_toks[:tokenizer.model_max_length - num_extra_tokens - len(ist_toks + inp_toks)]
                    data_dict[i]['output'] = tokenizer.convert_tokens_to_string(res_toks)
                else:
                    ist_toks = ist_toks[:tokenizer.model_max_length - num_extra_tokens - len(res_toks + inp_toks)]
                    data_dict[i]['instruction'] = tokenizer.convert_tokens_to_string(ist_toks)
        else:
            ist_toks = tokenizer.tokenize(data_dict[i]['instruction'])
            res_toks = tokenizer.tokenize(data_dict[i]['output'])
            if len(ist_toks + res_toks) > tokenizer.model_max_length - num_extra_tokens:
                max_len_in_pair = max((len(ist_toks), len(res_toks)))
                if max_len_in_pair == len(res_toks):
                    res_toks = res_toks[:tokenizer.model_max_length - num_extra_tokens - len(ist_toks)]
                    data_dict[i]['output'] = tokenizer.convert_tokens_to_string(res_toks)
                else:
                    ist_toks = ist_toks[:tokenizer.model_max_length - num_extra_tokens - len(res_toks)]
                    data_dict[i]['instruction'] = tokenizer.convert_tokens_to_string(ist_toks)
    return data_dict


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        model_name = tokenizer.name_or_path
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)[:100]
        list_data_dict = utils.jload(data_path)
        logging.warning("Truncating inputs...")
        list_data_dict = truncate_inputs(list_data_dict, tokenizer)

        logging.warning("Formatting inputs...")
        if "qa" in data_path:
            PROMPT_DICT = QA_PROMPT_DICT
        else:
            PROMPT_DICT = ALPACA_PROMPT_DICT
        # if 'alpaca' in data_path:
        #     PROMPT_DICT = QA_PROMPT_DICT
        # else:
        #     PROMPT_DICT = ALPACA_PROMPT_DICT
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        logging.warning(f"First source text:\n{sources[0]}")
        # adding eos_token for calculating loss?
        if 'gpt' in model_name or \
        'opt' in model_name or \
        'llama' in model_name:
            targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        else:
            targets = [f"{example['output']}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def load_model(model_name_or_path, cache_dir, use_flash_attention_2):
    """
    load different types of models
    """
    model_lower = model_name_or_path.lower()
    if 'gpt' in model_lower or \
    'opt' in model_lower or \
    'llama' in model_lower:
        # decoder-only models follow original implementations
        return transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_flash_attention_2=use_flash_attention_2
        )
    elif 'bert' in model_lower:
        # encoder models decode targets by masking all the outputs
        return transformers.AutoModelForMaskedLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_flash_attention_2=use_flash_attention_2
        )
    elif 't5' in model_lower or 'bart' in model_lower:
        # encoder-decoder models use seq2seq training
        return transformers.AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            use_flash_attention_2=use_flash_attention_2
        )
    else:
        raise NotImplementedError("The model is not implemented currently.")

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = load_model(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2=model_args.use_flash_attention_2
    )

    # decoder-only models can pad on right
    # encoder-only models can pad on right
    # T5 encoder can pad on both right and left
    # BART encoder must be padded on right
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()