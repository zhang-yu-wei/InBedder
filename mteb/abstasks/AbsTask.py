import random
from abc import ABC, abstractmethod

import datasets
import numpy as np
import torch
import json


class AbsTask(ABC):
    def __init__(self, seed=42, **kwargs):
        self.dataset = None
        self.data_loaded = False
        self.is_multilingual = False
        self.is_crosslingual = False
        self.save_suffix = kwargs.get("save_suffix", "")

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

    def load_data(self, **kwargs):
        """
        Load dataset from HuggingFace hub
        """
        if self.data_loaded:
            return

        # TODO: add split argument
        if "data_path" in self.description:
            with open(self.description["data_path"], 'r') as f:
                self.dataset = json.load(f)
        else:
            config_name = self.description.get("config_name", None)
            if config_name is not None and isinstance(config_name, list):
                self.dataset = []
                for cn in config_name:
                    self.dataset.append(datasets.load_dataset(
                        self.description["hf_hub_name"], revision=self.description.get("revision", None),
                        name=cn
                    ))
            else:
                self.dataset = datasets.load_dataset(
                    self.description["hf_hub_name"], revision=self.description.get("revision", None),
                    name=config_name
                )
        self.data_loaded = True
    
    def _add_instruction(self, pattern, sentences, lang=None, instructions=None):
        """
        Add instructions into text (might not always be before the text)
        
        allow for more prompts - instructions
        """
        assert "{input}" in pattern and "{instruction}" in pattern

        if instructions is not None:
            return [pattern.replace("{input}", sentence).replace("{instruction}", instruction) for sentence, instruction in zip(sentences, instructions)]
        else:
            assert 'instruction' in self.description
            
            if isinstance(self.description['instruction'], dict):
                instruction = self.description['instruction'][lang]
            elif isinstance(self.description['instruction'], str):
                instruction = self.description['instruction']
            else:
                raise ValueError
            return [pattern.replace("{input}", sentence).replace("{instruction}", instruction) for sentence in sentences]
    
    @property
    @abstractmethod
    def description(self):
        """
        Returns a description of the task. Should contain the following fields:
        name: Name of the task (usually equal to the class name. Should be a valid name for a path on disc)
        description: Longer description & references for the task
        type: Of the set: [sts]
        eval_splits: Splits used for evaluation as list, e.g. ['dev', 'test']
        main_score: Main score value for task
        instruction: Instruction to be added into text
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model, split="test"):
        """
        Evaluates a Sentence Embedding Model on the task.
        Returns a dict (that can be serialized to json).
        :param model: Sentence embedding method. Implements a encode(sentences) method, that encodes sentences
        and returns a numpy matrix with the sentence embeddings
        :param split: Which datasplit to be used.
        """
        raise NotImplementedError
