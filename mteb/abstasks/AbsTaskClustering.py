import numpy as np
import tqdm
import os

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask


class AbsTaskClustering(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def load_data(self, **kwargs):
        super().load_data(**kwargs)
        if "data_path" in self.description:
            sentences = []
            labels= []
            for k in self.dataset:
                sentences.extend(self.dataset[k])
                labels.extend([k] * len(self.dataset[k]))
            self.dataset = {'test': [{"sentences": sentences, "labels": labels}]}
        if "text" in self.dataset['test'][0] and "cluster" in self.dataset['test'][0]:
            sentences = []
            labels= []
            for d in self.dataset['test']:
                sentences.append(d['text'])
                labels.append(d['cluster'])
            self.dataset = {'test': [{"sentences": sentences, "labels": labels}]}

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        disable_instruction = kwargs['disable_instruction']
        pattern = kwargs['pattern']
        predefined_instruction = kwargs.get("instruction", None)
        truncate_to_length = kwargs.get("truncate_to_length", None)
        params = {}
        params.update(kwargs)
        # a list of words to be filtered in the generation side
        params['filter_words'] = kwargs['filter_words'][self.description['name']] if 'filter_words' in kwargs and kwargs['filter_words'] is not None and self.description['name'] in kwargs['filter_words'] else None

        v_measures = []
        cluster_components = []
        for i, cluster_set in tqdm.tqdm(enumerate(self.dataset[split]), total=len(self.dataset[split]), desc="Clustering"):
            params['cache_dir'] = os.path.join(kwargs['cache_dir'], f'{self.description["name"]}_experiment_{i}') if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None else None
            if truncate_to_length is not None:
                sentences = [' '.join(sent.split(" ")[-truncate_to_length:]) for sent in cluster_set["sentences"]]
            else:
                sentences = cluster_set["sentences"]
            
            if disable_instruction:
                evaluator = ClusteringEvaluator(sentences, cluster_set["labels"], **params)
            else:
                sentences_instruct = self._add_instruction(pattern, sentences, instructions=[predefined_instruction] * len(sentences) if predefined_instruction is not None else predefined_instruction)
                evaluator = ClusteringEvaluator(sentences_instruct, cluster_set["labels"], **params)
            metrics = evaluator(model)
            v_measures.append(metrics["v_measure"])
            cluster_components.append(metrics["cluster_components"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        if "data_path" in self.description:
            return {"v_measure": v_mean, "v_measure_std": v_std, "cluster_components": cluster_components}
        else:
            return {"v_measure": v_mean, "v_measure_std": v_std}
