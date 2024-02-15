import os
import logging

from ..evaluation.evaluators import STSEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)

class AbsTaskSTS(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.description["min_score"]

    @property
    def max_score(self):
        return self.description["max_score"]

    def load_data(self, **kwargs):
        super().load_data(**kwargs)
        if 'test' not in self.dataset:
            self.dataset = {'test': self.dataset}

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                logger.info(f"Task: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, lang, **kwargs)
        else:
            logger.info(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, None, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, lang=None, **kwargs):
        def normalize(x):
            return (x - self.min_score) / (self.max_score - self.min_score)
        disable_instruction = kwargs['disable_instruction']
        pattern = kwargs['pattern']
        if disable_instruction:
            # sentence1 = data_split["sentence1"]
            # sentence2 = data_split["sentence2"]
            sentence1 = [d['sentence1'] for d in data_split]
            sentence2 = [d['sentence2'] for d in data_split]
        else:
            if 'instruction' in data_split[0]:
                instructions = [d['instruction'] for d in data_split]
            else:
                instructions = None
            sentence1 = self._add_instruction(pattern, [d['sentence1'] for d in data_split], lang, instructions=instructions)
            sentence2 = self._add_instruction(pattern, [d['sentence2'] for d in data_split], lang, instructions=instructions)
        
        params = {}
        params.update(kwargs)
        params['cache_dir'] = os.path.join(kwargs['cache_dir'], self.description["name"]) if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None else None
        # a list of words to be filtered in the generation side
        params['filter_words'] = kwargs['filter_words'][self.description['name']] if 'filter_words' in kwargs and kwargs['filter_words'] is not None and self.description['name'] in kwargs['filter_words'] else None

        normalized_scores = list(map(normalize, [d['score'] for d in data_split]))
        evaluator = STSEvaluator(sentence1, sentence2, normalized_scores, **params)
        metrics = evaluator(model)
        return metrics
