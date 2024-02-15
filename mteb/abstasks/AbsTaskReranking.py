import os
from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask


class AbsTaskReranking(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()
        
        disable_instruction = kwargs['disable_instruction']
        pattern = kwargs['pattern']
        params = {}
        params.update(kwargs)
        # a list of words to be filtered in the generation side
        params['filter_words'] = kwargs['filter_words'][self.description['name']] if 'filter_words' in kwargs and kwargs['filter_words'] is not None and self.description['name'] in kwargs['filter_words'] else None
        params['cache_dir'] = os.path.join(kwargs['cache_dir'], self.description["name"]) if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None else None

        data_split = self.dataset[split]

        if not disable_instruction:
            data_instruct = []
            for datum in data_split:
                new_datum = {}
                new_datum['query'] = self._add_instruction(pattern, datum['query'] if isinstance(datum['query'], list) else [datum['query']])
                new_datum['positive'] = self._add_instruction(pattern, datum['positive'])
                new_datum['negative'] = self._add_instruction(pattern, datum['negative'])
                data_instruct.append(new_datum)
            evaluator = RerankingEvaluator(data_instruct, **params)
        else:
            evaluator = RerankingEvaluator(data_split, **params)

        scores = evaluator(model)

        return dict(scores)
