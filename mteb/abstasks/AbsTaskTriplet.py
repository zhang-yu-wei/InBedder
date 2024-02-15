import numpy as np
import tqdm
import json
import os

from ..evaluation.evaluators import TripletEvaluator
from .AbsTask import AbsTask


class AbsTaskTriplet(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data(kwargs["data_path"])
        
        disable_instruction = kwargs['disable_instruction']
        pattern = kwargs['pattern']
        params = {}
        params.update(kwargs)
        # a list of words to be filtered in the generation side
        params['filter_words'] = kwargs['filter_words'][self.description['name']] if 'filter_words' in kwargs and kwargs['filter_words'] is not None and self.description['name'] in kwargs['filter_words'] else None
    
        results = {}
        for goal_driven_triplets in self.dataset:
            params['cache_dir'] = os.path.join(kwargs['cache_dir'], f"{self.description['name']}_{goal_driven_triplets['Goal']}") if 'cache_dir' in kwargs and kwargs['cache_dir'] is not None else None

            description = goal_driven_triplets["Description"]
            triplets = goal_driven_triplets["Triplets"]

            if disable_instruction:
                evaluator = TripletEvaluator(anchor=[t[0] for t in triplets],
                                             positive=[t[1] for t in triplets],
                                             negative=[t[2] for t in triplets], **params)
            else:
                anchor = [pattern.replace("{input}", t[0]).replace("{instruction}", description) for t in triplets]
                positive = [pattern.replace("{input}", t[1]).replace("{instruction}", description) for t in triplets]
                negative = [pattern.replace("{input}", t[2]).replace("{instruction}", description) for t in triplets]
                evaluator = TripletEvaluator(anchor=anchor, positive=positive, negative=negative, **params)
            
            metrics = evaluator(model)
            for k in metrics:
                results[f"{goal_driven_triplets['Goal']}_{k}"] = metrics[k]
        return results