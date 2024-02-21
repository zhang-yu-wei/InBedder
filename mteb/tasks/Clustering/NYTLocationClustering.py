from ...abstasks.AbsTaskClustering import AbsTaskClustering


class NYTLocationClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "NYTLocationClustering",
            "description": (
                "Clustering news based on location."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "Where did the news happen?",
            # "data_path": "InstructEmbedBench/results/nyt_location.json"
            "hf_hub_name": "BrandonZYW/NYTClustering",
            "config_name": "location",
            "eval_splits": ["test"],
            "revision": "462fb544c8993a0421fd969a7a0f19f1fbd975bf",
        }
