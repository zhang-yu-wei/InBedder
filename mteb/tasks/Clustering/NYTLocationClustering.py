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
            "data_path": "InstructEmbedBench/results/nyt_location.json"
        }
