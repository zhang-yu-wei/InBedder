from ...abstasks.AbsTaskTriplet import AbsTaskTriplet


class ACLAbstract(AbsTaskTriplet):
    @property
    def description(self):
        return {
            "name": "ACLAbstract",
            "category": "s2s",
            "description": (
                "Search for relevant papers."
            ),
            "type": "TripletTask",
            "eval_splits": ["test"],
            "data_path": "InstructEmbedBench/results/anthology-gpt-4-1106-preview-anthology_search_query_v2_triplets_converted.json"
        }