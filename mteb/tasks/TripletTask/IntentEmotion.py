from ...abstasks.AbsTaskTriplet import AbsTaskTriplet


class IntentEmotion(AbsTaskTriplet):
    @property
    def description(self):
        return {
            "name": "IntentEmotion",
            "category": "s2s",
            "description": (
                "Goal-driven triplet evaluation based on intent or emotion. Instructions are contained in the data files."
            ),
            "type": "TripletTask",
            "eval_splits": ["test"],
            "data_path": "InstructEmbedBench/results/banking77-gpt-4-triplet_v1_converted.json"
        }
