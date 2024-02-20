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
            # "data_path": "InstructEmbedBench/results/banking77-gpt-4-triplet_v1_converted.json"
            "hf_hub_name": "BrandonZYW/IntentEmotion",
            "revision": "d0101c15083166b77371cf542c4045decc0a6e92",
            "config_name": ["emotion", "intent"],
            "instruction": ['How does the customer feel?', 'What does the customer need?'],
        }
