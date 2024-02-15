from ...abstasks import AbsTaskClassification, MultilingualTask

_LANGUAGES = ["en", "de", "en-ext", "ja"]


class AmazonCounterfactualClassification(MultilingualTask, AbsTaskClassification):
    @property
    def description(self):
        return {
            "name": "AmazonCounterfactualClassification",
            "hf_hub_name": "mteb/amazon_counterfactual",
            "description": (
                "A collection of Amazon customer reviews annotated for counterfactual detection pair classification."
            ),
            "reference": "https://arxiv.org/abs/2104.06893",
            "category": "s2s",
            "type": "Classification",
            "eval_splits": ["validation", "test"],
            "eval_langs": _LANGUAGES,
            "main_score": "accuracy",
            "n_experiments": 10,
            "samples_per_label": 32,
            "revision": "e8379541af4e31359cca9fbcf4b00f2671dba205",
            # llama-7b-chat understands counterfactual, try to let it explain "what is counterfactual"
            "instruction": {
                "en": "Does the following Amazon customer review contains counterfactual statements?\n[SENTENCE]\n\nShort Answer: ",
                "en-ext": "Does the following Amazon customer review contains counterfactual statements?\n[SENTENCE]\n\nShort Answer: ",
                "de": "Enthält die folgende Amazon-Kundenrezension kontrafaktische Aussagen?\n[SENTENCE]\n\nKurze Antwort: ",
                "ja": "次の Amazon カスタマー レビューには反事実的な記述が含まれていますか?\n[SENTENCE]\n\n短い答え："
            }
        }
