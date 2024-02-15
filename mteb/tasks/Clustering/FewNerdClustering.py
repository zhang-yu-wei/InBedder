from ...abstasks.AbsTaskClustering import AbsTaskClustering


class FewNerdClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FewNerdClustering",
            "description": (
                "Clustering entity types."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "Here is a sentence. Please tell me the type of the specified entity appended after the sentence.",
            "data_path": "InstructEmbedBench/datasets/few_nerd_nat/converted_data.json",
            "correct_instructions": [
                "After reading the provided sentence, identify and describe the type of the entity that is mentioned at the end.",
                "Examine this sentence and then specify what category the highlighted entity at its conclusion belongs to.",
                "Review the given sentence and determine the nature of the entity that follows.",
                "Please analyze the sentence below and inform me about the classification of the entity that is appended.",
                "Upon reading the sentence, your task is to identify and state the type of the entity mentioned last.",
                "Look at the sentence provided and then articulate the kind of entity that is added at the end.",
                "Read the following sentence and then describe the category of the specified entity that appears at its end.",
                "Consider the sentence presented and clarify the type of the entity that concludes it.",
                "Observe the sentence below and then specify the nature of the entity that is attached at the end.",
                "Interpret the given sentence and then explain the classification of the entity that is mentioned in its final part."
            ],
            "implicit_instructions": [
                "Take a look at this sentence and share your thoughts on the entity mentioned at the end.",
                "Consider the sentence below and let's discuss the nature of what's described last.",
                "Reflect on the provided sentence and offer insights into the entity concluding it.",
                "Examine this sentence closely and hint at the classification of the final entity.",
                "Observe the sentence and ponder the type of the entity it concludes with.",
                "Read through this sentence and casually mention your understanding of the last mentioned entity.",
                "Peruse the sentence and subtly indicate the category of the entity at its end.",
                "Absorb the content of this sentence and allude to what you think the final entity represents.",
                "Analyze this sentence in a relaxed manner and infer the type of the appended entity.",
                "Take in the sentence provided and give a gentle nod towards the nature of its concluding entity."
            ],
            "incorrect_instructions": [
                "Read the sentence and identify the primary action verb used within it.",
                "Please summarize the main idea expressed in the given sentence in your own words.",
                "Examine this sentence and list any adjectives that appear, explaining their impact on the sentence's tone.",
                "Focus on the sentence structure and explain whether it's a simple, compound, complex, or compound-complex sentence.",
                "Determine the mood conveyed by the sentence and describe how it's achieved through word choice or syntax.",
                "Identify any figurative language used in the sentence, such as metaphors or similes, and explain their effect.",
                "Analyze the sentence for its use of tense and discuss how this affects the overall meaning.",
                "Point out any instances of passive voice in the sentence and suggest ways to rewrite them in active voice.",
                "Look for any subordinate clauses in the sentence and explain their role in contributing to the overall message.",
                "Examine the sentence for punctuation usage and discuss how it influences the rhythm or clarity of the sentence."
            ]
        }
