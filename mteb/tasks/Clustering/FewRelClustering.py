from ...abstasks.AbsTaskClustering import AbsTaskClustering


class FewRelClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FewRelClustering",
            "description": (
                "Clustering relation types between two entities."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "Here is a sentence. Please tell me the relation type between two specified entities appended after the sentence.",
            "data_path": "InstructEmbedBench/datasets/few_rel_nat/converted_data.json",
            "correct_instructions": [
                "This statement is provided for your analysis. Identify and describe the type of relationship that exists between the two entities mentioned at the end of the sentence.",
                "Examine the following sentence and determine the nature of the connection between the two entities specified in the appendage.",
                "After reading the sentence given, please elucidate on the type of association present between the two named entities that follow.",
                "Review this sentence and subsequently identify the kind of link or interaction between the two entities that are highlighted at its conclusion.",
                "Consider the content of this sentence, and then clarify the relationship type existing between the two designated entities mentioned thereafter.",
                "In the sentence provided, analyze and state the type of relationship observed between the two specific entities added at the end.",
                "Upon examining the given sentence, please indicate and explain the nature of the relationship between the two entities listed subsequently.",
                "Here's a sentence for you to assess. Identify the kind of bond or connection that exists between the two entities specified right after the sentence.",
                "Read the following sentence and then articulate the type of relationship that is present between the two entities mentioned in the subsequent addendum.",
                "Observe the sentence provided and determine the relationship type between the two particular entities that are mentioned at its end."
            ],
            "implicit_instructions": [
                "Take a look at this sentence and consider the connection between the two entities mentioned afterwards. What do you think links them?",
                "Read the provided sentence and reflect on how the two entities that follow might be related to each other.",
                "Examine the sentence closely. What are your thoughts on the relationship of the two names you'll find at the end?",
                "Explore the dynamics in the sentence given, particularly focusing on the two entities introduced at its close.",
                "After reading the sentence, share your insights about how the two subsequent entities might interact or connect.",
                "Ponder over this sentence and the two entities that come after. How would you describe their connection?",
                "Consider the context of the sentence and think about the two entities that follow. What kind of bond do they share?",
                "Look into the sentence and its conclusion with two specific entities. What relationship do you perceive between them?",
                "Reflect on the sentence and the two names that come after. In what way do you think they are related?",
                "Peruse the sentence and observe the two appended entities. Can you infer a relationship between them?"
            ],
            "incorrect_instructions": [
                "Review this paragraph and summarize its main idea in your own words.",
                "Look at the list of words provided and create a short story that includes all of them.",
                "Examine the sentence given and identify any grammatical errors or areas for improvement.",
                "Read the passage and then write a brief analysis discussing its tone and style.",
                "Consider the provided data and create a graph or chart that best represents the information.",
                "Analyze this poem for its use of metaphor and imagery, and explain their significance.",
                "Translate the following sentence into another language you are familiar with.",
                "Evaluate the argument presented in this text and write a counterargument.",
                "Study the historical event described here and list its major causes and effects.",
                "Observe the sequence of events in this narrative and predict what might happen next."
            ]
        }
