from ...abstasks.AbsTaskClustering import AbsTaskClustering


class FewEventClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FewEventClustering",
            "description": (
                "Clustering event types."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "Here is a sentence. Please tell me the type of the specified event according to the trigger words appended after the sentence.",
            "data_path": "InstructEmbedBench/datasets/few_event/converted_data.json",
            "correct_instructions": [
                "After reading the provided sentence, identify the type of event based on the trigger words that follow.",
                "Examine this sentence and determine the event type using the appended trigger words.",
                "Please classify the type of event in the sentence using the trigger words given at the end.",
                "Review the sentence given and use the subsequent trigger words to specify the event type.",
                "Analyze the sentence and ascertain the event type based on the trigger words attached afterward.",
                "Consider the sentence provided and identify what type of event is indicated by the trigger words that come after.",
                "Read the sentence and, using the trigger words that follow, tell us what type of event it refers to.",
                "Look at the provided sentence and use the appended trigger words to determine the nature of the event.",
                "Observe the sentence and use the trigger words that are attached below it to classify the event type.",
                "Inspect this sentence and, with the help of the trigger words that follow, discern the type of event it implies.",
            ],
            "implicit_instructions": [
                "Take a look at this sentence and consider what the following trigger words might reveal about the event it describes.",
                "Explore the meaning of this sentence further by reflecting on the trigger words provided at the end.",
                "See if you can infer the type of event in the sentence based on the additional words that follow.",
                "Ponder over the sentence and how the subsequent trigger words might shed light on its event type.",
                "Examine the sentence and think about what the attached trigger words could suggest about the event.",
                "Contemplate the sentence and the trigger words that come after to understand the event's nature.",
                "Delve into this sentence and use the clues from the trigger words following it to grasp the event type.",
                "Consider what the trigger words following this sentence might indicate about the type of event it pertains to.",
                "Reflect on how the appended trigger words might define the event mentioned in the sentence.",
                "Study the sentence and let the trigger words that follow guide you to an understanding of the event type."
            ],
            "incorrect_instructions": [
                "Review this paragraph and summarize its main idea in your own words.",
                "Look at the list of items below and categorize them into groups based on their similarities.",
                "Examine the graph presented and explain the trend it depicts over time.",
                "Read the dialogue between two characters and infer their emotional states from their conversation.",
                "Analyze the poem provided and identify the primary literary devices used by the poet.",
                "Observe the image and write a brief description of the scene it portrays.",
                "Listen to the audio clip and transcribe the spoken words accurately.",
                "Study the historical event described and list the key factors that led to its occurrence.",
                "Examine the mathematical equation and solve for the unknown variable.",
                "Watch the video and then explain the process demonstrated step by step.",
            ]
        }
