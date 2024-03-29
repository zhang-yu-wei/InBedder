from ...abstasks.AbsTaskClustering import AbsTaskClustering


class FeedbacksClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "FeedbacksClustering",
            "description": (
                "Clustering human feedbacks on machine-generated texts. There are only 3 clusters that are perfectly balanced in order to avoid any biases from imbalanced clustering during evaluation."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "Here is a human feedback on machine-generated text. Categorize the type of reason why the human like or dislike it.",
            # "data_path": "InstructEmbedBench/results/goalex_feedbacks_0-gpt-4-feedbacks_v3_converted.json",
            "hf_hub_name": "BrandonZYW/FeedbacksClustering",
            "eval_splits": ["test"],
            "revision": "365ef3e171ad4dab887b08495b10940a1aecb04b",
            "correct_instructions": [
                "Please classify the human's feedback on this AI-generated text, identifying their reasons for either liking or disliking it.",
                "Examine the human response to the machine-produced text and determine the basis of their approval or disapproval.",
                "Analyze this human feedback on the text generated by the machine and categorize the rationale behind their positive or negative reaction.",
                "Review the human's critique of the AI-created text and identify the factors that contributed to their preference or aversion.",
                "Assess the human's evaluation of the computer-generated text, focusing on understanding the grounds for their liking or disliking it.",
                "Inspect the feedback provided by a human on this machine-composed text, and categorize the reasoning behind their positive or negative opinion.",
                "Sort the human's reaction to the AI-produced text into categories based on why they found it appealing or unappealing.",
                "Interpret the human's appraisal of the machine-generated text, identifying the specific reasons for their satisfaction or dissatisfaction.",
                "Break down the human feedback on the AI-generated text, and systematically categorize the cause of their liking or disliking.",
                "Take the human response to the text created by the machine and delineate the types of reasons behind their favorable or unfavorable view."
            ],
            "implicit_instructions": [
                "Examine this human feedback on AI-generated text and explore what might have influenced their opinion.",
                "Consider the provided human response to the machine text and subtly identify what swayed their views.",
                "Reflect on the reasons behind the human's reaction to this computer-crafted text, focusing on underlying factors.",
                "Observe the human's impressions of the AI text and deduce the subtle nuances that shaped their perspective.",
                "Delve into this feedback on machine-generated text and infer the reasons for the human's viewpoint.",
                "Analyze the human response to the text, reading between the lines to understand their sentiment.",
                "Scrutinize the feedback from a human perspective on AI text and intuit what drove their reaction.",
                "Contemplate the human's take on this machine-generated text and ascertain the factors influencing their stance.",
                "Perceive the nuances in the human feedback on the AI text and speculate on what led to their feelings.",
                "Evaluate the human response to this text and piece together the elements that formed their opinion, be it positive or negative.",
            ],
            "incorrect_instructions": [
                "Review this article on sustainable agriculture practices and summarize its key recommendations.",
                "Analyze the latest trends in renewable energy technologies and prepare a brief report on their potential impacts.",
                "Examine the historical significance of the Renaissance period in Europe, focusing on its contributions to art and science.",
                "Conduct a study on consumer behavior in online shopping and identify the top factors that influence purchase decisions.",
                "Create a presentation on the importance of physical exercise in maintaining mental health, using recent studies as references.",
                "Develop a marketing strategy for a new health food product, targeting a specific demographic.",
                "Write a short story set in a futuristic world where artificial intelligence governs society.",
                "Compare and contrast the political systems of two different countries, highlighting their strengths and weaknesses.",
                "Design a survey to gather opinions on public transportation improvements in urban areas.",
                "Investigate the effects of climate change on marine ecosystems and propose measures for conservation."
            ]
        }
