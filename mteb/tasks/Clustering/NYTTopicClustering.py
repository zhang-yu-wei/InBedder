from ...abstasks.AbsTaskClustering import AbsTaskClustering


class NYTTopicClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "NYTTopicClustering",
            "description": (
                "Clustering news based on topic."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "What is the topic of news?",
            # "data_path": "InstructEmbedBench/results/nyt_topic.json",
            "hf_hub_name": "BrandonZYW/NYTClustering",
            "config_name": "topic",
            "eval_splits": ["test"],
            "revision": "462fb544c8993a0421fd969a7a0f19f1fbd975bf",
            "correct_instructions": [
                "I'm curious, what is the article about?",
                "Could you tell me what subject the article is focused on?",
                "Can you inform me about the central topic covered in the article?",
                "What is the main theme of the news article?",
                "What subject matter does this news article address?",
                "Could you summarize the key topic of the article?",
                "I'd like to know, what is the focus of this news piece?",
                "Can you describe the primary issue discussed in the news article?",
                "What is the core topic the article is centered on?",
                "What specific theme is the article dealing with?"
            ],
            "implicit_instructions": [
                "I'd be interested to hear your thoughts on what the article covers.",
                "What's the latest in the news that's caught your attention?",
                "Have you come across any interesting headlines lately?",
                "What's making the headlines in today's paper?",
                "I'm curious about the current news trends; what's being discussed?",
                "What are journalists focusing on these days?",
                "Any significant stories in the news you think I should be aware of?",
                "I'm looking to catch up on the news; what are the hot topics?",
                "What's the buzz in the media right now?",
                "I heard there's an interesting piece in the news; can you fill me in?"
            ],
            "incorrect_instructions": [
                "Could you tell me which publication or journalist wrote this article?",
                "What kind of viewpoints or analyses does the article present?",
                "How does this article relate to or impact current events or public opinion?",
                "Does the article provide any historical context or background information?",
                "Are there any significant data or statistics mentioned in the article?",
                "Does the article feature any interviews or notable quotes from people?",
                "What region or area does this news article focus on?",
                "Does the article discuss any specific policies or legislation?",
                "Does the article make any predictions or speculate about future events?",
                "What has been the public or reader reaction to this news piece?"
            ]
        }
