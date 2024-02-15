from ...abstasks.AbsTaskClustering import AbsTaskClustering


class TwentyNewsgroupsClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "TwentyNewsgroupsClustering",
            "hf_hub_name": "mteb/twentynewsgroups-clustering",
            "description": "Clustering of the 20 Newsgroups dataset (subject only).",
            "reference": "https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html",
            "type": "Clustering",
            "category": "s2s",
            "eval_splits": ["test"],
            "eval_langs": ["en"],
            "main_score": "v_measure",
            "revision": "6125ec4e24fa026cec8a478383ee943acfbd5449",
            "instruction": "Categorize the topic of the news article.",
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
