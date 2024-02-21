from ...abstasks.AbsTaskClustering import AbsTaskClustering


class RateMyProfClustering(AbsTaskClustering):
    @property
    def description(self):
        return {
            "name": "RateMyProfClustering",
            "description": (
                "Clustering student reviews on courses and professors. There are only 4 clusters that are perfectly balanced in order to avoid any biases from imbalanced clustering during evaluation."
            ),
            "type": "Clustering",
            "category": "s2s",
            "main_score": "v_measure",
            "instruction": "Here is a student review on the course and the professor. Categorize the aspect the student is discussing about.",
            # "data_path": "InstructEmbedBench/results/goalex_rate_my_prof_0-gpt-4-ratemyprof_v1_converted.json",
            "hf_hub_name": "BrandonZYW/RateMyProfClustering",
            "eval_splits": ["test"],
            "revision": "253b8501d191f7c3d3aff3bc27c6d7f7112589ba",
            "correct_instructions": [
                "Examine this student's review of the course and the professor, and identify the specific aspects being discussed.",
                "Please analyze the student's feedback regarding the course and professor, and determine the key topics covered.",
                "Read through the student's evaluation of the course and professor, and classify the main points they are addressing.",
                "Take a look at this student review about the course and instructor, and segregate the areas they are commenting on.",
                "Review this student's comments on the course and professor, and sort out the different aspects they mention.",
                "Go through the student's appraisal of the course and professor, and label the specific subjects they are referring to.",
                "Assess the student's critique on the course and professor, and pinpoint the various aspects they are focusing on.",
                "Examine the student's feedback on the course and professor, and categorize the different elements they discuss.",
                "Inspect this review from a student about the course and professor, and identify the distinct areas they are evaluating.",
                "Peruse the student's observations about the course and professor, and classify the particular aspects they are examining."
            ],
            "implicit_instructions": [
                "Take a moment to reflect on this student's review of the course and professor. What themes do you notice?",
                "As you read through the student's thoughts on the course and instructor, consider what stands out to you.",
                "What aspects do you think this student review of the course and professor focuses on?",
                "Peruse this student's perspective on the course and professor, and think about what they emphasize.",
                "Consider the student's viewpoint on the course and professor. What do you feel are the key areas they touch upon?",
                "Observe the details in this student's review of the course and professor. What elements seem prominent?",
                "While reading the student's review, try to grasp which aspects of the course and professor they find noteworthy.",
                "Contemplate the student's feedback on the course and professor. Which aspects seem to be highlighted?",
                "As you go through the student's review, what key points about the course and professor do you perceive?",
                "Engage with the student's review on the course and professor, and think about the aspects they seem to focus on."
            ],
            "incorrect_instructions": [
                "Please draft a brief summary of the last chapter you read from your assigned textbook.",
                "Create a mind map that illustrates the key themes of the novel 'To Kill a Mockingbird'.",
                "Conduct a small experiment demonstrating a basic principle of physics and document your findings.",
                "Compose a short essay discussing the impact of social media on modern communication.",
                "Design a poster that effectively communicates the importance of environmental conservation.",
                "Develop a basic mobile app prototype that solves a common everyday problem.",
                "Write a script for a five-minute skit that humorously explores historical events.",
                "Craft a detailed business plan for a startup idea focusing on sustainable technology.",
                "Prepare a presentation on the latest advancements in renewable energy sources.",
                "Analyze a set of data using statistical methods and present your conclusions in a report."
            ]
        }
