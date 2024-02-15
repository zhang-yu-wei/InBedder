"""
directly generate csv result table content
"""
import os
import json
import csv
import re
from collections import defaultdict

# tasks = {'TwentyNewsgroupsClustering': ["v_measure"], 'IntentEmotion': ["Emotion_success_rate", "Intent_success_rate"], 'FeedbacksClustering': ["v_measure"], "STSBenchmark": ["cosine_spearman"], "AskUbuntuDupQuestions": ["map"], "SciDocsRR": ["map"], "StackOverflowDupQuestions": ["map"], "ArxivClusteringS2S": ["v_measure"], "STS12": ["cosine_spearman"], "RedditClustering": ["v_measure"], "FewRelClustering": ["v_measure"], "RateMyProfClustering": ["v_measure"], "FewNerdClustering": ["v_measure"], "FewEventClustering": ["v_measure"], "InstructSTSBenchmark": ["cosine_spearman"], "NYTTopicClustering": ["v_measure"], "NYTLocationClustering": ["v_measure"]}
# tasks = {'TwentyNewsgroupsClustering': ["v_measure"], 'IntentEmotion': ["Emotion_success_rate", "Intent_success_rate"], 'FeedbacksClustering': ["v_measure"], "FewRelClustering": ["v_measure"], "RateMyProfClustering": ["v_measure"], "FewNerdClustering": ["v_measure"], "FewEventClustering": ["v_measure"], "InstructSTSBenchmark": ["cosine_spearman"], "AskUbuntuDupQuestions": ["map"], "SciDocsRR": ["map"], "StackOverflowDupQuestions": ["map"], "NYTTopicClustering": ["v_measure"], "NYTLocationClustering": ["v_measure"]}
tasks = {"IntentEmotionMean": ["harmonic_mean"], "NYTClustering": ["harmonic_mean"], "InstructSTSBenchmark": ["cosine_spearman"]}
# tasks = {"TwentyNewsgroupsClustering": ["v_measure"], "AskUbuntuDupQuestions": ["map"], "SciDocsRR": ["map"], "StackOverflowDupQuestions": ["map"]}

# models = ['opt-350m-pretrain-ml=20_', 'flan-t5-large-pretrain-ml=20_', 'llama-2-7b-chat-pretrain-ml=40_', 'llama-2-7b-chat-pretrain-ml=40-whiten_', 'llama-2-7b-chat-pretrain-filter-ml=40_', 'llama-2-13b-chat-pretrain-ml=40_', 'llama-2-13b-chat-pretrain-filter-ml=40_', 'llama-2-13b-chat-pretrain-ml=40-whiten_', 'opt-350m-alpaca-ml=20_', 'flan-t5-large-alpaca-ml=20_', 'opt-350m-qa-ml=20_', 'opt-1.3b-qa-ml=20_', 'opt-2.7b-qa-ml=20_', 'opt-6.7b-qa-ml=20_', 'opt-1.3b-alpaca-ml=20_', 'e5-large-v2', 'roberta-large-qa-ml=3_', 'roberta-large-qa-ml=1_', 'roberta-large-qa-ml=10_', 'roberta-large-alpaca-ml=3_', 'flan-t5-large-qa-ml=20_', 'Llama-2-7b-hf-qa-ml=20']
models = ['opt-350m-qa-ml=20_', 'opt-1.3b-qa-ml=20_', 'opt-2.7b-qa-ml=20_', 'opt-6.7b-qa-ml=20_', 'e5-large-v2', 'roberta-large-qa-ml=3_', 'Llama-2-7b-hf-qa-ml=20_', 'Llama-2-7b-hf-e5-large-v2-qa-ml=20_', 'Llama-2-7b-hf-qa-ml=40_', 'Llama-2-7b-hf-qa-ml=3_', 'Llama-2-7b-hf-qa-ml=20-lora-2e-4_', 'Llama-2-7b-hf-qa-noprocess-ml=20_', 'llama-2-7b-chat-e5-large-v2-pretrain-ml=40_', 'llama-2-13b-chat-e5-large-v2-pretrain-ml=40_', 'llama-2-7b-chat-e5-large-v2-pretrain-filter-ml=40_', 'instructor-large', 'flan-t5-large-e5-large-v2-pretrain-ml=40_', 'flan-t5-xl-e5-large-v2-pretrain-ml=40_', 'flan-t5-large-e5-large-v2-pretrain-filter-ml=40_', 'flan-t5-xl-e5-large-v2-pretrain-filter-ml=40_', 'llama-2-13b-chat-e5-large-v2-pretrain-filter-ml=40', 'flan-t5-large-qa-ml=20_', 't5-v1_1-large-qa-ml=20_', 'flan-t5-xl-qa-ml=20_', 'opt-1.3b-alpaca-ml=20_', 'opt-2.7b-alpaca-ml=20_', 'opt-350m-alpaca-ml=20_', 'roberta-large-alpaca-ml=3_', 'Llama-2-7b-hf-e5-large-v2-qa-ml=3_', 'llama-2-7b-chat-e5-large-v2-pretrain-multiple-ml=40_', 'Llama-2-7b-hf-e5-large-v2-qa-ml=3-multiple_']
# models = ['bowce-opt-350m-qa-ml=20_', 'opt-350m-qa-ml=20_']

all_results = os.listdir("results_hf")

results_dict = defaultdict(list)
for model in models:
    # find all instances with model name on it
    cur_results = [res for res in all_results if res.startswith(model)]

    for res in cur_results:
        if 'layer' in res:
            pattern = r'layer_(\d+)'
            match = re.search(pattern, res)

            if match:
                layer_number = match.group(1)
                method = res[:match.start()-1] # is usually model + aggregation method
            else:
                raise ValueError()

            results_dict[method].append({"folder_name": res, "layer_number": int(layer_number)})
        else:
            results_dict[res].append({"folder_name": res, "layer_number": -1})

for method in results_dict:
    results_dict[method].sort(key=lambda k: k["layer_number"], reverse=True)

performance_dict = {}
for method in results_dict:
    if len(results_dict[method]) > 1:
        folder_name = results_dict[method][0]["folder_name"]
    else:
        folder_name = results_dict[method][0]["folder_name"]
    results_path = os.path.join("results_hf", folder_name) # last layer
    performance_dict[folder_name] = {}
    for task in tasks:
        task_path = os.path.join(results_path, task+".json")
        if not os.path.exists(task_path):
            for metric in tasks[task]:
                performance_dict[folder_name][task + ":" + metric] = "-"
        else:
            with open(task_path, 'r') as f:
                performances = json.load(f)
            for metric in tasks[task]:
                if '_' in metric and task in ["STS12", "STSBenchmark", "InstructSTSBenchmark"]:
                    m1, m2 = metric.split('_')
                    if m1 == "cosine":
                        m1 = "cos_sim"
                    performance = performances['test'][m1][m2]
                else:
                    performance = performances['test'][metric]
                performance_dict[folder_name][task + ":" + metric] = round(float(performance), 4) * 100

methods_sorted = sorted(list(performance_dict.keys()))
tasks_sorted = sorted(list(performance_dict[methods_sorted[0]]))

final_table = [[""] + tasks_sorted + ["Avg"]]
for method in methods_sorted:
    final_table.append([method] + [performance_dict[method][t] for t in tasks_sorted])
    if all([t != '-' for t in final_table[-1][1:]]):
        final_table[-1].append(sum(final_table[-1][1:]) / len(final_table[-1][1:]))
    else:
        final_table[-1].append('-')

with open("tables/instruction_awareness_tests.csv", 'w') as f:
# with open("tables/mteb_subset.csv", 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(final_table)
