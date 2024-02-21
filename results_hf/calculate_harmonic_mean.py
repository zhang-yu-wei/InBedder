"""some of the tasks need to be converted to harmonic mean"""
import json
import os


def harmonic_mean(n1, n2):
    return 2/(1/(n1 + 1e-8) + 1/(n2 + 1e-8))

results_folders = os.listdir(".")
for folder in results_folders:
    if not os.path.isdir(os.path.join(".", folder)):
        continue
    results_files = os.listdir(os.path.join(".", folder))
    if "IntentEmotion.json" in results_files:
        with open(os.path.join(".", folder, "IntentEmotion.json"), 'r') as f:
            results = json.load(f)
        results['test']['harmonic_mean'] = harmonic_mean(results['test']['Emotion_success_rate'], results['test']['Intent_success_rate'])
        with open(os.path.join(".", folder, "IntentEmotionMean.json"), 'w') as f:
            json.dump(results, f, indent=4)
    if "NYTTopicClustering.json" in results_files and "NYTLocationClustering.json" in results_files:
        with open(os.path.join(".", folder, "NYTTopicClustering.json"), 'r') as f:
            topic_results = json.load(f)
        with open(os.path.join(".", folder, "NYTLocationClustering.json"), 'r') as f:
            location_results = json.load(f)
        all_results = {"test": {"NYTTopicClustering": topic_results['test']['v_measure'], "NYTLocationClustering": location_results['test']['v_measure'], "harmonic_mean": harmonic_mean(topic_results['test']['v_measure'], location_results['test']['v_measure'])}}
        with open(os.path.join(".", folder, "NYTClustering.json"), 'w') as f:
            json.dump(all_results, f, indent=4)
