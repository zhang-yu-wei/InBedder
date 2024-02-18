import os
import json
import fire
import logging
import datasets
import numpy as np
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from lm_encoders_hf import CausalLMEncoder, Seq2SeqLMEncoder, MaskededLMEncoder
from transformers import set_seed

from sklearn.cluster import KMeans

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(config_file, use_cache):
    with open(config_file, 'r') as f:
        configs = json.load(f)
    assert isinstance(configs, dict)

    set_seed(configs['seed'])
    data_path = configs.get("data_path", None)
    hf_dataset = configs.get("hf_dataset", None)

    if data_path is not None:
        # data_path must be a json file
        with open(data_path, 'r') as f:
            dataset = json.load(f)
        if isinstance(dataset, list):
            if isinstance(dataset[0], str):
                corpus = dataset
            elif isinstance(dataset[0], dict):
                keys = dataset[0].keys()
                assert all([d.keys() == keys for d in dataset])
                corpus = [' '.join([d[k] for k in keys]) for d in dataset]
        elif isinstance(dataset, dict):
            corpus = sum(dataset.values(), [])
    elif hf_dataset is not None:
        # optimize this when necessary
        dataset = datasets.load_dataset(hf_dataset)
        corpus = list(dataset['sentences'])
    else:
        raise ValueError()

    instruct_id = configs["instruct_id"]
    instruction = configs['instruction']
    corpus = [configs['pattern'].replace('{input}', s).replace('{instruction}', instruction) for s in corpus]

    hf_name = configs.get("hf_name", "")
    model_path = configs.get("model_path", "")
    model_name_or_path = hf_name if hf_name != "" else model_path

    # the name for cache
    model_id = configs.get("model_id", "")
    model_id = model_id if model_id != "" else hf_name.split("/")[-1]

    model_type = configs.get("model_type", "")
    if model_type == "causal":
        model_handle = CausalLMEncoder
    elif model_type == "seq2seq":
        model_handle = Seq2SeqLMEncoder
    elif model_type == "masked":
        model_handle = MaskededLMEncoder
    else:
        raise NotImplementedError()

    reencoder = SentenceTransformer(configs["reencoder"]) if "reencoder" in configs else None

    # process filter words
    filter_words = configs.get("filter_words", None)

    # define model
    model = model_handle(model_name_or_path=model_name_or_path,
                        tokenizer_path=configs.get("tokenizer_path", None),
                        max_input_length=configs['max_input_length'],
                        dtype=configs["dtype"],
                        reencoder=reencoder,
                        use_flash_attention_2=configs.get('use_flash_attention_2', False),
                        use_whitening=configs.get('use_whitening', False),
                        ppl_filtering=configs.get('ppl_filtering', None),
                        last_layer_only=True,
                        **configs["generation_configs"])

    output_value = configs['output_value']
    model.set_output_value(output_value)

    output_path = f"propose_results/{model_id}_{output_value}_{instruct_id}{'_seed='+str(configs['seed'])}.json"
    logger.info(f"Save path: {output_path}")
    cache_dir = output_path.replace(".json", "")
    
    if not use_cache:
        cache_dir = None

    logger.info("Encoding corpus ...")
    corpus_embeddings, generations = model.encode(corpus, batch_size=configs['batch_size'], filter_words=filter_words, cache_dir=cache_dir, return_generations=True)
    corpus_embeddings = np.asarray(corpus_embeddings)

    logger.info("Fitting clustering algorithm ...")
    clustering_model = KMeans(n_clusters=configs["n_clusters"], n_init=10, random_state=42)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    ass2sent = defaultdict(list)
    for i, (assignment, generation) in enumerate(zip(cluster_assignment, generations)):
        ass2sent[int(assignment)].append((generation, i))
    
    with open(output_path, 'w') as f:
        json.dump(ass2sent, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)