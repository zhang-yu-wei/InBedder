"""
Evaluation code
"""
import os
import json
import fire
import logging
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from mteb import MTEB
# select a few tasks to save time
from mteb.tasks import IntentEmotion, FeedbacksClustering, TwentyNewsgroupsClustering, AskUbuntuDupQuestions, SciDocsReranking, StackOverflowDupQuestions, FewRelClustering, RateMyProfClustering, FewNerdClustering, FewEventClustering, InstructSTSBenchmark, NYTLocationClustering, NYTTopicClustering
from sentence_transformers import SentenceTransformer
from InstructorEmbedding import INSTRUCTOR
from lm_encoders_hf import CausalLMEncoder, Seq2SeqLMEncoder, MaskededLMEncoder, CausalLMEncoderMultiple
from transformers import set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)


def main(config_file, output_value: str = None, overwrite_results: bool = False, cache_outputs: bool = True, last_layer_only: bool = True):
    with open(config_file, 'r') as f:
        configs = json.load(f)
    assert isinstance(configs, dict)

    set_seed(configs['seed'])

    hf_name = configs.get("hf_name", "")
    model_path = configs.get("model_path", "")
    model_name_or_path = hf_name if hf_name != "" else model_path

    # the name for cache
    model_id = configs.get("model_id", "")
    model_id = model_id if model_id != "" else hf_name.split("/")[-1]
    print(model_id)

    model_type = configs.get("model_type", "")
    if model_type not in ["sentence-transformers", "instructor"]:
        if model_type == "causal":
            model_handle = CausalLMEncoder
        elif model_type == "causal_multiple":
            model_handle = CausalLMEncoderMultiple
        elif model_type == "seq2seq":
            model_handle = Seq2SeqLMEncoder
        elif model_type == "masked":
            model_handle = MaskededLMEncoder
        else:
            raise NotImplementedError()
        
        # assume that reencoder is usually a generic embedder
        reencoder = SentenceTransformer(configs["reencoder"])

        # no filter words will be used if you don't specify it directly.
        # process filter words
        filter_words = configs.get("filter_words", None)
        if filter_words is not None:
            for task in filter_words:
                filter_words[task] += stopwords.words("english")

        # define model
        model = model_handle(model_name_or_path=model_name_or_path,
                            tokenizer_path=configs.get("tokenizer_path", None),
                            max_input_length=configs['max_input_length'],
                            dtype=configs["dtype"],
                            reencoder=reencoder,
                            use_flash_attention_2=configs.get('use_flash_attention_2', False),
                            last_layer_only=last_layer_only,
                            reencode_times=configs.get("reencode_times", 1),
                            **configs["generation_configs"])
        
        if output_value is not None:
            all_output_values = [output_value]
        else:
            if last_layer_only:
                all_output_values = model.get_last_output_values()
            else:
                all_output_values = [output_value] if output_value is not None else model.get_all_output_values()
        # iterate for all possible output values
        os.makedirs("results_hf", exist_ok=True)
        os.makedirs("cache_hf", exist_ok=True)
        for ov in all_output_values:
            model.set_output_value(ov)

            output_folder = f"results_hf/{model_id}_{ov}{'_instruct' if not configs['disable_instruction'] else ''}{'_seed='+str(configs['seed'])}"
            cache_dir = f"cache_hf/{model_id}{'_instruct' if not configs['disable_instruction'] else ''}{'_seed='+str(configs['seed'])}"

            evaluation = MTEB(tasks=[
                InstructSTSBenchmark(),
                IntentEmotion(),
                NYTLocationClustering(),
                NYTTopicClustering(),
                FeedbacksClustering(),
                FewRelClustering(),
                RateMyProfClustering(),
                FewNerdClustering(),
                FewEventClustering(),
                TwentyNewsgroupsClustering(),
                AskUbuntuDupQuestions(),
                SciDocsReranking(),
                StackOverflowDupQuestions(),
            ])
            if not cache_outputs:
                cache_dir = None
            evaluation.run(model,
                        output_folder=output_folder,
                        eval_splits=["test"],
                        disable_instruction=configs['disable_instruction'], batch_size=configs['batch_size'],
                        seed=configs['seed'],
                        cache_dir=cache_dir, overwrite_results=overwrite_results, pattern=configs['pattern'], filter_words=filter_words, truncate_to_length=configs.get("truncate_to_length", None))
    elif model_type == "sentence-transformers":
        model = SentenceTransformer(model_name_or_path)
        output_folder = f"results_hf/{model_id}{'_instruct' if not configs['disable_instruction'] else ''}"

        evaluation = MTEB(tasks=[
            InstructSTSBenchmark(),
            IntentEmotion(),
            NYTLocationClustering(),
            NYTTopicClustering(),
            FeedbacksClustering(),
            FewRelClustering(),
            RateMyProfClustering(),
            FewNerdClustering(),
            FewEventClustering(),
            TwentyNewsgroupsClustering(),
            AskUbuntuDupQuestions(),
            SciDocsReranking(),
            StackOverflowDupQuestions(),
        ])
        evaluation.run(model,
                    output_folder=output_folder,
                    eval_splits=["test"],
                    disable_instruction=configs['disable_instruction'], batch_size=configs['batch_size'],
                    overwrite_results=overwrite_results, pattern=configs.get('pattern', None))
    else:
        model = INSTRUCTOR(model_name_or_path)
        output_folder = f"results_hf/{model_id}{'_instruct' if not configs['disable_instruction'] else ''}"

        evaluation = MTEB(tasks=[
            InstructSTSBenchmark(),
            IntentEmotion(),
            NYTLocationClustering(),
            NYTTopicClustering(),
            FeedbacksClustering(),
            FewRelClustering(),
            RateMyProfClustering(),
            FewNerdClustering(),
            FewEventClustering(),
            TwentyNewsgroupsClustering(),
            AskUbuntuDupQuestions(),
            SciDocsReranking(),
            StackOverflowDupQuestions(),
        ])
        evaluation.run(model,
                    output_folder=output_folder,
                    eval_splits=["test"],
                    disable_instruction=configs['disable_instruction'], batch_size=configs['batch_size'],
                    overwrite_results=overwrite_results, pattern=configs['pattern'])


if __name__ == "__main__":
    fire.Fire(main)