import logging, os

import numpy as np
import sklearn
import sklearn.cluster

logger = logging.getLogger(__name__)

from .Evaluator import Evaluator


class ClusteringEvaluator(Evaluator):
    def __init__(self, sentences, labels, clustering_batch_size=500, batch_size=32, limit=None, **kwargs):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.batch_size = batch_size

        self.cache_dir = kwargs['cache_dir']
        self.filter_words = kwargs['filter_words']

    def __call__(self, model):
        logger.info(f"Encoding {len(self.sentences)} sentences...")
        encode_params = {"batch_size": self.batch_size}
        if self.cache_dir is not None:
            encode_params["cache_dir"] = self.cache_dir
        if self.filter_words is not None:
            encode_params["filter_words"] = self.filter_words
        corpus_embeddings = np.asarray(model.encode(self.sentences, **encode_params))

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)), batch_size=self.clustering_batch_size, n_init="auto"
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment)
        # validated that there are still variances
        # for seed in [100, 13, 42]:
        #     logger.info("Fitting Mini-Batch K-Means model...")
        #     clustering_model = sklearn.cluster.MiniBatchKMeans(
        #         n_clusters=len(set(self.labels)), batch_size=self.clustering_batch_size, n_init="auto", random_state=seed
        #     )
        #     clustering_model.fit(corpus_embeddings)
        #     cluster_assignment = clustering_model.labels_

        #     logger.info("Evaluating...")
        #     v_measures.append(sklearn.metrics.cluster.v_measure_score(self.labels, cluster_assignment))

        from collections import defaultdict, Counter
        assignments = defaultdict(list)
        for ca, label in zip(cluster_assignment, self.labels):
            assignments[ca].append(label)
        # breakpoint()
        cluster_components = [dict(Counter(assignments[ca])) for ca in assignments]

        return {"v_measure": v_measure, "cluster_components": cluster_components}
