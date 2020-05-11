import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from spectralcluster import SpectralClusterer
from umap import UMAP

from embedding import consts


def _denoising(predicted_labels):
    noise_cluster_name = -1

    return list(map(lambda i, _: predicted_labels[i], np.where(np.array(predicted_labels) != noise_cluster_name)[0],
                    predicted_labels))


def umap_transform(embeddings,
                   n_components=consts.umap_params.n_components,
                   n_neighbors=consts.umap_params.n_neighbors):
    return UMAP(
        n_components=n_components,
        metric='cosine',
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42
    ).fit_transform(embeddings)


def tsne_transform(embeddings,
                   n_components=consts.tsne_params.n_components,
                   n_iter=consts.tsne_params.n_iter,
                   n_iter_without_progress=consts.tsne_params.n_iter_without_progress,
                   metric=consts.tsne_params.metric,
                   learning_rate=consts.tsne_params.learning_rate,
                   perplexity=consts.tsne_params.perplexity,
                   init=consts.tsne_params.init):
    return TSNE(n_components=n_components,
                n_iter=n_iter,
                n_iter_without_progress=n_iter_without_progress,
                metric=metric,
                learning_rate=learning_rate,
                perplexity=perplexity,
                init=init
                ).fit_transform(embeddings)


def cluster_by_hdbscan(embeddings,
                       min_cluster_size=consts.hdbscan_params.min_cluster_size,
                       min_samples=consts.hdbscan_params.min_samples):

    return _denoising(HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(embeddings))


def cluster_by_dbscan(embeddings,
                      eps=consts.dbscan_params.eps,
                      min_samples=consts.dbscan_params.min_samples):
    return _denoising(DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings))


def cluster_by_spectral(embeddings):
    return SpectralClusterer(p_percentile=0.95, gaussian_blur_sigma=1).predict(embeddings)


def setup_knn(embeddings_pull, ground_truth_labels, n_neighbors=15):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    classifier.fit(embeddings_pull, ground_truth_labels)

    return classifier


def identify_speakers(classifier, embeddings, predicted_labels):
    clusters = list(np.unique(predicted_labels))

    identified_speakers = dict.fromkeys(clusters)

    for cluster in identified_speakers.keys():
        indexes = np.where(np.array(predicted_labels) == cluster)[0]

        selected_embeddings = []
        for i in indexes:
            selected_embeddings.append(embeddings[i])

        selected_embeddings = np.array(selected_embeddings)

        neigh_dist, neigh_ind = classifier.kneighbors(selected_embeddings, return_distance=True)

    return identified_speakers
