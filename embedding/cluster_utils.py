import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from spectralcluster import SpectralClusterer
from umap import UMAP


def _remove_noise_cluster(predicted_labels):
    noise_cluster_name = -1

    return list(map(lambda i, _: predicted_labels[i], np.where(np.array(predicted_labels) != noise_cluster_name)[0],
                    predicted_labels))


def umap_transform(embeddings,
                   n_components=2,
                   n_neighbors=15):
    return UMAP(
        n_components=n_components,
        metric='cosine',
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42
    ).fit_transform(embeddings)


def tsne_transform(embeddings,
                   n_components=2,
                   n_iter=3_000,
                   n_iter_without_progress=300,
                   metric='cosine',
                   learning_rate=250,
                   perplexity=30,
                   init='pca'):
    return TSNE(n_components=n_components,
                n_iter=n_iter,
                n_iter_without_progress=n_iter_without_progress,
                metric=metric,
                learning_rate=learning_rate,
                perplexity=perplexity,
                init=init
                ).fit_transform(embeddings)


def cluster_by_hdbscan(embeddings,
                       min_cluster_size=15,
                       min_samples=5):

    return _remove_noise_cluster(HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
                                 .fit_predict(embeddings))


def cluster_by_dbscan(embeddings,
                      eps=0.5,
                      min_samples=5):
    return _remove_noise_cluster(DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings))


def cluster_by_spectral(embeddings):
    return SpectralClusterer(p_percentile=0.95, gaussian_blur_sigma=1).predict(embeddings)
