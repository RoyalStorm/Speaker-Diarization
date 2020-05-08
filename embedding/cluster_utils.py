from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from spectralcluster import SpectralClusterer
from umap import UMAP

from embedding import consts


def umap_transform(embeddings,
                   n_components=consts.umap_params.n_components,
                   n_neighbors=consts.umap_params.n_neighbors
                   ):
    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42
    ).fit_transform(embeddings)


def tsne_transform(embeddings,
                   n_components=consts.tsne_params.n_components,
                   n_iter=consts.tsne_params.n_iter,
                   learning_rate=consts.tsne_params.learning_rate,
                   perplexity=consts.tsne_params.perplexity
                   ):
    return TSNE(n_components=n_components,
                n_iter=n_iter,
                learning_rate=learning_rate,
                perplexity=perplexity
                ).fit_transform(embeddings)


def cluster_by_hdbscan(embeddings,
                       min_cluster_size=consts.hdbscan_params.min_cluster_size,
                       min_samples=consts.hdbscan_params.min_samples
                       ):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(embeddings)


def cluster_by_dbscan(embeddings,
                      eps=consts.dbscan_params.eps,
                      min_samples=consts.dbscan_params.min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings)


def cluster_by_spectral(embeddings):
    return SpectralClusterer(p_percentile=0.95, gaussian_blur_sigma=1).predict(embeddings)


def setup_knn(embeddings_pull, ground_truth_labels, n_neighbors=10):
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    classifier.fit(embeddings_pull, ground_truth_labels)

    return classifier
