from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from spectralcluster import SpectralClusterer
from umap import UMAP


def umap_transform(embeddings, n_components=2, n_neighbors=15):
    return UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        random_state=42
    ).fit_transform(embeddings)


def tsne_transform(embeddings, n_components=2, n_iter=3000, learning_rate=250, perplexity=30):
    return TSNE(n_components=n_components,
                n_iter=n_iter,
                learning_rate=learning_rate,
                perplexity=perplexity
                ).fit_transform(embeddings)


def cluster_by_hdbscan(embeddings, min_cluster_size=5, min_samples=10):
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(embeddings)


def cluster_by_dbscan(embeddings, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(embeddings)


def cluster_by_spectral(embeddings):
    return SpectralClusterer(p_percentile=0.95, gaussian_blur_sigma=1).predict(embeddings)


def setup_knn(embeddings_pull, ground_truth_labels):
    classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')
    classifier.fit(embeddings_pull, ground_truth_labels)

    return classifier
