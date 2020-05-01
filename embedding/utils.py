# Third Party
import os

import hdbscan
import librosa
import numpy as np
import tensorflow as tf
import umap
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from spectralcluster import SpectralClusterer
from tensorboard.plugins import projector
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import global_variables_initializer
from tensorflow.compat.v1.summary import FileWriter
from tensorflow.compat.v1.train import Saver


# ===============================================
#       code from Arsha for loading dataset.
# ===============================================
def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]

        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])

        return extended_wav


def linear_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram

    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = linear_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape

    if mode == 'train':
        randtime = np.random.randint(0, time - spec_len)
        spec_mag = mag_T[:, randtime:randtime + spec_len]
    else:
        spec_mag = mag_T

    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)

    return (spec_mag - mu) / (std + 1e-5)


def umap_transformation(feats):
    return umap.UMAP(
        n_neighbors=15,
        min_dist=0.0,
        n_components=2,
        random_state=42
    ).fit_transform(feats)


def tsne_transformation(feats):
    return TSNE(n_components=2,
                perplexity=30,
                learning_rate=250,
                n_iter=3000,
                n_iter_without_progress=500
                ).fit_transform(feats)


def cluster_by_spectral(feats):
    clusters = SpectralClusterer(
        p_percentile=0.95,
        gaussian_blur_sigma=1)

    return clusters.predict(feats)


def cluster_by_dbscan(feats):
    m = 5

    def eps(m):
        eps = 0.5

        return eps

    dbscan = DBSCAN(eps=eps(m), min_samples=m)
    feats = umap_transformation(feats)
    clusters = dbscan.fit_predict(feats)

    noise_cluster_name = -1

    return list(map(lambda i, _: clusters[i], np.where(np.array(clusters) != noise_cluster_name)[0], clusters))


def setup_knn(embeddings_pull, ground_truth_labels):
    classifier = KNeighborsClassifier(n_neighbors=10, weights='distance')
    classifier.fit(embeddings_pull, ground_truth_labels)

    return classifier


def cluster_by_hdbscan(feats):
    feats = umap_transformation(feats)

    return hdbscan.HDBSCAN(min_samples=10).fit_predict(feats)


def visualize(feats, speaker_labels, mode):
    if mode == 'real_world':
        folder_path = f'./embedding/projections/{mode}'
    elif mode == 'test':
        folder_path = f'./projections/{mode}'
    else:
        raise TypeError('"mode" should be "real_world" or "test"')

    with open(os.path.join(folder_path, 'metadata.tsv'), 'w') as metadata:
        for label in speaker_labels:
            if mode == 'real_world':
                metadata.write(f'spk_{label}\n')
            else:
                metadata.write(f'{label}\n')

    sess = InteractiveSession()

    with tf.device("/cpu:0"):
        embedding = tf.Variable(feats, trainable=False, name=mode)
        global_variables_initializer().run()
        saver = Saver()
        writer = FileWriter(folder_path, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding'
        embed.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(writer, config)

        saver.save(sess, os.path.join(folder_path, 'model.ckpt'), global_step=feats.shape[0] - 1)
