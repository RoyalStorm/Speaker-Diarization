# Third Party
import os

import librosa
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN, Birch
from sklearn.neighbors import KNeighborsClassifier
from tensorboard.plugins import projector


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


def setup_dbscan(feats):
    dbscan = DBSCAN(min_samples=10)
    dbscan.fit(feats)

    return dbscan.labels_


def setup_birch(feats):
    birch = Birch(n_clusters=None, branching_factor=500, threshold=0.5)
    birch.fit(feats)
    labels = birch.predict(feats)

    return labels


def classify_noise(feats, clusters):
    unique = np.unique(clusters)
    delegated_embeddings = dict.fromkeys(unique)

    noise_cluster_name = -1

    for label in unique:
        indexes = np.where(np.array(clusters) == label)
        delegated_embeddings[label] = np.median(feats[np.amin(indexes):np.amax(indexes)], axis=0)

    del delegated_embeddings[noise_cluster_name]

    noise_embeddings_indexes = np.where(np.array(clusters) == noise_cluster_name)[0]

    for i in noise_embeddings_indexes:
        min_distance = 1
        speaker_label = None

        for key, value in delegated_embeddings.items():
            dist = cosine(feats[i], value)  # less is better

            if dist < min_distance:
                min_distance = dist
                speaker_label = key

        clusters[i] = speaker_label

    return clusters


def visualize(feats, speaker_labels, mode):
    if mode == 'real_world':
        folder_path = f'./ghostvlad/projections/{mode}'
    elif mode == 'test':
        folder_path = f'./projections/{mode}'
    else:
        raise TypeError('mode should be "real_world" or "test"')

    with open(os.path.join(folder_path, 'metadata.tsv'), 'w+') as metadata:
        for label in speaker_labels:
            if mode == 'real_world':
                metadata.write(f'spk_{label}\n')
            else:
                metadata.write(f'{label}\n')

    sess = tf.InteractiveSession()

    with tf.device("/cpu:0"):
        embedding = tf.Variable(feats, trainable=False, name=mode)
        tf.global_variables_initializer().run()
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter(folder_path, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding'
        embed.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(writer, config)

        saver.save(sess, os.path.join(folder_path, 'model.ckpt'), global_step=feats.shape[0] - 1)
