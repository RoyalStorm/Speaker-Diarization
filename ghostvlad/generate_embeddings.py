from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import random

import librosa as lr
import numpy as np
import toolkits
import model

parser = argparse.ArgumentParser()

# Set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'pre_trained/weights.h5', type=str)
parser.add_argument('--data_path', default='D://dataset', type=str)
parser.add_argument('--epochs', default=5, type=int)

# Set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)

# Set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

args = parser.parse_args()


def similar(matrix):
    """Calculate speaker-embeddings similarity in pretty format output.

    Args: matrix:
    """
    ids = matrix.shape[0]

    for i in range(ids):
        for j in range(ids):
            dist = np.linalg.norm(matrix[i, :] - matrix[j, :])

            print('%.2f  ' % dist, end='')

            if (j + 1) % 3 == 0 and j != 0:
                print("| ", end='')

        if (i + 1) % 3 == 0 and i != 0:
            print('\n')
            print('*' * 80, end='')

        print('\n')


def load_wav(vid_path, sr):
    wav, sr_ret = lr.load(vid_path, sr=sr)
    assert sr_ret == sr

    intervals = lr.effects.split(wav, top_db=20)
    wav_output = []

    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])

    wav_output = np.array(wav_output)

    return wav_output


def linear_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = lr.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram

    return linear.T


def load_data(path_spk_tuples, win_length=400, sr=16000, hop_length=160, n_fft=512, min_win_time=240,
              max_win_time=1600):
    win_time = np.random.randint(min_win_time, max_win_time, 1)[0]  # win_length in [240,1600] ms
    win_spec = win_time // (1000 // (sr // hop_length))  # win_length in spectrum
    hop_spec = win_spec // 2

    wavs = np.array([])
    change_points = []
    paths = list(zip(*path_spk_tuples))[0]
    speakers = list(zip(*path_spk_tuples))[1]

    for path in paths:
        wav = load_wav(path, sr=sr)  # VAD
        wavs = np.concatenate((wavs, wav))
        change_points.append(wavs.shape[0] // hop_length)  # change_point in spectrum

    linear_spect = linear_spectogram_from_wav(wavs, hop_length, win_length, n_fft)
    mag, _ = lr.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    utterance_specs = []
    utterance_speakers = []

    cur_spec = 0
    cur_speaker = speakers[0]
    i = 0
    while True:
        if cur_spec + win_spec > time:
            break
        spec_mag = mag_T[:, cur_spec:cur_spec + win_spec]

        if cur_spec + win_spec // 2 > change_points[i]:  # cur win_spec span to the next speaker
            i += 1
            cur_speaker = speakers[i]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterance_specs.append(spec_mag)
        utterance_speakers.append(cur_speaker)
        cur_spec += hop_spec

    return utterance_specs, utterance_speakers


def prepare_data(dataset_path):
    paths_list = []
    speakers_labels_list = []

    for i, speaker_dir in enumerate(os.listdir(dataset_path)):
        wav_path = os.path.join(dataset_path, speaker_dir)

        for wav in os.listdir(wav_path):
            utterance_path = os.path.join(wav_path, wav)
            paths_list.append(utterance_path)
            speakers_labels_list.append(i)

    path_spk_list = list(zip(paths_list, speakers_labels_list))

    return path_spk_list


def main():
    # GPU config
    toolkits.initialize_GPU(args)

    # Construct the dataset generator
    params = {'dim': (257, None, 1),
              'nfft': 512,
              'min_slice': 720,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True
              }

    network_eval = model.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                num_class=params['n_classes'],
                                                mode='eval',
                                                args=args)

    if args.resume:
        if os.path.isfile(args.resume):
            network_eval.load_weights(os.path.join(args.resume), by_name=True)
            print('Successfully loading model {}.'.format(args.resume))
        else:
            raise IOError("No checkpoint found at '{}'".format(args.resume))
    else:
        raise IOError('Please type in the model to load')

    """The feature extraction process has to be done sample-by-sample,
       because each sample is of different lengths.
    """

    path_spk_tuples = prepare_data(args.data_path)
    train_sequence = []
    train_cluster_id = []

    for epoch in range(args.epochs):  # Random choice utterances from whole wav files
        # A merged utterance contains [10,20] utterances
        splits_count = np.random.randint(10, 20, 1)[0]
        path_spks = random.sample(path_spk_tuples, splits_count)
        utterance_specs, utterance_speakers = load_data(path_spks, min_win_time=500, max_win_time=1600)

        feats = []
        for spec in utterance_specs:
            spec = np.expand_dims(np.expand_dims(spec, 0), -1)
            v = network_eval.predict(spec)
            feats += [v]

        feats = np.array(feats)[:, 0, :]  # [splits, embedding dim]
        train_sequence.append(feats)
        train_cluster_id.append(utterance_speakers)

        print('Epoch:{}, utterance length: {}, speakers: {}'.format(epoch, len(utterance_speakers), len(path_spks)))

    np.savez('training_data', train_sequence=train_sequence, train_cluster_id=train_cluster_id)


if __name__ == "__main__":
    main()
