"""A demo script showing how to DIARIZATION ON WAV USING UIS-RNN."""

import sys

import librosa
import numpy as np

import uisrnn

sys.path.append('ghostvlad')
sys.path.append('visualization')
import toolkits
import model as spkModel
from viewer import PlotDiar

# ===========================================
#        Parse the argument
# ===========================================
import argparse

parser = argparse.ArgumentParser()
# set up training configuration.
parser.add_argument('--gpu', default='', type=str)
parser.add_argument('--resume', default=r'ghostvlad/pretrained/weights.h5', type=str)
parser.add_argument('--data_path', default='4persons', type=str)
# set up network configuration.
parser.add_argument('--net', default='resnet34s', choices=['resnet34s', 'resnet34l'], type=str)
parser.add_argument('--ghost_cluster', default=2, type=int)
parser.add_argument('--vlad_cluster', default=8, type=int)
parser.add_argument('--bottleneck_dim', default=512, type=int)
parser.add_argument('--aggregation_mode', default='gvlad', choices=['avg', 'vlad', 'gvlad'], type=str)
# set up learning rate, training loss and optimizer.
parser.add_argument('--loss', default='softmax', choices=['softmax', 'amsoftmax'], type=str)
parser.add_argument('--test_type', default='normal', choices=['normal', 'hard', 'extend'], type=str)

global args
args = parser.parse_args()

SAVED_MODEL_NAME = 'pretrained/saved_model.uisrnn_benchmark'


def append2dict(speakerSlice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    time_dict = {'start': int(value[0] + 0.5), 'stop': int(value[1] + 0.5)}
    if key in speakerSlice:
        speakerSlice[key].append(time_dict)
    else:
        speakerSlice[key] = [time_dict]

    return speakerSlice


def arrange_result(labels, time_spec_rate):  # {'1': [{'start':10, 'stop':20}, {'start':30, 'stop':40}], '2': [{'start':90, # 'stop':100}]}
    last_label = labels[0]
    speaker_slice = {}
    j = 0
    for i, label in enumerate(labels):
        if label == last_label:
            continue
        speaker_slice = append2dict(speaker_slice, {last_label: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        last_label = label
    speaker_slice = append2dict(speaker_slice, {last_label: (time_spec_rate * j, time_spec_rate * (len(labels)))})
    return speaker_slice


def gen_map(intervals):  # interval slices to maptable
    slice_len = [sliced[1] - sliced[0] for sliced in intervals.tolist()]
    map_table = {}  # vad erased time to origin time, only split points
    idx = 0
    for i, sliced in enumerate(intervals.tolist()):
        map_table[idx] = sliced[0]
        idx += slice_len[i]
    map_table[sum(slice_len)] = intervals[-1, -1]

    keys = [k for k, _ in map_table.items()]
    keys.sort()
    return map_table, keys


def fmt_time(time_in_milliseconds):
    millisecond = time_in_milliseconds % 1000
    minute = time_in_milliseconds // 1000 // 60
    second = (time_in_milliseconds - minute * 60 * 1000) // 1000
    time = '{}:{:02d}.{}'.format(minute, second, millisecond)
    return time


def load_wav(vid_path, sr):
    wav, _ = librosa.load(vid_path, sr=sr)
    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
        wav_output.extend(wav[sliced[0]:sliced[1]])
    return np.array(wav_output), (intervals / sr * 1000).astype(int)


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
    return linear.T


# 0s        1s        2s                  4s                  6s
# |-------------------|-------------------|-------------------|
# |-------------------|
#           |-------------------|
#                     |-------------------|
#                               |-------------------|
def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5, overlap_rate=0.5):
    wav, intervals = load_wav(path, sr=sr)
    linear_spectpgram = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spectpgram)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    spec_len = sr / hop_length / embedding_per_second
    spec_hop_len = spec_len * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    while True:  # slide window.
        if cur_slide + spec_len > time:
            break
        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_len + 0.5)]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_len

    return utterances_spec, intervals


def main(wav_path, embedding_per_second=1.0, overlap_rate=0.5):
    # gpu configuration
    toolkits.initialize_GPU(args)

    params = {'dim': (257, None, 1),
              'nfft': 512,
              'spec_len': 250,
              'win_length': 400,
              'hop_length': 160,
              'n_classes': 5994,
              'sampling_rate': 16000,
              'normalize': True
              }

    network_eval = spkModel.vggvox_resnet2d_icassp(input_dim=params['dim'],
                                                   num_class=params['n_classes'],
                                                   mode='eval', args=args)
    network_eval.load_weights(args.resume, by_name=True)

    model_args, _, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    uisrnn_model = uisrnn.UISRNN(model_args)
    uisrnn_model.load(SAVED_MODEL_NAME)

    specs, intervals = load_data(wav_path, embedding_per_second=embedding_per_second, overlap_rate=overlap_rate)
    map_table, keys = gen_map(intervals)

    feats = []
    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = network_eval.predict(spec)
        feats += [v]

    feats = np.array(feats)[:, 0, :].astype(float)  # [splits, embedding dim]
    predicted_label = uisrnn_model.predict(feats, inference_args)

    time_spec_rate = 1000 * (1.0 / embedding_per_second) * (1.0 - overlap_rate)  # speaker embedding every ?ms
    center_duration = int(1000 * (1.0 / embedding_per_second) // 2)
    speaker_slice = arrange_result(predicted_label, time_spec_rate)

    for spk, timeDicts in speaker_slice.items():  # time map to origin wav (contains mute)
        for tid, timeDict in enumerate(timeDicts):
            s = 0
            e = 0
            for i, key in enumerate(keys):
                if s != 0 and e != 0:
                    break
                if s == 0 and key > timeDict['start']:
                    offset = timeDict['start'] - keys[i - 1]
                    s = map_table[keys[i - 1]] + offset
                if e == 0 and key > timeDict['stop']:
                    offset = timeDict['stop'] - keys[i - 1]
                    e = map_table[keys[i - 1]] + offset

            speaker_slice[spk][tid]['start'] = s
            speaker_slice[spk][tid]['stop'] = e

    for spk, timeDicts in speaker_slice.items():
        print('========= ' + str(spk) + ' =========')
        for timeDict in timeDicts:
            s = timeDict['start']
            e = timeDict['stop']
            s = fmt_time(s)  # change point moves to the center of the slice
            e = fmt_time(e)
            print(s + ' ==> ' + e)

    p = PlotDiar(map=speaker_slice, wav=wav_path, gui=True, size=(25, 6))
    p.draw()
    p.plot.show()


if __name__ == '__main__':
    main(r'wavs/rmdmy.wav', embedding_per_second=1.2, overlap_rate=0.4)
