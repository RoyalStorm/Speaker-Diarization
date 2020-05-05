import os
from datetime import datetime

import librosa
import numpy as np
import simpleder
import tensorflow as tf
from tensorboard.plugins import projector
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import global_variables_initializer
from tensorflow.compat.v1.summary import FileWriter
from tensorflow.compat.v1.train import Saver

from embedding import consts


def _append_2_dict(speaker_slice, spk_period):
    key = list(spk_period.keys())[0]
    value = list(spk_period.values())[0]
    time_dict = {'start': int(value[0] + 0.5), 'stop': int(value[1] + 0.5)}

    if key in speaker_slice:
        speaker_slice[key].append(time_dict)
    else:
        speaker_slice[key] = [time_dict]

    return speaker_slice


def _arrange_result(labels, time_spec_rate):
    last_label = labels[-1]
    speaker_slice = {}
    j = 0

    for i, label in enumerate(labels):
        if label == last_label:
            continue

        speaker_slice = _append_2_dict(speaker_slice, {last_label: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        last_label = label

    speaker_slice = _append_2_dict(speaker_slice, {last_label: (time_spec_rate * j, time_spec_rate * (len(labels)))})

    return speaker_slice


def _beautify_time(time_in_milliseconds):
    minute = time_in_milliseconds // 1_000 // 60
    second = (time_in_milliseconds - minute * 60 * 1_000) // 1_000
    millisecond = time_in_milliseconds % 1_000

    time = f'{minute}:{second:02d}.{millisecond}'

    return time


def _path_to_audio(audio_folder):
    for file in os.listdir(audio_folder):
        if file.endswith('.wav'):
            return os.path.join(audio_folder, file)

    raise FileExistsError(f'Folder "{audio_folder}" not contains *.wav file')


def der(ground_truth_map, result_map):
    def convert(map):
        segments = []

        for cluster in sorted(map.keys()):
            for row in map[cluster]:
                segments.append((str(cluster), row['start'] / 1000, row['stop'] / 1000))

        segments.sort(key=lambda segment: segment[1])

        return segments

    ground_truth_map = convert(ground_truth_map)
    result_map = convert(result_map)
    der = simpleder.DER(ground_truth_map, result_map)

    return round(der, 5)


def gen_map(intervals):  # interval slices to map table
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


def generate_embeddings(model, specs):
    embeddings = []

    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = model.predict(spec)
        embeddings.append(list(v))

    embeddings = np.array(embeddings)[:, 0, :].astype(float)

    return embeddings


def linear_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    return linear.T


def ground_truth_map(audio_folder):
    ground_truth_map_file = None

    for file in os.listdir(audio_folder):
        if file == consts.ground_truth_map_file:
            ground_truth_map_file = os.path.join(audio_folder, file)
            break

    with open(ground_truth_map_file, 'r') as file:
        spk_number = 0
        ground_truth_map = {spk_number: []}

        def empty(line):
            return line in ['\n', '\r\n']

        for line in file:
            if empty(line):
                spk_number += 1
                ground_truth_map[spk_number] = []
            else:
                start, stop = line.split(' ')[0], line.split(' ')[1].replace('\n', '')
                dt_start = datetime.strptime(start, '%M:%S.%f')
                dt_stop = datetime.strptime(stop, '%M:%S.%f')

                start = dt_start.minute * 60_000 + dt_start.second * 1_000 + dt_start.microsecond / 1_000
                stop = dt_stop.minute * 60_000 + dt_stop.second * 1_000 + dt_stop.microsecond / 1_000

                ground_truth_map[spk_number].append({'start': start, 'stop': stop})

    return ground_truth_map


def result_map(map_table, keys, predicted_labels):
    time_spec_rate = 1000 * (1.0 / consts.slide_window_params.embedding_per_second) * (
            1.0 - consts.slide_window_params.overlap_rate)  # speaker embedding every ?ms
    speaker_slice = _arrange_result(predicted_labels, time_spec_rate)

    # Time map to origin wav (contains mute)
    for speaker, timestamps_list in speaker_slice.items():
        print('========= ' + str(speaker) + ' =========')

        for timestamp_id, timestamp in enumerate(timestamps_list):
            s = 0
            e = 0

            for i, key in enumerate(keys):
                if s != 0 and e != 0:
                    break

                if s == 0 and key > timestamp['start']:
                    offset = timestamp['start'] - keys[i - 1]
                    s = map_table[keys[i - 1]] + offset

                if e == 0 and key > timestamp['stop']:
                    offset = timestamp['stop'] - keys[i - 1]
                    e = map_table[keys[i - 1]] + offset

            speaker_slice[speaker][timestamp_id]['start'] = s
            speaker_slice[speaker][timestamp_id]['stop'] = e

            s = _beautify_time(timestamp['start'])  # Change point moves to the center of the slice
            e = _beautify_time(timestamp['stop'])

            print(s + ' --> ' + e)

    return speaker_slice


def save_and_report(plot, result_map, der, audio_folder=consts.audio_folder):
    with open(os.path.join(audio_folder, consts.result_map_file), 'w') as file:
        plot.save()

        for i, cluster in enumerate(sorted(result_map.keys())):
            if i != 0:
                file.write('\n')

            file.write(f'{cluster}\n')

            for segment in result_map[cluster]:
                file.write(f'{_beautify_time(segment["start"])} --> {_beautify_time(segment["stop"])}\n')

        file.write(f'\n{der}')

        print(f'Diarization done. All results saved in {audio_folder}.')


def slide_window(audio_folder, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5,
                 overlap_rate=0.5):
    def vad(audio_path, sr):
        audio, _ = librosa.load(audio_path, sr=sr)
        intervals = librosa.effects.split(audio, top_db=20)
        audio_output = []

        for sliced in intervals:
            audio_output.extend(audio[sliced[0]:sliced[1]])

        return np.array(audio_output), (intervals / sr * 1000).astype(int)

    wav, intervals = vad(_path_to_audio(audio_folder), sr=sr)
    linear_spectogram = linear_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spectogram)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape

    spec_length = sr / hop_length / embedding_per_second
    spec_hop_length = spec_length * (1 - overlap_rate)

    cur_slide = 0.0
    utterances_spec = []

    # Slide window
    while True:
        if cur_slide + spec_length > time:
            break

        spec_mag = mag_T[:, int(cur_slide + 0.5): int(cur_slide + spec_length + 0.5)]

        # Preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)
        utterances_spec.append(spec_mag)

        cur_slide += spec_hop_length

    return utterances_spec, intervals


def visualize(embeddings, labels):
    with open(os.path.join(consts.projections_folder, 'metadata.tsv', 'w')) as metadata:
        for label in labels:
            metadata.write(f'speaker_{label}\n')

    sess = InteractiveSession()

    with tf.device("/cpu:0"):
        embedding = tf.Variable(embeddings, trainable=False, name='diarization')
        global_variables_initializer().run()
        saver = Saver()
        writer = FileWriter(consts.projections_folder, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding'
        embed.metadata_path = os.path.join(consts.projections_folder, 'metadata.tsv')

        projector.visualize_embeddings(writer, config)

        saver.save(sess, os.path.join(consts.projections_folder, 'model.ckpt'))
