import os
from datetime import datetime

import librosa
import numpy as np
import tensorflow as tf
from pyannote.core import Segment, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate
from tensorboard.plugins import projector
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1 import global_variables_initializer
from tensorflow.compat.v1.summary import FileWriter
from tensorflow.compat.v1.train import Saver

from embedding import consts
from embedding.consts import model


def _append_2_dict(speaker_slice, segment):
    key, value = list(segment.items())[0]
    time_dict = {'start': int(value[0] + 0.5), 'stop': int(value[1] + 0.5)}

    if key in speaker_slice:
        speaker_slice[key].append(time_dict)
    else:
        speaker_slice[key] = [time_dict]

    return speaker_slice


def _arrange_result(predicted_labels, time_spec_rate):
    last_label = predicted_labels[0]
    speaker_slice = {}
    j = 0

    for i, label in enumerate(predicted_labels):
        if label == last_label:
            continue

        speaker_slice = _append_2_dict(speaker_slice, {last_label: (time_spec_rate * j, time_spec_rate * i)})
        j = i
        last_label = label

    speaker_slice = _append_2_dict(speaker_slice,
                                   {last_label: (time_spec_rate * j, time_spec_rate * (len(predicted_labels)))})

    return speaker_slice


def _beautify_time(time_in_milliseconds):
    minute = time_in_milliseconds // 1_000 // 60
    second = (time_in_milliseconds - minute * 60 * 1_000) // 1_000
    millisecond = time_in_milliseconds % 1_000

    time = f'{minute}:{second:02d}.{millisecond}'

    return time


def _gen_map(intervals):  # interval slices to map table
    slice_len = [sliced[1] - sliced[0] for sliced in intervals.tolist()]
    map_table = {}  # vad erased time to origin time, only split points
    idx = 0

    for i, sliced in enumerate(intervals.tolist()):
        map_table[idx] = sliced[0]
        idx += slice_len[i]

    map_table[sum(slice_len)] = intervals[-1, -1]

    return map_table


def find_wav(dir):
    for file in os.listdir(dir):
        if file.endswith('.wav'):
            return os.path.join(dir, file)

    raise FileExistsError(f'Folder "{dir}" not contains *.wav file')


def _vad(audio_path, sr):
    audio, _ = librosa.load(audio_path, sr=sr)
    intervals = librosa.effects.split(audio, top_db=20)
    audio_output = []

    for sliced in intervals:
        audio_output.extend(audio[sliced[0]:sliced[1]])

    return np.array(audio_output), (intervals / sr * 1000).astype(int)


def der(reference, hypothesis):
    def convert(map):
        annotation = Annotation()

        for cluster in sorted(map.keys()):
            for row in map[cluster]:
                annotation[Segment(row['start'] / 1000, row['stop'] / 1000)] = str(cluster)

        return annotation

    return DiarizationErrorRate(convert(reference), convert(hypothesis), detailed=True)


def generate_embeddings(specs):
    embeddings = []

    for spec in specs:
        spec = np.expand_dims(np.expand_dims(spec, 0), -1)
        v = model.predict(spec)
        embeddings.append(list(v))

    embeddings = np.array(embeddings)[:, 0, :].astype(float)

    return embeddings


def reference(dir):
    reference_file = None

    for file in os.listdir(dir):
        if file == consts.reference_file:
            reference_file = os.path.join(dir, file)
            break

    with open(reference_file, 'r') as file:
        reference = {}

        for line in file:
            start, stop, speaker = line.split(' ')[0], line.split(' ')[1], line.split(' ')[2].replace('\n', '')

            dt_start = datetime.strptime(start, '%M:%S.%f')
            dt_stop = datetime.strptime(stop, '%M:%S.%f')

            start = dt_start.minute * 60_000 + dt_start.second * 1_000 + dt_start.microsecond / 1_000
            stop = dt_stop.minute * 60_000 + dt_stop.second * 1_000 + dt_stop.microsecond / 1_000

            if speaker in reference.keys():
                reference[speaker].append({'start': start, 'stop': stop})
            else:
                reference[speaker] = [{'start': start, 'stop': stop}]

    return reference


def linear_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    return linear.T


def result_map(intervals, predicted_labels):
    # Speaker embedding every ? ms
    time_spec_rate = 1_000 * (1.0 / consts.slide_window_params.embedding_per_second) * (
            1.0 - consts.slide_window_params.overlap_rate)

    speaker_slice = _arrange_result(predicted_labels, time_spec_rate)

    map_table = _gen_map(intervals)
    keys = [*map_table]

    # Time map to origin wav (contains mute)
    for speaker, timestamps_list in sorted(speaker_slice.items()):
        for i, timestamp in enumerate(timestamps_list):
            s = 0
            e = 0

            for j, key in enumerate(keys):
                if s != 0 and e != 0:
                    break

                if s == 0 and key > timestamp['start']:
                    offset = timestamp['start'] - keys[j - 1]
                    s = map_table[keys[j - 1]] + offset

                if e == 0 and key > timestamp['stop']:
                    offset = timestamp['stop'] - keys[j - 1]
                    e = map_table[keys[j - 1]] + offset

            speaker_slice[speaker][i]['start'] = s
            speaker_slice[speaker][i]['stop'] = e

    return speaker_slice


def save_and_report(plot, result_map, der=None, dir=consts.audio_dir):
    # Make checkpoint
    checkpoint_dir = os.path.join(dir, f'{datetime.now():%Y%m%dT%H%M%S}')
    os.mkdir(checkpoint_dir)

    # Write result segments and DER in file
    with open(os.path.join(checkpoint_dir, consts.result_map_file), 'w') as result:
        plot.save(checkpoint_dir)

        for i, cluster in enumerate(sorted(result_map.keys())):
            if i != 0:
                result.write('\n')

            result.write(f'{cluster}\n')

            for segment in result_map[cluster]:
                result.write(f'{_beautify_time(segment["start"])} --> {_beautify_time(segment["stop"])}\n')

        result.write(f'\n{der}')

    print(f'Diarization done. All results saved in {checkpoint_dir}.')


def slide_window(audio_path, win_length=400, sr=16000, hop_length=160, n_fft=512, embedding_per_second=0.5,
                 overlap_rate=0.5):
    wav, intervals = _vad(audio_path, sr=sr)
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


def visualize(embeddings, predicted_labels, dir):
    dir = os.path.join(dir, 'projections')

    with open(os.path.join(dir, 'metadata.tsv'), 'w') as metadata:
        for label in predicted_labels:
            metadata.write(f'spk_{label}\n')

    sess = InteractiveSession()

    with tf.device("/cpu:0"):
        embedding = tf.Variable(embeddings, trainable=False, name='projections')
        global_variables_initializer().run()
        saver = Saver()
        writer = FileWriter(dir, sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = 'embedding'
        embed.metadata_path = 'metadata.tsv'

        projector.visualize_embeddings(writer, config)

        saver.save(sess, os.path.join(dir, 'model.ckpt'))
