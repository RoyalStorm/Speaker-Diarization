# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

import uisrnn


def diarization_experiment(model_args, training_args, inference_args):
    """Experiment pipeline.

    Load dataset --> train model --> test model --> output result

    Args:
      model_args: model configurations
      training_args: training configurations
      inference_args: inference configurations
    """

    # Train data
    train_data = np.load('./ghostvlad/data/training_data.npz', allow_pickle=True)

    train_sequences = train_data['train_sequence']
    train_cluster_ids = train_data['train_cluster_id']

    train_sequences = [seq.astype(float) + 0.00001 for seq in train_sequences]
    train_cluster_ids = [np.array(cid).astype(str) for cid in train_cluster_ids]

    # Test data
    """test_data = np.load('./ghostvlad/data/testing_data.npz', allow_pickle=True)

    test_sequences = test_data['train_sequence']
    test_cluster_ids = test_data['train_cluster_id']

    test_sequences = [seq.astype(float) + 0.00001 for seq in test_sequences]
    test_cluster_ids = [np.array(cid).astype(str) for cid in test_cluster_ids]"""

    model = uisrnn.UISRNN(model_args)

    # Training
    history = model.fit(train_sequences, train_cluster_ids, training_args)
    iterations = np.arange(0, training_args.train_iteration)

    model.save(f'./src/model/ru_model_{training_args.train_iteration}_{datetime.now().strftime("%m.%dT%H.%M")}.uis-rnn')
    with open('history.txt', 'w') as f:
        f.write(str(history))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=iterations,
        y=history['train_loss'],
        name='<b>train_loss</b>',
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=iterations,
        y=history['sigma2_prior'],
        name='<b>sigma2_prior</b>',
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=iterations,
        y=history['negative_log_likelihood'],
        name='<b>negative_log_likelihood</b>',
        connectgaps=True
    ))
    fig.add_trace(go.Scatter(
        x=iterations,
        y=history['regularization'],
        name='<b>regularization</b>',
        connectgaps=True
    ))
    fig.show()

    # Testing.
    # You can also try uisrnn.parallel_predict to speed up with GPU.
    # But that is a beta feature which is not thoroughly tested, so proceed with caution.
    """model.load('./src/last_model/ru_model_20200309T2107.uis-rnn')

    predicted_cluster_ids = []
    test_record = []

    for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
        predicted_cluster_id = model.predict(test_sequence, inference_args)
        predicted_cluster_ids.append(predicted_cluster_id)

        accuracy = uisrnn.compute_sequence_match_accuracy(list(test_cluster_id), list(predicted_cluster_id))
        test_record.append((accuracy, len(test_cluster_id)))

        print('Ground truth labels:')
        print(test_cluster_id)
        print('Ground truth labels len: ', len(test_cluster_id))
        print('Predicted labels:')
        print(predicted_cluster_id)
        print('Predicted labels len: ', len(predicted_cluster_id))
        print('-' * 80)

    output_string = uisrnn.output_result(model_args, training_args, test_record)

    print('Finished diarization experiment')
    print(output_string)"""


def train():
    """The train function."""
    model_args, training_args, inference_args = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 512

    training_args.enforce_cluster_id_uniqueness = False
    training_args.batch_size = 10
    training_args.learning_rate = 5e-5
    training_args.train_iteration = 2750
    training_args.num_permutations = 10
    # training_args.grad_max_norm = 5.0
    training_args.learning_rate_half_life = 0

    diarization_experiment(model_args, training_args, inference_args)


if __name__ == '__main__':
    train()
