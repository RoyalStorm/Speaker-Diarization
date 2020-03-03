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
import datetime

import matplotlib.pyplot as plt
import numpy as np

import uisrnn

SAVED_MODEL_NAME = './src/last_model/ru_model_' + datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.uis-rnn'''


def diarization_experiment(model_args, training_args):
    """Experiment pipeline.

    Load dataset --> train model --> test model --> output result

    Args:
      model_args: model configurations
      training_args: training configurations
      inference_args: inference configurations
    """

    train_data = np.load('./ghostvlad/training_data.npz', allow_pickle=True)
    train_sequence = train_data['train_sequence']
    train_cluster_id = train_data['train_cluster_id']
    train_sequence_list = [seq.astype(float) + 0.00001 for seq in train_sequence]
    train_cluster_id_list = [np.array(cid).astype(str) for cid in train_cluster_id]

    model = uisrnn.UISRNN(model_args)

    # training
    train_loss = model.fit(train_sequence_list, train_cluster_id_list, training_args)
    model.save(SAVED_MODEL_NAME)

    iterations = np.arange(0, training_args.train_iteration)

    plt.style.use('ggplot')
    plt.figure()

    plt.plot(iterations, train_loss, label='train_loss')

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss')
    plt.legend()
    plt.show()

    '''
    # testing
    # we can also skip training by calling：
    model.load(SAVED_MODEL_NAME)
    for (test_sequence, test_cluster_id) in zip(test_sequences, test_cluster_ids):
      predicted_label = model.predict(test_sequence, inference_args)
      predicted_labels.append(predicted_label)
      accuracy = uisrnn.compute_sequence_match_accuracy(
          test_cluster_id, predicted_label)
      test_record.append((accuracy, len(test_cluster_id)))
      print('Ground truth labels:')
      print(test_cluster_id)
      print('Predicted labels:')
      print(predicted_label)
      print('-' * 80)
  
    output_string = uisrnn.output_result(model_args, training_args, test_record)
  
    print('Finished diarization experiment')
    print(output_string)
    '''


def main():
    """The main function."""
    model_args, training_args, _ = uisrnn.parse_arguments()
    model_args.observation_dim = 512
    model_args.rnn_depth = 1
    model_args.rnn_hidden_size = 512

    training_args.enforce_cluster_id_uniqueness = False
    training_args.batch_size = 20
    training_args.learning_rate = 1e-4
    training_args.train_iteration = 30
    training_args.num_permutations = 20
    # training_args.grad_max_norm = 5.0
    training_args.learning_rate_half_life = 100

    diarization_experiment(model_args, training_args)


if __name__ == '__main__':
    main()
