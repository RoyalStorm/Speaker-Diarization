from embedding import model


class Params(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


# Contains *.wav file and ref.txt file with true map
audio_dir = './sample'

# File name with ground truth map
reference_file = 'ref.txt'
# File name with ground result map
result_map_file = 'result.txt'

# Default slide window params
slide_window_params = Params()
slide_window_params.embedding_per_second = 2
slide_window_params.overlap_rate = 0.4

slide_window_params.nfft = 512
slide_window_params.spec_len = 250
slide_window_params.win_length = 400
slide_window_params.hop_length = 160
slide_window_params.sampling_rate = 16_000
slide_window_params.normalize = True

# Default NN params
nn_params = Params()
nn_params.weights = './embedding/pre_trained/weights.h5'
nn_params.input_dim = (257, None, 1)
nn_params.num_classes = 5994

nn_params.mode = 'eval'  # 'train'
nn_params.gpu = ''
nn_params.net = 'resnet34s'  # 'resnet34s' or 'resnet34l'
nn_params.ghost_cluster = 2
nn_params.vlad_cluster = 8
nn_params.bottleneck_dim = 512
nn_params.aggregation_mode = 'gvlad'  # 'avg', 'vlad' or 'gvlad'
nn_params.loss = 'softmax'  # 'softmax' or 'amsoftmax'
nn_params.test_type = 'normal'  # 'normal', 'hard' or 'extend'
nn_params.optimizer = 'adam'  # 'sgd'

model = model.vggvox_resnet2d_icassp(input_dim=nn_params.input_dim,
                                     num_class=nn_params.num_classes,
                                     mode=nn_params.mode,
                                     params=nn_params)
model.load_weights(nn_params.weights, by_name=True)
