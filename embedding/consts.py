class Params(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


# Contains *.wav file and ground_truth.txt file with true map
audio_dir = './sample'

# File name with ground truth map
ground_truth_map_file = 'ground_truth.txt'
# Log file name
log_file = 'log.txt'
# File name with ground result map
result_map_file = 'result.txt'

# Default UMAP params
umap_params = Params()
umap_params.name = 'UMAP'
umap_params.n_components = 2
umap_params.n_neighbors = 15

# Default t-SNE params
tsne_params = Params()
tsne_params.name = 't-SNE'
tsne_params.n_components = 2
tsne_params.n_iter = 3000
tsne_params.learning_rate = 250
tsne_params.perplexity = 30

# Default HDBSCAN params
hdbscan_params = Params()
hdbscan_params.name = 'HDBSCAN'
hdbscan_params.min_cluster_size = 5
hdbscan_params.min_samples = 10

# Default DBSCAN params
dbscan_params = Params()
dbscan_params.name = 'DBSCAN'
dbscan_params.eps = 0.5
dbscan_params.min_samples = 5

# Default slide window params
slide_window_params = Params()
slide_window_params.embedding_per_second = 1.8
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
