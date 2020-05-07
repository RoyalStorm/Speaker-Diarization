from embedding import cluster_utils, model, consts, new_utils, toolkits
from visualization.viewer import PlotDiar

toolkits.initialize_GPU(consts.nn_params.gpu)

model = model.vggvox_resnet2d_icassp(input_dim=consts.nn_params.input_dim,
                                     num_class=consts.nn_params.num_classes,
                                     mode=consts.nn_params.mode,
                                     params=consts.nn_params)
model.load_weights(consts.nn_params.weights, by_name=True)

specs, intervals = new_utils.slide_window(audio_folder=consts.audio_dir,
                                          embedding_per_second=consts.slide_window_params.embedding_per_second,
                                          overlap_rate=consts.slide_window_params.overlap_rate)

embeddings = new_utils.generate_embeddings(model, specs)
embeddings = cluster_utils.umap_transform(embeddings)

predicted_labels = cluster_utils.cluster_by_hdbscan(embeddings)

ground_truth_map = new_utils.ground_truth_map(consts.audio_dir)
result_map = new_utils.result_map(intervals, predicted_labels)

der = new_utils.der(ground_truth_map, result_map)

plot = PlotDiar(true_map=ground_truth_map, map=result_map, wav=consts.audio_dir, gui=True, size=(24, 6))
plot.draw_true_map()
plot.draw_map()
plot.show()

new_utils.save_and_report(plot, result_map, der)
