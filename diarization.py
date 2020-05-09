from embedding import cluster_utils, consts, new_utils, toolkits
from visualization.viewer import PlotDiar

toolkits.initialize_GPU(consts.nn_params.gpu)

wav = new_utils._get_audio(consts.audio_dir)
specs, intervals = new_utils.slide_window(audio_path=wav[0],
                                          embedding_per_second=consts.slide_window_params.embedding_per_second,
                                          overlap_rate=consts.slide_window_params.overlap_rate)

embeddings = new_utils.generate_embeddings(specs)
embeddings = cluster_utils.umap_transform(embeddings)

predicted_labels = cluster_utils.cluster_by_hdbscan(embeddings)

"""voices_pull_embeddings, true_labels = new_utils.load_voices_pull(consts.pull_dir)
voices_pull_embeddings = cluster_utils.umap_transform(voices_pull_embeddings)
classifier = cluster_utils.setup_knn(voices_pull_embeddings, true_labels)
identified_speakers = cluster_utils.identify_speakers(classifier, embeddings, predicted_labels)"""

ground_truth_map = new_utils.ground_truth_map(consts.audio_dir)
result_map = new_utils.result_map(intervals, predicted_labels)

der = new_utils.der(ground_truth_map, result_map)

plot = PlotDiar(true_map=ground_truth_map, map=result_map, wav=consts.audio_dir, gui=True, size=(24, 6))
plot.draw_true_map()
plot.draw_map()
plot.show()

new_utils.save_and_report(plot=plot,
                          result_map=result_map,
                          dim_reduce_params=consts.umap_params,
                          cluster_params=consts.hdbscan_params,
                          der=der)
