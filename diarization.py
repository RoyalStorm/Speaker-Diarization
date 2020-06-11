from embedding import cluster_utils, consts, new_utils, toolkits
from visualization.viewer import PlotDiar

toolkits.initialize_GPU(consts.nn_params.gpu)

wav = new_utils.find_wav(consts.audio_dir)
specs, intervals = new_utils.slide_window(audio_path=wav,
                                          embedding_per_second=consts.slide_window_params.embedding_per_second,
                                          overlap_rate=consts.slide_window_params.overlap_rate)

embeddings = new_utils.generate_embeddings(specs)
embeddings = cluster_utils.umap_transform(embeddings)

predicted_labels = cluster_utils.cluster_by_hdbscan(embeddings)

reference = new_utils.reference(consts.audio_dir)
hypothesis = new_utils.result_map(intervals, predicted_labels)

der = new_utils.der(reference, hypothesis)

plot = PlotDiar(true_map=reference, map=hypothesis, wav=consts.audio_dir, gui=True, size=(24, 6))
plot.draw_true_map()
plot.draw_map()
plot.show()

new_utils.save_and_report(plot=plot, result_map=hypothesis, der=der['diarization error rate'])
