from embedding import cluster_utils, model, consts, new_utils, toolkits
from visualization.viewer import PlotDiar

# Step 1. We may initialize GPU device, but it optional.
toolkits.initialize_GPU(consts.nn_params.gpu)

# Step 2. First of all, we need to create model and load weights.
model = model.vggvox_resnet2d_icassp(input_dim=consts.nn_params.input_dim,
                                     num_class=consts.nn_params.num_classes,
                                     mode=consts.nn_params.mode,
                                     params=consts.nn_params)
model.load_weights(consts.nn_params.weights, by_name=True)

# Step 3. Now we need to apply slide window to selected audio.
specs, intervals = new_utils.slide_window(audio_folder=consts.audio_folder,
                                          embedding_per_second=consts.slide_window_params.embedding_per_second,
                                          overlap_rate=consts.slide_window_params.overlap_rate)

# Step 4. Generate embeddings from slices audio.
embeddings = new_utils.generate_embeddings(model, specs)

# Step 5. It step optionally, but I highly recommend reduce embeddings dimension to 2 or 3.
embeddings = cluster_utils.umap_transform(embeddings)

# Step 6. Cluster all embeddings. Labels may contains noise (label will be "-1"), it should be remove from list.
predicted_labels = cluster_utils.cluster_by_hdbscan(embeddings)

# Step 7. We can visualize generated embeddings with predicted labels.
# new_utils.visualize(embeddings, predicted_label)

# Step 8. Read real segments from file.
ground_truth_map = new_utils.ground_truth_map(consts.audio_folder)

# Step 9. Get result segments.
result_map = new_utils.result_map(intervals, predicted_labels)

# Step 10. Get DER (diarization error rate).
der = new_utils.der(ground_truth_map, result_map)

# Step 11. And now we can show both plots (ground truth and result).
plot = PlotDiar(true_map=ground_truth_map, map=result_map, wav=consts.audio_folder, gui=True, size=(24, 6))
plot.draw_true_map()
plot.draw_map()
plot.show()

# Step 12. Save timestamps, der, plot and report about it.
new_utils.save_and_report(plot, result_map, der)
