def neuron_groups(img, layer, n_groups=6, attr_classes=[]):
    # Compute activations

    with tf.Graph().as_default(), tf.Session():
        t_input = tf.placeholder_with_default(img, [None, None, 3])
        T = render.import_model(model, t_input, t_input)
        acts = T(layer).eval()

    # We'll use ChannelReducer (a wrapper around scikit learn's factorization tools)
    # to apply Non-Negative Matrix factorization (NMF).

    nmf = ChannelReducer(n_groups, "NMF")

    #TODO this is misleading...because it actually uses the reducer here too. It's not just reshaping
    #TODO why are they using the "CHANNEL"-reducer for SPATIAL??
    spatial_factors = nmf.fit_transform(acts)[0].transpose(2, 0, 1).astype("float32")

    #TODO grouping happens here?
    channel_factors = nmf._reducer.components_.astype("float32")

    # Let's organize the channels based on their horizontal position in the image


    #TODO this is just for the visualization!!??
    x_peak = np.argmax(spatial_factors.max(1), 1)
    ns_sorted = np.argsort(x_peak)
    spatial_factors = spatial_factors[ns_sorted]
    channel_factors = channel_factors[ns_sorted]

    # And create a feature visualziation of each group


    #TODO: Note that the feature viz doesn't include the spatial data
    param_f = lambda: param.image(80, batch=n_groups)
    obj = sum(objectives.direction(layer, channel_factors[i], batch=i)
              for i in range(n_groups))
    group_icons = render.render_vis(model, obj, param_f, verbose=False)[-1]

    # We'd also like to know about attribution

    # First, let's turn each group into a vector over activations
    #TODO: activations == spatial_factors??
    group_vecs = [spatial_factors[i, ..., None] * channel_factors[i]
                  for i in range(n_groups)]

    #TODO: the just print it?? do that somewhere separetly
    attrs = np.asarray([raw_class_group_attr(img, layer, attr_class, group_vecs)
                        for attr_class in attr_classes])

    print attrs

    lucid_svelte.GroupWidget({
        "img": _image_url(img),
        "n_groups": n_groups,
        "spatial_factors": [_image_url(factor[..., None] / np.percentile(spatial_factors, 99) * [1, 0, 0]) for factor in
                            spatial_factors],
        "group_icons": [_image_url(icon) for icon in group_icons]
    })
