
def _init_model_with_state_dict(state):

    rnn_type = "LSTM" if "rnn_type" not in state.keys() else state["rnn_type"]
    use_dropout = 0.0 if "use_dropout" not in state.keys() else state["use_dropout"]
    use_word_dropout = (
        0.0 if "use_word_dropout" not in state.keys() else state["use_word_dropout"]
    )
    use_locked_dropout = (
        0.0
        if "use_locked_dropout" not in state.keys()
        else state["use_locked_dropout"]
    )
    train_initial_hidden_state = (
        False
        if "train_initial_hidden_state" not in state.keys()
        else state["train_initial_hidden_state"]
    )
    beta = 1.0 if "beta" not in state.keys() else state["beta"]
    weights = None if "weight_dict" not in state.keys() else state["weight_dict"]
    reproject_embeddings = True if "reproject_embeddings" not in state.keys() else state["reproject_embeddings"]
    if "reproject_to" in state.keys():
        reproject_embeddings = state["reproject_to"]

    model = SequenceTagger(
        hidden_size=state["hidden_size"],
        embeddings=state["embeddings"],
        tag_dictionary=state["tag_dictionary"],
        tag_type=state["tag_type"],
        use_crf=state["use_crf"],
        use_rnn=state["use_rnn"],
        rnn_layers=state["rnn_layers"],
        dropout=use_dropout,
        word_dropout=use_word_dropout,
        locked_dropout=use_locked_dropout,
        train_initial_hidden_state=train_initial_hidden_state,
        rnn_type=rnn_type,
        beta=beta,
        loss_weights=weights,
        reproject_embeddings=reproject_embeddings,
    )
    model.load_state_dict(state["state_dict"])
    return model