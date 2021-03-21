import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def _wrap_layer(name,
                input_layer,
                build_output,
                dropout_rate=0.0,
                trainable=True):
    """Wrap layers with residual, normalization and dropout.
    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name='%s-Dropout' % name,
        )(build_output)
    else:
        dropout_layer = build_output
    if isinstance(input_layer, list):
        input_layer = input_layer[0]
    print("INPUT LAYER", input_layer)
    print("BUILD OUTPUT", build_output)
    print("DROPOUT", dropout_layer)
    add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
    normal_layer = keras.layers.LayerNormalization(
        trainable=trainable,
        name='%s-Norm' % name,
    )(add_layer)
    return normal_layer


def attention_builder(name,
                      head_num,
                      key_dim,
                      input_layer,
                      trainable=True):
    """Get multi-head self-attention builder.
    :param input_layer:
    :param key_dim:
    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param trainable: Whether the layer is trainable.
    :return:
    """

    def _attention_builder(query, value):
        return layers.MultiHeadAttention(
            num_heads=head_num,
            key_dim=key_dim,
            trainable=trainable,
            name=name,
        )(query, value)

    return _attention_builder(query=input_layer, value=input_layer)


def feed_forward_builder(name,
                         hidden_dim,
                         activation,
                         input_layer,
                         trainable=True):
    """Get position-wise feed-forward layer builder.
    :param input_layer:
    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :param trainable: Whether the layer is trainable.
    :return:
    """

    def _feed_forward_builder(x):
        return layers.Dense(
            units=hidden_dim,
            activation=activation,
            trainable=trainable,
            name=name,
        )(x)

    return _feed_forward_builder(input_layer)


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          key_dim,
                          feed_forward_activation='tanh',
                          dropout_rate=0.0,
                          trainable=True):
    """Multi-head self-attention and feed-forward layer.
    :param key_dim: depth of query and key
    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_output=attention_builder(
            key_dim=key_dim,
            name=attention_name,
            head_num=head_num,
            trainable=trainable,
            input_layer=input_layer
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    feed_forward_layer = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_output=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=key_dim,
            activation=feed_forward_activation,
            trainable=trainable,
            input_layer=input_layer
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
    )
    return feed_forward_layer
