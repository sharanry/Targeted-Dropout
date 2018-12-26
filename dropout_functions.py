import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.keras import backend as K


def targeted_weight_dropout(w, targ_rate, drop_rate, training):
    drop_rate = drop_rate
    targ_rate = targ_rate

    w_shape = w.shape
    w = tf.reshape(w, [-1, w_shape[-1]])
    norm = tf.abs(w)
    idx = tf.to_int32(targ_rate * tf.to_float(tf.shape(w)[0]))
    threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
    mask = norm < threshold[None, :]

    if not training:
        w = (1 - tf.to_float(mask)) * w
        w = tf.reshape(w, w_shape)
        return w

    mask = tf.where(
      tf.logical_and((1. - drop_rate) < tf.random_uniform(tf.shape(w)), mask),
      tf.ones_like(w, dtype=tf.float32), tf.zeros_like(w, dtype=tf.float32))
    w = (1 - mask) * w
    w = tf.reshape(w, w_shape)
    return w

def weight_pruning(w, prune_rate):
    w_shape = w.shape
    w = tf.reshape(w, [-1, w_shape[-1]])
    norm = tf.abs(w)
    idx = tf.to_int32(prune_rate * tf.to_float(tf.shape(w)[0]))
    print(w.shape, norm.shape, idx, tf.contrib.framework.sort(norm, axis=0))
    threshold = tf.contrib.framework.sort(norm, axis=0)[idx]
    mask = norm < threshold[None, :]

    # mask = tf.where(
    #   tf.logical_and((1. - drop_rate) < tf.random_uniform(tf.shape(w)), mask),
    #   tf.ones_like(w, dtype=tf.float32), tf.zeros_like(w, dtype=tf.float32))
    w = mask * w
    w = tf.reshape(w, w_shape)
    return w

def targeted_unit_dropout(x, targ_rate, drop_rate, training):
    drop_rate = drop_rate
    targ_rate = targ_rate

    w = tf.reshape(x, [-1, x.shape[-1]])
    norm = tf.norm(w, axis=0)
    idx = int(targ_rate * int(w.shape[1]))
    sorted_norms = tf.contrib.framework.sort(norm)
    threshold = sorted_norms[idx]
    mask = (norm < threshold)[None, :]
    mask = tf.tile(mask, [w.shape[0], 1])

    mask = tf.where(
        tf.logical_and((1. - drop_rate) < tf.random_uniform(tf.shape(w)),
                        mask), tf.ones_like(w, dtype=tf.float32),
        tf.zeros_like(w, dtype=tf.float32))
    x = tf.reshape((1 - mask) * w, x.shape)
    return x

class TargetedDense(tf.keras.layers.Dense):
    def __init__(self,
                units,
                targeted_dropout_type,   
                activation=None,
                use_bias=True,
                kernel_initializer=tf.keras.initializers.glorot_normal(seed=42),
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                targeted_dropout_rate=0.25,
                dropout_rate=0.25,
                **kwargs):
        super(TargetedDense, self).__init__(units=units,
                activation=activation,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs)
        self.targeted_dropout_rate = targeted_dropout_rate
        self.dropout_rate = dropout_rate
        self.targeted_dropout_type = targeted_dropout_type
    def call(self, inputs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        # Newly added lines for targeted dropout 
        # print(self.kernel[:5])
        if(self.targeted_dropout_type=="weight"):
            self.kernel.assign(targeted_weight_dropout(self.kernel, self.targeted_dropout_rate, self.dropout_rate, K.learning_phase()))
        elif(self.targeted_dropout_type=="unit"):
            self.kernel.assign(targeted_unit_dropout(self.kernel, self.targeted_dropout_rate, self.dropout_rate, K.learning_phase()))
        else:
            raise ValueError("Should be of 'weight' or 'unit'")

        # print(self.kernel[:5])
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.kernel)

        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


def get_targeted_dropout_model(targeted_dropout_type, conf):
    drop_rate, target_rate = conf
    return tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    TargetedDense(1000, targeted_dropout_type, activation=tf.nn.relu, use_bias=False, targeted_dropout_rate=target_rate, dropout_rate=drop_rate),
    TargetedDense(1000, targeted_dropout_type, activation=tf.nn.relu, use_bias=False, targeted_dropout_rate=target_rate, dropout_rate=drop_rate),
    TargetedDense(500, targeted_dropout_type, activation=tf.nn.relu, use_bias=False, targeted_dropout_rate=target_rate, dropout_rate=drop_rate),
    TargetedDense(200, targeted_dropout_type, activation=tf.nn.relu, use_bias=False, targeted_dropout_rate=target_rate, dropout_rate=drop_rate),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed=42))
    ])

def prune_units(model, pruning_perc=10):
    pruned_model = tf.keras.models.Sequential()
    prev_mask = []
    pruned_model.add(tf.keras.layers.Flatten())
    for weights in model.trainable_weights[:-1]:
#         weights = layer.weights[0].numpy()
        weights = weights.numpy()
        c_norm = np.linalg.norm(weights, ord=2, axis=0)
        tres = np.percentile(c_norm, pruning_perc)
        mask = c_norm >= tres
        
        weights = weights[:, mask]
#         print(prev_mask)
        if type(prev_mask)==np.bool_ :
            weights = weights[prev_mask,:]
        elif len(prev_mask)!=0:
            weights = weights[prev_mask,:]
        prev_mask=mask
#         if type(prev_mask)!=list:
#             prev_mask = [prev_mask]
        layer_new = tf.keras.layers.Dense(weights.shape[1], activation=tf.nn.relu, use_bias=False, weights=[weights])
        pruned_model.add(layer_new)

    weights = model.trainable_weights[-1]
    weights = weights.numpy()
    if type(prev_mask)==np.bool_ :
        weights = weights[prev_mask,:]
        layer_new = tf.keras.layers.Dense(weights.shape[1], activation=tf.nn.sigmoid, use_bias=False, weights=[weights])
        pruned_model.add(layer_new)
    elif len(prev_mask)!=0:
        weights = weights[prev_mask,:]
        layer_new = tf.keras.layers.Dense(weights.shape[1], activation=tf.nn.sigmoid, use_bias=False, weights=[weights])
        pruned_model.add(layer_new)
    return pruned_model