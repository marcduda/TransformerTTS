import tensorflow as tf


def norm_tensor(tensor):
    return tf.math.divide(
        tf.math.subtract(
            tensor,
            tf.math.reduce_min(tensor)
        ),
        tf.math.subtract(
            tf.math.reduce_max(tensor),
            tf.math.reduce_min(tensor)
        )
    )


def neg_one_one_norm(tensor, axis=1):
    # Subtract the mean, and scale to the interval [-1,1]
    tensor_minusmean = tensor - tf.math.reduce_mean(tensor, axis=axis)
    return tensor_minusmean / tf.abs(tf.reduce_max(tensor_minusmean, axis=axis))