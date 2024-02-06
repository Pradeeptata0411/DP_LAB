import tensorflow as tf

input1 = tf.constant([0, 0], dtype=tf.float32)
input2 = tf.constant([0, 1], dtype=tf.float32)
input3 = tf.constant([1, 0], dtype=tf.float32)
input4 = tf.constant([1, 1], dtype=tf.float32)

weights = tf.constant([0.5, 0.8], dtype=tf.float32)

threshold = tf.constant(0.9, dtype=tf.float32)

def muchulooch_pits_mode(input, weights, threshold):
    wei_sum = tf.reduce_sum(tf.multiply(input, weights))
    output = tf.cond(wei_sum >= threshold, lambda: 1.0, lambda: 0.0)
    return output

ans1 = muchulooch_pits_mode(input1, weights, threshold)
ans2 = muchulooch_pits_mode(input2, weights, threshold)
ans3 = muchulooch_pits_mode(input3, weights, threshold)
ans4 = muchulooch_pits_mode(input4, weights, threshold)

print("AND GATE OUTPUTS: ")
print("Input 1, Output:", ans1)
print("Input 2, Output:", ans2)
print("Input 3, Output:", ans3)
print("Input 4, Output:", ans4)

import tensorflow as tf

input1 = tf.constant([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 1]], dtype=tf.float32)

weights = tf.constant([0.5, 0.6, 0.2], dtype=tf.float32)

threshold = tf.constant(1, dtype=tf.float32)

def muchulooch_pits_mode(input, weights, threshold):
    wei_sum = tf.reduce_sum(tf.multiply(input, weights), axis=1)
    output = tf.cast(tf.greater_equal(wei_sum, threshold), dtype=tf.float32)
    return output

ans1 = muchulooch_pits_mode(input1, weights, threshold)


print("AND GATE OUTPUTS for 3 inputs: ")
print("Input's, Output:", ans1)