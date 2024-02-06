import tensorflow as tf

input=tf.constant([1.5,4.4,3.3], dtype=tf.float32)
weights=tf.constant([0.2,0.1,0.3], dtype=tf.float32)
threshold=tf.constant(0.7, dtype=tf.float32)


def muchulooch_pits_mode(input,weights,threshold):
    wei_sum = tf.reduce_sum(tf.multiply(input,weights))
    ouptput=tf.cond(wei_sum >= threshold ,lambda:1.0,lambda :0.0)
    return ouptput

ans=muchulooch_pits_mode(input,weights,threshold)
print("ans is:-", ans)

