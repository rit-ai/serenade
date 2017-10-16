import tensorflow as tf
import librosa

N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs

import numpy as np
megadeth,_ = read_audio_spectum("music/md.mp3")
tay,_ = read_audio_spectum("music/ts.mp3")

import random
def get_batches(batch_size=1,window_size=10):
    m_indices = random.sample([x for x in range(1025-window_size)],batch_size)
    t_indices = random.sample([x for x in range(1025-window_size)],batch_size)
    inputs = []
    outputs = []
    for m in m_indices:
        item = megadeth[m:m+window_size].reshape((1,-1))
        inputs.append(item[0])
        outputs.append([1,0])
    for t in t_indices:
        item = tay[t:t+window_size].reshape((1,-1))
        inputs.append(item[0])
        outputs.append([0,1])
    return np.asarray(inputs).astype(np.float32),np.asarray(outputs).astype(np.float32)

x = tf.placeholder(tf.float32,[None,4300])
y = tf.placeholder(tf.float32,[None,2])

w = tf.get_variable("w",[4300,2])
b = tf.get_variable("b",[2])

output = tf.matmul(x,w) + b

correct = tf.equal(tf.argmax(y,1),tf.argmax(output,1))
accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))

loss = tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y)
train_op = tf.train.AdamOptimizer().minimize(loss)
init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)
    for i in range(100):
        inputs,labels = get_batches(batch_size=20)
        feed = {
            x: inputs,
            y: labels
        }
        sess.run(train_op,feed_dict=feed)
	print sess.run(accuracy,feed_dict=feed)

