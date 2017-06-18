import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# read mnist data
mnist = input_data.read_data_sets("data",one_hot=True)

batchSize = 100
numIters = 1001
displayStep = 50
LOGPATH = "tensorboard/7"

tf.reset_default_graph()
# placeholders 
X = tf.placeholder(tf.float32, [None,28,28,1], name="X")
Y_ = tf.placeholder(tf.float32,[None,10], name = "labels")
lr = tf.placeholder(tf.float32)
pKeep = tf.placeholder(tf.float32)

# weights and biases
W1 = tf.Variable(tf.truncated_normal([5,5,1,4], stddev=0.1), name="W1")
b1 = tf.Variable(tf.ones([4])/10, name="b1")
W2 = tf.Variable(tf.truncated_normal([5,5,4,8], stddev=0.1), name = "W2")
b2 = tf.Variable(tf.ones([8])/10, name="b2")
W3 = tf.Variable(tf.truncated_normal([4,4,8,12], stddev=0.1), name="W3")
b3 = tf.Variable(tf.ones([12])/10, name="b3")
W4 = tf.Variable(tf.truncated_normal([12*7*7,200], stddev=0.1), name="W4")
b4 = tf.Variable(tf.ones([200])/10, name="b4")
W5 = tf.Variable(tf.truncated_normal([200,10], stddev=0.1), name="W5")
b5 = tf.Variable(tf.ones([10])/10, name="b5")


# for tensorboard plotting
tf.summary.histogram('weights-w1',W1)
tf.summary.histogram('weights-w2',W2)
tf.summary.histogram('weights-w3',W3)
tf.summary.histogram('weights-w4',W4)
tf.summary.histogram('weights-w5',W5)

tf.summary.histogram('bias-b1',b1)
tf.summary.histogram('bias-b2',b2)
tf.summary.histogram('bias-b3',b3)
tf.summary.histogram('bias-b4',b4)
tf.summary.histogram('bias-b5',b5)

init = tf.global_variables_initializer()

# model
# convolution layer
stride = 1    # 28*28
Y1conv = tf.nn.conv2d(X,W1,strides=[1,stride,stride,1],padding='SAME')
Y1 = tf.nn.relu(Y1conv + b1)

stride = 2  # 14*14
Y2conv = tf.nn.conv2d(Y1,W2,strides=[1,stride,stride,1],padding='SAME')
Y2 = tf.nn.relu(Y2conv + b2)

stride = 2   # 7*7
Y3conv = tf.nn.conv2d(Y2,W3,strides=[1,stride,stride,1],padding='SAME')
Y3 = tf.nn.relu(Y3conv + b3)

YY = tf.reshape(Y3,shape=[-1,7*7*12])
Y4 = tf.nn.relu(tf.matmul(YY,W4) + b4)

Ylogits = tf.matmul(Y4,W5) + b5
Y = tf.nn.softmax(Ylogits)

# loss/cost function 
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_, logits=Ylogits))*100
tf.summary.scalar('loss',loss)

# gradient
train_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuracy',accuracy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# save path
mergerd_summary = tf.summary.merge_all()
writer_train = tf.summary.FileWriter(LOGPATH+"/train")
writer_test = tf.summary.FileWriter(LOGPATH+"/test")
# writer_train.add_graph(sess.graph)

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0
testData = {X:np.reshape(mnist.test.images, (-1,28,28,1)), Y_:mnist.test.labels, pKeep:1.0} 

for i in range(numIters):
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    batchX, batchY = mnist.train.next_batch(batchSize)
    trainData = {X:np.reshape(batchX, (-1, 28,28,1)), Y_:batchY, lr:learning_rate, pKeep:0.75}
    # train
    sess.run(train_step,feed_dict=trainData)

    if i%displayStep == 0:
        [train_accuracy, cost, s] = sess.run([accuracy, loss,mergerd_summary], feed_dict=trainData)
        writer_train.add_summary(s, i)
        writer_train.flush()
        print("Train: " + str(i) + ": accuracy:" + str(train_accuracy) + " loss: " + str(cost))
        # test graph
        [test_accuracy, cost, s] = sess.run([accuracy, loss, mergerd_summary], feed_dict=testData)
        writer_test.add_summary(s, i)
        writer_test.flush()
        print("Test: " + str(i) + ": accuracy:" + str(test_accuracy) + " loss: " + str(cost))







