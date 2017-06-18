import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import math

# read mnist data
mnist = input_data.read_data_sets("data",one_hot=True)

batchSize = 100
numIters = 5001
displayStep = 50
LOGPATH = "tensorboard/6"

tf.reset_default_graph()
# placeholders 
X = tf.placeholder(tf.float32, [None,28,28,1], name="X")
Y_ = tf.placeholder(tf.float32,[None,10], name = "labels")
lr = tf.placeholder(tf.float32)
pKeep = tf.placeholder(tf.float32)

# weights and biases
W1 = tf.Variable(tf.truncated_normal([28*28,128], stddev=0.1), name="W1")
b1 = tf.Variable(tf.ones([128])/10, name="b1")

W2 = tf.Variable(tf.truncated_normal([128,256], stddev=0.1), name = "W2")
b2 = tf.Variable(tf.ones([256])/10, name="b2")

W3 = tf.Variable(tf.truncated_normal([256,10]), name="W3")
b3 = tf.Variable(tf.ones([10])/10, name="b3")

# for tensorboard plotting
tf.summary.histogram('weights-w1',W1)
tf.summary.histogram('weights-w2',W2)
tf.summary.histogram('weights-w3',W3)

tf.summary.histogram('bias-b1',b1)
tf.summary.histogram('bias-b2',b2)
tf.summary.histogram('bias-b3',b3)

init = tf.global_variables_initializer()

X = tf.reshape(X,[-1,784])
# model
# multi layer network - we use sigmoid activation for first 2 layers, and softmax for last layer
Y1 = tf.nn.relu(tf.matmul(X,W1) + b1)
Y1d = tf.nn.dropout(Y1,pKeep)

Y2 = tf.nn.relu(tf.matmul(Y1d,W2) + b2)
Y2d = tf.nn.dropout(Y2,pKeep)

Ylogits = tf.matmul(Y2d,W3) + b3
Y = tf.nn.softmax(Ylogits)

# loss/cost function - 
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
writer = tf.summary.FileWriter(LOGPATH)
writer.add_graph(sess.graph)

# learning rate decay
max_learning_rate = 0.003
min_learning_rate = 0.0001
decay_speed = 2000.0

for i in range(numIters):
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
    batchX, batchY = mnist.train.next_batch(batchSize)
    trainData = {X:batchX, Y_:batchY, lr: learning_rate, pKeep:0.75}
    # train
    sess.run(train_step,feed_dict=trainData)

    if i%5 == 0:
        [train_accuracy, s] = sess.run([accuracy, mergerd_summary], feed_dict=trainData)
        writer.add_summary(s, i)
    
    if i%displayStep == 0:
        acc, cost = sess.run([accuracy, loss], feed_dict=trainData)
        print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(cost))

testData = {X:mnist.test.images, Y_:mnist.test.labels, pKeep:1.0} 
acc, cost = sess.run([accuracy,loss], feed_dict=testData)
print("*****Test accuracy:" + str(acc) + " loss: " + str(cost) + "*****\n")






