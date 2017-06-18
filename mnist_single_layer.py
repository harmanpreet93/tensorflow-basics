import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("data",one_hot=True)

batchSize = 100
learningRate = 0.001
numIters =5001
displayStep = 50

X = tf.placeholder(tf.float32, [None,28,28,1])
Y_ = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.random_normal([784,10]))
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

X = tf.reshape(X,[-1,784])
# model
Y = tf.nn.softmax(tf.matmul(X,W) + b)

# loss/cost function - here cross_entropy 
loss = -tf.reduce_sum(Y_*tf.log(Y+ 1e-10))
train_step = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(Y,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(numIters):
    batchX, batchY = mnist.train.next_batch(batchSize)
    trainData = {X:batchX, Y_:batchY}
    # train
    sess.run(train_step,feed_dict=trainData)
    
    if i%displayStep == 0:
        acc, cost = sess.run([accuracy, loss], feed_dict=trainData)
        print(str(i) + ": accuracy:" + str(acc) + " loss: " + str(cost))

testData = {X:mnist.test.images, Y_:mnist.test.labels} 
acc, cost = sess.run([accuracy,loss], feed_dict=testData)
print("*****Test accuracy:" + str(acc) + " loss: " + str(cost) + "*****\n")






