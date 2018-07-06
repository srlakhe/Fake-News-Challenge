
# coding: utf-8

# In[30]:


from fncbaseline.utils import dataset, generate_test_splits, score
def evaluate_answer(predicted,true):
    inv_category_dict = {0:'unrelated', 1: 'agree', 2: 'disagree', 3: 'discuss'}
    t = np.argmax(true,axis = 1)
    ground = list()
    pred = list()
    #print(predicted)
    for i in predicted:
        pred.append(inv_category_dict[i[0][0]])
    for i in t:
        ground.append(inv_category_dict[i])
    score.report_score(ground, pred)


# In[1]:


def create_labels(data):

#     Usage
#     y_train = create_labels(train_dataset)
#     y_test = create_labels(test_dataset)

    from keras.utils.np_utils import to_categorical
    category_dict = {'unrelated': 0 , 'agree':1, 'disagree':2, 'discuss':3}
    y = list()
    NUM_CLASSES = 4
    for stance in data.stances:
        y.append(category_dict[stance['Stance']])

    y_cat = np.zeros((len(y),NUM_CLASSES))
    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    return y_cat


# In[2]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

X_train = np.squeeze(np.load('./word2vec_headline_body_train.npy'))
X_test = np.squeeze(np.load('./word2vec_headline_body_test.npy'))

X_train_head = X_train[0,:,:300]
X_train_body = X_train[1,:,:300]
X_test_head  = X_test[0,:,:300]
X_test_body  = X_test[1,:,:300]

y_train = np.load('./y_train.npy')
y_test  = np.load('./y_test.npy')

# Define paramaters for the model
learning_rate = 0.01
batch_size = 1
n_epochs = 30
feature_dimension = int(X_train_head.shape[1])


# In[3]:



# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
#mnist = input_data.read_data_sets('/data/mnist', one_hot=True) 

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# each lable is one hot vector.

X_head = tf.placeholder(tf.float32, [batch_size, feature_dimension], name='X_head_placeholder') 
X_body = tf.placeholder(tf.float32, [batch_size, feature_dimension], name='X_body_placeholder') 

Y = tf.placeholder(tf.int32, [batch_size, 4], name='Y_placeholder')

# Step 3: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y

w_affine = tf.Variable(tf.random_normal(shape=[feature_dimension,feature_dimension], stddev=0.01), name='weights_affine')
w_head = tf.Variable(tf.random_normal(shape=[feature_dimension,1], stddev=0.01), name='weights_heads')
w_body = tf.Variable(tf.random_normal(shape=[feature_dimension,1], stddev=0.01), name='weights_body')
w_logit = tf.Variable(tf.random_normal(shape=[1,4], stddev=0.01), name='weights_logit')
b_logit = tf.Variable(tf.zeros([1, 1]), name="bias_logit")
b = tf.Variable(tf.zeros([1, 1]), name="bias")

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
biaffine = tf.matmul(tf.matmul(X_head, w_affine),X_body,transpose_b=True) + tf.matmul(X_head,w_head)+tf.matmul(X_body,w_body) + b 
logits = tf.matmul(biaffine, w_logit) + b_logit
# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y, name='loss')
loss = tf.reduce_mean(entropy) # computes the mean over all the examples in the batch

# Step 6: define training op
# using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

preds = tf.nn.softmax(logits)
model_pred = tf.argmax(preds, 1)
correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(


# In[29]:



sess = tf.Session()
    # to visualize using TensorBoard
writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)


sess.run(tf.global_variables_initializer())	
n_batches = int(X_train_head.shape[0]/batch_size)
for i in range(1000): # train the model n_epochs times
    start_time = time.time()
    total_loss = 0
    j = 0
    print("HI")
    for _ in range(X_train_head.shape[0]):
        X_batch_head, X_batch_body, Y_batch = np.reshape(X_train_head[j,:],(1,300)),np.reshape(X_train_body[j,:],(1,300)),np.reshape(y_train[j,:],(1,4))
        _, loss_batch = sess.run([optimizer, loss], feed_dict={X_head: X_batch_head, X_body:X_batch_body, Y:Y_batch}) 
        total_loss += loss_batch
        j+=1

    print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

    print('Total time: {0} seconds'.format(time.time() - start_time))
    if (i+1)%20==0:
        total_correct_preds = 0
        y_pred_test = list()
        for j in range(X_test_head.shape[0]):
            X_batch_head, X_batch_body, Y_batch = np.reshape(X_test_head[j,:],(1,300)),np.reshape(X_test_body[j,:],(1,300)),np.reshape(y_test[j,:],(1,4))
            model_pred_x = sess.run([model_pred], feed_dict={X_head: X_batch_head, X_body:X_batch_body, Y:Y_batch}) 
            y_pred_test.append(model_pred_x)
        
        evaluate_answer(y_pred_test,y_test)

print('Optimization Finished!') # should be around 0.35 after 25 epochs

# test the model

writer.close()
sess.close()

