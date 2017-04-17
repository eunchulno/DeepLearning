#https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/tutorials/mnist/pros/

#자동으로 MNIST 데이터 셋을 불러온다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#계산 그래프를 구성하는 작업과 그 그래프를 실행하는 작업을 나누어 주기 위해 InteractiveSession을 사용한다.
import tensorflow as tf
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

##### Convolution Nureul Network #####
# 합성곱 신경망 모델을 구성하기 위해서는 많은 수의 가중치와 편향을 사용하게 됩니다. 대칭성을 깨뜨리고 기울기(gradient)가 0이 되는 것을 방지하기 위해, 가중치에 약간의 잡음을 주어 초기화합니다. 또한, 모델에 ReLU 뉴런이 포함되므로, "죽은 뉴런"을 방지하기 위해 편향을 작은 양수(0.1)로 초기화합니다. 매번 모델을 만들 때마다 반복하는 대신, 아래 코드와 같이 이러한 일을 해 주는 함수 두 개를 생성합니다.
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#TensorFlow는 합성곱과 풀링 계층(layer)을 유연하게 다룰 수 있도록 해 줍니다. 경계의 패딩(padding)과스트라이드(stride)에 대해 다양한 선택을 할 수 있습니다. 이번 예시에서는 스트라이드를 1로, 출력크기가 입력과 같게 되도록 0으로 패딩하도록 설정합니다. 풀링은 2x2 크기의 맥스 풀링을 적용합니다.마찬가지로 코드를 간단히 하기 위해 합성곱과 풀링을 위한 함수를 아래 코드와 같이 생성합니다.

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding='SAME')
#첫번째 계층을 만들기 위해 필터를 만든다.
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 앞의 x를 CNN에 적합하게 reshape한다.
x_image = tf.reshape(x, [-1,28,28,1])

# 첫번째 합성 계층을 구한다.
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
# pool과정을 통해 14x14 매트릭스로 변환된다.
h_pool1 = max_pool_2x2(h_conv1)


# 두번째 계층을 구성한다.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 64개의 깊이를 갖는 7x7 매트릭스로 변환된다.
h_pool2 = max_pool_2x2(h_conv2)

##### 직렬화 하여 NN을 실행한다. #####
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 오버피팅이 되는 것을 막기 위해 드롭아웃을 적용한다.
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 두번째 레이어
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 이번에는 경사 하강법 알고리즘 대신 더 복잡한 ADAM 최적화 알고리즘을 사용합니다. 또한, 드롭아웃 확률을 설정하는 추가변수인 keep_prob을 feed_dict 인수를 통해 전달합니다. 아래의 코드는 훈련 과정에서 100회 반복 시마다 로그를 작성합니다.
# 훈련 모델
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 평가 모델
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
# 훈련
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %0.04f"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
# 평가
print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
