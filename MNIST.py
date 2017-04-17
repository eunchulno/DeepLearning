#https://tensorflowkorea.gitbooks.io/tensorflow-kr/g3doc/tutorials/mnist/pros/

#자동으로 MNIST 데이터 셋을 불러온다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
#계산 그래프를 구성하는 작업과 그 그래프를 실행하는 작업을 나누어 주기 위해 InteractiveSession을 사용한다.
import tensorflow as tf
sess = tf.InteractiveSession()

###### 소프트 맥스 회기모델이다.######
# 입출력 노드를 정의한다.
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# 변수를 정의한다. 여기서는 0으로 초기값을 설정한다.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# 위의 설정된 값으로 변수를 초기화 한다.
sess.run(tf.global_variables_initializer())

# 소프트 맥스모델을 활성함수로 한다.
y = tf.nn.softmax(tf.matmul(x,W) + b)

# 비용 함수는 실제 클래스와 모델의 예측결과간 크로스 엔트로피 함수이다.
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

##### 학습시키기 #####
# gradient descent 알고리즘을 사용하여 크로스 엔트로피를 최소화 시킨다.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 위의 train_step을 실행하면, 경사 하강법을 통해 각각의 매개 변수를 변화시키게 된다.
for i in range(1000):
     batch = mnist.train.next_batch(50)
     train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 모델을 평가하는 모델 정의
# tf.argmax(y,1)은 모델이 입력을 받고 가장 그럴듯하다고 생각한 레이블이고, tf.argmax(y_,1)은 실제레이블이다.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# 결과를 평균내어 정확도를 구한다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 정확도를 얻는다.
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
