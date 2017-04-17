#단일 퍼셉트론으로만 Xor 문제를 해결해보기
import tensorflow as tf
import numpy as np

tf.set_random_seed(777)
# 샘플데이터
x_data = np.array([[0, 0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0],[1],[1],[0]], dtype = np.float32)
#플레이스 홀더
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
#찾아야할 변수를 입력
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
#가설 설정
hypothesis = tf.sigmoid(tf.matmul(X,W)+b)
#reduce_mean == demension에 따른 평균값을 구함 default는 0
#전체 output에 대한 오차 평균을 구하는 것임
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))
#GradiendDescent 방법으로 최적화 한다.
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
####### 최종 평가 ######
#0.5 보다 크면 1이라고 생각
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
#전체 오차 평균을 구함.
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
#Session = A class for running TensorFlow operations.
with tf.Session() as sess:
#변수에 대한 랜덤 초기화
    sess.run(tf.global_variables_initializer())
#학습횟수 결정
    for step in range(10001):
        sess.run(train, feed_dict={X:x_data, Y: y_data})
        #if step % 100 ==0:
            #print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print ("\nHypothesis: ", h, "\nCorrect: ",c , "\nAccuracy: ", a)
