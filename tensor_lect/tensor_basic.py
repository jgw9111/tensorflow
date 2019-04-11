import  tensorflow as tf

var = tf.Variable([5.0],dtype=tf.float32) #기본형 float 정수사이 무한한 값? ->값과 값 사이를 계속 연결해야 하기 때문 #변수 / 생성자에 의해 만들어지는 객체
con = tf.constant([10.0],dtype=tf.float32) #상수
session = tf.Session()
init = tf.global_variables_initializer() #초기화 반드시(변수들만 초기화함)

session.run(init)

print(session.run(var*con))
print('>>>>>>>>>>>')
session.run(var.assign([10.0])) #연산은 세션이 run 할 때 이루어짐 / 값의 변화는 노드에서 이루어진다
print(session.run(var))

p1 = tf.placeholder(dtype=tf.float32) #tensorflow에 input 공간을 주는 placeholder
p2 = tf.placeholder(dtype=tf.float32)

t1= p1*3
t2= p1*p2
#연산은 안 되고, 식만 줌

session =tf.Session()
print(session.run(t2,feed_dict={p1:4.0,p2:[2.0,5.0]})) #p2 행렬연산

print(session.run(t1,{p1:[4.0]})) #feed_dict 생략