# Seq2Seq Chatbot with Retrieval method

# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import sys

print (sys.version)
print (tf.__version__) #1.1이상 가능

# 질문에 따른 답변 정의
train_data = [
    ['안녕', '방가 방가'],
    ['뭐하니?', '일하고 있쥐'],
    ['집에 안가?', '좀만 더 하고 가려구..'],
    ['주말에 뭐할꺼니?', '토욜날 운동 가려구 해, 너는?'],
    ['그냥 집에 있으려구.. 담주 저녁에 소주 한잔 어때?', '좋지 어디서?'],
    ['강남역에 새로 생긴 맛집이 있드라 거기서 보자', '좋았어 몇시에?'],
    ['7시쯤', 'OK'],
    ['그럼 어여 정리하고 들어가고.. 담주에 보자~', ' 응 그래'],
    ['수고~', '주말잘보내~'],
]

from konlpy.tag import Mecab

mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
train_data2 = list(map(lambda x: mecab.morphs(' '.join(x)), train_data))
print(train_data2)
import itertools

char_array = list(itertools.chain.from_iterable(train_data2))

char_array = ['P', '[', ']'] + list(set(char_array))  # Padding값을 0으로 주어 weight제외
print(char_array)

train_data2 = []

for qna_data in train_data:
    train_data2 = train_data2 + list(map(lambda x: mecab.morphs(x), qna_data))

print(train_data2)

# max_input_text = 7
# max_output_text = 7
max_input_text = max(len(s) for s in train_data2)
max_output_text = max(len(s) for s in train_data2)
print(max_input_text)
print(max_output_text)

# enumerate 방법 사용 index : value 정렬
num_dic = {n: i for i, n in enumerate(char_array)}

dic_len = len(num_dic)

print ("# Char List : " + str(num_dic))
print ("# Char Size : " + str(dic_len))

def make_train_data(train_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in train_data:
        # 인코더 셀의 입력값. 입력단어의 글자들을 한글자씩 떼어 배열로 만든다.
        seq_0 = mecab.morphs(seq[0])
        seq_1 = mecab.morphs(seq[1])
        input = [num_dic[n] for n in seq_0 + ['P'] * (max_input_text - len(seq_0)) ]# P는 Padding 값
        # 디코더 셀의 입력값. 시작을 나타내는 [ 심볼을 맨 앞에 붙여준다. (Seq의 구분)
        output = [num_dic[n] for n in (['['] + seq_1 + ['P'] * (max_output_text - len(seq_1)))]
        # 학습을 위해 비교할 디코더 셀의 출력값. 끝나는 것을 알려주기 위해 마지막에 ] 를 붙인다.
        target = [num_dic[n] for n in (seq_1 + ['P'] * (max_output_text - len(seq_1)) + [']'] )]
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)
    return input_batch, output_batch, target_batch

file_path = './model'
def model_file(file_path, flag):
    if(flag):
        import os
        saver = tf.train.Saver(tf.global_variables())

        if(not os.path.exists(file_path)):
            os.makedirs(file_path)
        saver.save(sess, ''.join(file_path + "/.model"))
        print("Model Saved")
    else:
        import shutil
        try:
            shutil.rmtree(file_path)
            print("Model Deleted")
        except OSError as e:
            if e.errno == 2:
                # 파일이나 디렉토리가 없음!
                print ('No such file or directory to remove')
                pass
            else:
                raise

# 옵션 설정
learning_rate = 0.01
n_hidden = 256
total_epoch = 100
# one hot 위한 사이즈
n_class = n_input = dic_len

# 그래프 초기화
tf.reset_default_graph()
# Seq2Seq 모델은 인코더의 입력과 디코더의 입력의 형식이 같다.
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

# 인코더
with tf.variable_scope("encoder"):
    enc_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input,
                                            dtype=tf.float32)

# 디코더
with tf.variable_scope("decoder"):
    dec_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input,
                                            initial_state=enc_states,
                                            dtype=tf.float32)

model = tf.layers.dense(outputs, n_class, activation=None)

#onehot로 sparse사용
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
input_batch, output_batch, target_batch = make_train_data(train_data)

import matplotlib.pyplot as plt

def display_train():
    plot_X = []
    plot_Y = []
    for epoch in range(total_epoch):
        _, loss = sess.run([optimizer, cost],
                           feed_dict={enc_input: input_batch,
                                      dec_input: output_batch,
                                      targets: target_batch})
        plot_X.append(epoch + 1)
        plot_Y.append(loss)
    # Graphic display
    plt.plot(plot_X, plot_Y, label='cost')
    plt.show()

display_train()

# 최적화가 끝난 뒤, 변수를 저장합니다.
model_file(file_path, True)


# 단어를 입력받아 번역 단어를 예측하고 디코딩하는 함수
def predict(word):
    input_batch, output_batch, target_batch = make_train_data([word])
    # 결과가 [batch size, time step, input] 으로 나오기 때문에,
    # 2번째 차원인 input 차원을 argmax 로 취해 가장 확률이 높은 글자를 예측 값으로 만든다.
    # http://pythonkim.tistory.com/73
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction,
                      feed_dict={enc_input: input_batch,
                                 dec_input: output_batch,
                                 targets: target_batch})
    # 결과 값인 숫자의 인덱스에 해당하는 글자를 가져와 글자 배열을 만든다.
    decoded = [char_array[i] for i in result[0]]

    if 'P' in decoded:
        end = decoded.index('P')
        decoded = decoded[:end]
    elif ']' in decoded:
        end = decoded.index(']')
        decoded = decoded[:end]
    return decoded


print("Q: 안녕")
print("A: " + ' '.join(predict(['안녕', ''])))
print("Q: 뭐하니?")
print("A: " + ' '.join(predict(['뭐하니?', ''])))
print("Q: 집에안가?")
print("A: " + ' '.join(predict(['집에안가?', ''])))
print("Q: 주말에 뭐할꺼니?")
print("A: " + ' '.join(predict(['주말에 뭐할꺼니?', ''])))
print("Q: 그냥 집에 있으려구.. 담주 저녁에 소주 한잔 어때?")
print("A: " + ' '.join(predict(['그냥 집에 있으려구.. 담주 저녁에 소주 한잔 어때?', ''])))
print("Q: 강남역에 새로생긴 맛집이 있드라 거기서 보자")
print("A: " + ' '.join(predict(['강남역에 새로생긴 맛집이 있드라 거기서 보자', ''])))
print("Q: 7시쯤")
print("A: " + ' '.join(predict(['7시쯤', ''])))
print("Q: 그럼 어여 정리하고 들어가고.. 담주에 보자~")
print("A: " + ' '.join(predict(['그럼 어여 정리하고 들어가고.. 담주에 보자~', ''])))
print("Q: 수고~")
print("A: " + ' '.join(predict(['수고~', ''])))
model_file(file_path, False)

