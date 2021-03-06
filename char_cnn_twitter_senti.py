####################################################################
# char_cnn_twitter_senti.py
# Sentimental Analyzer with char-CNN
#

import requests
import tensorflow as tf
import numpy as np
import os
#from matplotlib.image import imread, imsave
#import matplotlib.pyplot as plt
import pandas as pd
from konlpy.tag import Twitter
from gensim.models import word2vec
print("load done")

vector_size = 50        # word2vec 크기
encode_length = 500     # 영화평 댓글 내의 단어 최대 개수.. 일단 100으로 하고, 나중에 데이터 뽑으면 최대값 구해서 수정하자
label_size = 2          # 긍정과 부정
embed_type = "w2v"      # onehot or w2v

# Choose single test
filter_type = "single"

filter_number = 32
filter_size = 2

# Choose multi test
#filter_type = "multi"

filter_sizes = [2,3,4,2,3,4,2,3,4]
#filter_sizes = [2,3,4,2,3,4]
num_filters = len(filter_sizes)

def load_data_list(which):
    max_sent_length = 0
    data = {'encode': [], 'decode': []}
    for i in range(2):
        if i == 0:  # positive case
            basepass = 'C:/Users/Genie/Downloads/aclImdb/' + which + '/pos'
            label = '1'
        else:
            basepass = 'C:/Users/Genie/Downloads/aclImdb/' + which + '/neg'
            label = '0'
        filelist = os.listdir(basepass)
        for file in filelist:
            file = basepass + '/' + file
            f = open(file, 'rt', encoding='UTF8')
            temp = f.read()
            temp = temp.replace('"', "'")
            num_of_word = temp.split()
            if max_sent_length < len(num_of_word):
                max_sent_length = len(num_of_word)
            # if len(temp) == 2470:
            #     print(temp)
            #     input()
            data['encode'].append(temp)
            data['decode'].append(label)
            # print(data)
            # input()
    return data, max_sent_length

# train_data_list =  {
#                 'encode' : ['판교에 오늘 피자 주문해줘','오늘 날짜에 호텔 예약 해줄레','모래 날짜에 판교 여행 정보 알려줘'],
#                 'decode' : ['0','1','2']
#              }
data, max_sent_length = load_data_list('train')     # 크기가 너무 커서 메모리 부족 에러 뜬다. 나눠야한다
data_size = len(data['encode'])               # 전체 개수가 얼마인지 보자

import random
def random_pop(pool):
    number = random.randint(0, len(pool)-1)
    return pool.pop(number)

pool = list(range(data_size))
def rand_sel_data(data, num_of_data):
    global pool
    ret_data = {'encode': [], 'decode': []}
    for i in range(num_of_data):
        val = random_pop(pool)
        ret_data['encode'].append(data['encode'][val])
        ret_data['decode'].append(data['decode'][val])
    return ret_data

train_data_list = rand_sel_data(data, 1000)     # 입력 숫자만큰 data로부터 데이터를 랜덤위치에서 빼온다. 중복 선택 안된다
train_data_list.get('encode')
# print(train_data_list)
# input()

# encode_length = max_sent_length       # 메모리가 부족해져서...

def train_vector_model(str_buf):
    #mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    twitter = Twitter()
    str_buf = train_data_list['encode']
    pos1 = twitter.pos(''.join(str_buf))
    pos2 = ' '.join(list(map(lambda x : '\n' if x[1] in ['SF'] else x[0], pos1))).split('\n')
    morphs = list(map(lambda x : twitter.morphs(x) , pos2))
    print(str_buf)
    model = word2vec.Word2Vec(size=vector_size, window=2, min_count=1)
    model.build_vocab(morphs)
    model.train(morphs, total_examples=model.corpus_count, epochs=model.iter)
    return model
model = train_vector_model(train_data_list)
print(model)

# Load Train Data
def load_csv(data_path):
    df_csv_read = pd.DataFrame(data_path)
    return df_csv_read

# Embedding using W2V
def embed(data):
    #mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    twitter = Twitter()
    inputs = []
    labels = []
    for encode_raw in data['encode']:
        encode_raw = twitter.morphs(encode_raw) #워드리스트를 리턴
        encode_raw = list(map(lambda x: encode_raw[x] if x < len(encode_raw) else '#', range(encode_length))) #입력문장들의 단어수를 일정하게만들기
        if (embed_type == 'onehot'):
            bucket = np.zeros(vector_size, dtype=float).copy()
            input = np.array(list(map(
                lambda x: onehot_vectorize(bucket, x) if x in model.wv.index2word else np.zeros(vector_size,
                                                                                                dtype=float),
                encode_raw)))
        else:
            input = np.array(list(
                map(lambda x: model[x] if x in model.wv.index2word else np.zeros(vector_size, dtype=float),
                    encode_raw))) #벡터로변환

        ######################################
        #print(data['encode'])
        #print(encode_raw)
        #input()
        #exit()

        inputs.append(input.flatten())

    for decode_raw in data['decode']:
        label = np.zeros(label_size, dtype=float)
        np.put(label, decode_raw, 1)
        labels.append(label)
    return inputs, labels


def onehot_vectorize(bucket, x):
    np.put(bucket, model.wv.index2word.index(x), 1)
    return bucket

#Embedding using W2V for Prediction Step
def inference_embed(data) :
    #mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
    twitter = Twitter()
    encode_raw = twitter.morphs(data)
    encode_raw = list(map(lambda x : encode_raw[x] if x < len(encode_raw) else '#', range(encode_length)))
    if(embed_type == 'onehot') :
        bucket = np.zeros(vector_size, dtype=float).copy()
        input = np.array(list(map(lambda x : onehot_vectorize(bucket, x) if x in model.wv.index2word else np.zeros(vector_size,dtype=float) , encode_raw)))
    else :
        input = np.array(list(map(lambda x : model[x] if x in model.wv.index2word else np.zeros(vector_size,dtype=float) , encode_raw)))
    return input

#Get the train and test data for feeding on CNN
def get_test_data():
    train_data, train_label = embed(load_csv(train_data_list))
    test_data, test_label = embed(load_csv(train_data_list))
    return train_label, test_label, train_data, test_data

#Create graph with single filter size
def create_s_graph(train=True):
    # placeholder is used for feeding data.
    x = tf.placeholder("float", shape=[None, encode_length * vector_size], name='x')
    y_target = tf.placeholder("float", shape=[None, label_size], name='y_target')

    # reshape input data
    x_image = tf.reshape(x, [-1, encode_length, vector_size, 1], name="x_image")

    # Build a convolutional layer and maxpooling with random initialization
    W_conv1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, 1, filter_number], stddev=0.1),
                          name="W_conv1")  # W is [row, col, channel, feature]
    b_conv1 = tf.Variable(tf.zeros([filter_number]), name="b_conv1")
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1, name="h_conv1")
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="h_pool1")

    # Build a fully connected layer
    h_pool2_flat = tf.reshape(h_pool1, [-1, int((encode_length / 2) * (vector_size / 2)) * filter_number],
                              name="h_pool2_flat")
    W_fc1 = tf.Variable(
        tf.truncated_normal([int((encode_length / 2) * (vector_size / 2)) * filter_number, 256], stddev=0.1),
        name='W_fc1')
    b_fc1 = tf.Variable(tf.zeros([256]), name='b_fc1')
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1, name="h_fc1")

    keep_prob = 1.0
    if (train):
        # Dropout Layer
        keep_prob = tf.placeholder("float", name="keep_prob")
        h_fc1 = tf.nn.dropout(h_fc1, keep_prob, name="h_fc1_drop")

    # Build a fully connected layer with softmax
    W_fc2 = tf.Variable(tf.truncated_normal([256, label_size], stddev=0.1), name='W_fc2')
    b_fc2 = tf.Variable(tf.zeros([label_size]), name='b_fc2')
    # y=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2, name="y")
    y = tf.matmul(h_fc1, W_fc2) + b_fc2

    # define the Loss function
    # cross_entropy = -tf.reduce_sum(y_target*tf.log(y), name = 'cross_entropy')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_target))

    # define optimization algorithm
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_target, 1))
    # correct_prediction is list of boolean which is the result of comparing(model prediction , data)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # tf.cast() : changes true -> 1 / false -> 0
    # tf.reduce_mean() : calculate the mean

    return accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1


print("define cnn graph func")


#Create graph with multi filter size
def create_m_graph(train=True):
    # placeholder is used for feeding data.
    x = tf.placeholder("float", shape=[None, encode_length * vector_size], name='x')
    y_target = tf.placeholder("float", shape=[None, label_size], name='y_target')

    # reshape input data
    x_image = tf.reshape(x, [-1, encode_length, vector_size, 1], name="x_image")
    # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, vector_size, 1, num_filters]
            W_conv1 = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

            conv = tf.nn.conv2d(
                x_image,
                W_conv1,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")

            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b_conv1), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, encode_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    keep_prob = 1.0
    if (train):
        keep_prob = tf.placeholder("float", name="keep_prob")
        h_pool_flat = tf.nn.dropout(h_pool_flat, keep_prob)

    # Final (unnormalized) scores and predictions
    W_fc1 = tf.get_variable(
        "W_fc1",
        shape=[num_filters_total, label_size],
        initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[label_size]), name="b")
    l2_loss += tf.nn.l2_loss(W_fc1)
    l2_loss += tf.nn.l2_loss(b_fc1)
    y = tf.nn.xw_plus_b(h_pool_flat, W_fc1, b_fc1, name="scores")
    predictions = tf.argmax(y, 1, name="predictions")

    # CalculateMean cross-entropy loss
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_target)
    cross_entropy = tf.reduce_mean(losses)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Accuracy
    correct_predictions = tf.equal(predictions, tf.argmax(y_target, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    return accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1


print("define cnn graph func")


#show the weight matrix
def show_layer(weight_list) :
    if(filter_type == 'multi') :
        show = np.array(weight_list).reshape(num_filters, filter_sizes[np.argmax(filter_sizes)], vector_size)
        for i, matrix in enumerate(show) :
            fig = plt.figure()
            plt.imshow(matrix)
        plt.show()
    else :
        show = np.array(weight_list).reshape(32, 2, 2)
        for i, matrix in enumerate(show) :
            fig = plt.figure()
            plt.imshow(matrix)
        plt.show()

summary = tf.summary.merge_all()


#Training
import time
def run():
    try:
        # get Data
        labels_train, labels_test, data_filter_train, data_filter_test = get_test_data()
        # reset Graph
        tf.reset_default_graph()

        # Create Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

        # create graph
        if (filter_type == 'single'):
            accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1 = create_s_graph(train=True)
        else:
            accuracy, x, y_target, keep_prob, train_step, y, cross_entropy, W_conv1 = create_m_graph(train=True)

        # set saver
        saver = tf.train.Saver(tf.global_variables())
        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # training the MLP
        for i in range(200):

            # Test code #################################################
            #print(data_filter_train)
            #print(y_target)
            #print(labels_train)
            #print(model['판교'])
            #input()
            #exit()
            #############################################################

            sess.run(train_step, feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 0.5})

            if i % 10 == 0:
                train_accuracy = sess.run(accuracy,
                                          feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 1})
                print("step %d, training accuracy: %.3f" % (i, train_accuracy))

        # 텐서보드 그래프 그리기
        summary_writer = tf.summary.FileWriter('log_dir', graph=sess.graph)

        # for given x, y_target data set
        print("test accuracy: %g" % sess.run(accuracy,
                                             feed_dict={x: data_filter_test, y_target: labels_test, keep_prob: 1}))

        # show weight matrix as image
        weight_vectors = sess.run(W_conv1, feed_dict={x: data_filter_train, y_target: labels_train, keep_prob: 1.0})
        # show_layer(weight_vectors)

        # Save Model
        path = './model/'
        if not os.path.exists(path):
            os.makedirs(path)
            print("path created")
        saver.save(sess, path)
        print("model saved")
    except Exception as e:
        raise Exception("error on training: {0}".format(e))
    finally:
        sess.close()


# run stuff
run()


#Testing
def predict(test_data):
    try:
        # reset Graph
        tf.reset_default_graph()
        # Create Session
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        # create graph
        if (filter_type == 'single'):
            _, x, _, _, _, y, _, _ = create_s_graph(train=False)
        else:
            _, x, _, _, _, y, _, _ = create_m_graph(train=False)

        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # set saver
        saver = tf.train.Saver()

        # Restore Model
        path = './model/'
        if os.path.exists(path):
            saver.restore(sess, path)
            print("model restored")

        # training the MLP
        # print("input data : {0}".format(test_data))
        y = sess.run([y], feed_dict={x: np.array([test_data])})
        print("result : {0}".format(y))
        print("result : {0}".format(np.argmax(y)))

    except Exception as e:
        raise Exception("error on training: {0}".format(e))
    finally:
        sess.close()


print("words in dict : {0}".format(model.wv.index2word))

# positive test
predict(np.array(inference_embed("I went and saw this movie last night after being coaxed to by a few friends of mine. I'll admit that I was reluctant to see it because from what I knew of Ashton Kutcher he was only able to do comedy. I was wrong. Kutcher played the character of Jake Fischer very well, and Kevin Costner played Ben Randall with such professionalism. The sign of a good movie is that it can toy with our emotions. This one did exactly that. The entire theater (which was sold out) was overcome by laughter during the first half of the movie, and were moved to tears during the second half. While exiting the theater I not only saw many women in tears, but many full grown men as well, trying desperately not to let anyone see them crying. This movie was great, and I suggest that you go see it before you judge.")).flatten())
# negative test
predict(np.array(inference_embed("Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in.")).flatten())

