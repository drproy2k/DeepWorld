########################################
#
# Word2Vec for Hana chatbot
#
# 2018.1.11.Thr Billy Inseong Hwang
#
########################################


import codecs
import re


from konlpy.tag import Mecab

root_dir = './data'
tokenized_file = root_dir+'/'+"tokenized_sent_w2v.txt"

"""
# 학습할 데이터 준비
writeFp = open(tokenized_file, "w", encoding="utf-8")

# 형태소 분석 --- (※2)
#mecab = Mecab('/home/billy/mecab-ko-dic-2.0.1-20150920')
mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')

fname = root_dir+'/'+'wiki.txt'
print('Tokenizing : ', fname)
readFp = codecs.open(fname, "r", encoding="utf-8")
i = 0
# 텍스트를 한 줄씩 처리하기
#fp_test = open('input_sent_log', "w", encoding="utf-8")
while True:
    line_org = readFp.readline()
    if not line_org: break
    line = re.sub('[-=.#/?:$}]', '', line_org)
    #if i > 1640000 :
        #fp_test.write(line+' '+str(i)+' ')
    if i % 20000 == 0:
        print("current - " + str(i))
    i += 1
    # 형태소 분석
    if i < 1650769 or i > 1650770: #원인모를 문서내 이슈 때문에 건너뛴다
        malist = mecab.pos(line)
    # 필요한 어구만 대상으로 하기
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외
        if (word[1] in ['NNG', 'MAG', 'NNP', 'NNG+NNG', 'SL', 'VV+ETM', 'NP', 'VA+ETM', 'VV+ETM', 'XSN', 'MM', 'VV', 'VV+EC', 'NR',
                            'VV+EP', 'VA', 'NNBC',  'SN'] and len(word[0]) >= 1):  # Check only Noun
            tmp_string = word[0]
            tmp_string = tmp_string.replace(" ", "")
            writeFp.write(tmp_string + " ")
readFp.close()
print('Finished..', fname)
writeFp.close()



# Build Word2Vec model
print('Build model')
from gensim.models import word2vec
data = word2vec.Text8Corpus(tokenized_file)
model = word2vec.Word2Vec(data, size=100)  #default CBOW
model.save("wiki_w2v.model")
print("ok")
"""


# Load model
from gensim.models import word2vec
model = word2vec.Word2Vec.load("wiki_w2v.model")


print('Enter a word(or p-word n-word p-word ): ')
while True :
    text_in = input()
    words = text_in.split()
    if text_in == 'q' : exit()
    if len(words)<2 :
        print(model.wv[words])
        print(model.most_similar(positive=[words[0]])[0:20])
    elif len(words)<3 :
        print(model.most_similar(positive=[words[0], words[1]])[0:20])
    else:
        print(model.most_similar(positive=[words[0], words[2]], negative=[words[1]])[0:20])


"""
from gensim.models import word2vec
model = word2vec.Word2Vec.load("wikinhanaqa.model")

print(model['안녕'])
print(model['AI'])
result1 = model.most_similar(positive='누구', negative='', topn=10)
print(result1)
"""

"""
X = model[model.wv.index2word]
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

font_name = matplotlib.font_manager.FontProperties(
                fname="/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 한글 폰트 위치를 넣어주세요
            ).get_name()
vocab = model.wv.index2word
matplotlib.rc('font', family=font_name)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X) #t-분포 확률적 임베딩(t-SNE)은 데이터의 차원 축소에 사용
df = pd.concat([pd.DataFrame(X_tsne),
                pd.Series(vocab)],
               axis=1)

df.columns = ['x', 'y', 'word']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
print(df)
ax.scatter(df['x'], df['y'])
ax.set_xlim(df['x'].max(), df['x'].min())
ax.set_ylim(df['y'].max(), df['y'].min())
for i, txt in enumerate(df['word']):
    ax.annotate(txt, (df['x'].iloc[i], df['y'].iloc[i]))
plt.show()
"""
