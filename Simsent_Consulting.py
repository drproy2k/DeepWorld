#################################################################
# File Name : SimSent_Consulting.py
#

import csv
import numpy as np
import os, glob, json

root_dir = './TestData/Consulting/DearGen'
dic_file_kr = "word-dic_DearGen_kr.json"
dic_file_en = "word-dic_DearGen_en.json"
i=0
label = i.__str__()
#fname = root_dir + '/' + 'label' + label + '.txt'

"""
def isKoreanText(text): #Twitter
    # 단어의 기본형 사용
    words = filter(text)
    flag = False
    for word in words:
        if not word[1] in ["Alpha", "Number"]: # 영어와 숫자로 된 문장만 영문 문서. 나머지는 한글문서로 본다.
            flag = True
    return flag
"""

def isKoreanText(text): # Mecab
    # 단어의 기본형 사용
    words = filter(text)
    flag = False
    for word in words:
        if not word[1] in ["SL", "SN"]: # 영어와 숫자로 된 문장만 영문 문서. 나머지는 한글문서로 본다.
            flag = True
    return flag

def getCsv(filename):
    global root_dir
    csvList_kr = []
    csvList_en = []
    fname = root_dir + '/' + filename
    f = open(fname, 'r')
    csvReader = csv.reader(f)
    line_counter = 0
    for line in csvReader:
        csvLine = []
        for idx in range(3):
            csvLine.append(line[idx+5])  #csv파일 내 컬럼에서 제목, 요약, 대표청구항만 선택 저장
        if line_counter != 0 :  #csv 파일 첫줄은 항목명칭 라인이라서 삭제
            text = csvLine[0] + ' ' + csvLine[1] + ' ' + csvLine[2] ################################################

            if str(csvLine[2]).strip() != '':
                text = csvLine[2]  # 대표청구항
            elif str(csvLine[1]).strip() != '':
                text = csvLine[1]  # 요약
                #print('요약: ', len(csvLine), text)
            elif str(csvLine[0]).strip() != '':
                text = csvLine[0]  # 제목
                #print('제목: ', len(csvLine), text)
            else:
                print('No data in this line', line_counter)
                input()

            if isKoreanText(text) == True :
                csvList_kr.append(text)
            else:
                csvList_en.append(text)
        line_counter += 1
    f.close()
    return csvList_kr, csvList_en


from xlrd import open_workbook
duplicateListKr = []
duplicateListEn = []
# 요약우선
def getExcel_sum(filename):
    global duplicateListKr
    global duplicateListEn
    print(filename)
    global root_dir
    xlsList_kr = []
    xlsList_en = []
    label_kr = []
    label_en = []
    fname = root_dir + '/' + filename
    wb = open_workbook(fname)
    FORMAT = ['발명의 명칭', '요약', '대표청구항', 'Label', 'SubLabel', '출원번호'] #5가지 컬럼만 선택 ##################################################
    xlsReader = []
    for s in wb.sheets():
        headerRow = s.row(0)
        columnIndex = [x for y in FORMAT for x in range(len(headerRow)) if y == headerRow[x].value]
        for row in range(1, s.nrows): #xls 파일 첫줄은 항목명칭 라인이라서 삭제
            currentRow = s.row(row)
            currentRowValues = [currentRow[x].value for x in columnIndex]
            xlsReader.append(currentRowValues)
        break   # 첫번재 시트만 처리한다.
    idx = 0
    file_num_list = []
    for line in xlsReader:
        if line[5] not in file_num_list:    # 중복 특허 제거
            file_num_list.append(line[5])
            # text = line[0] + ' ' + line[1] + ' ' + line[2] #########################################################
            if str(line[1]).strip() != '':
                text = line[1]  # 요약
                if len(text) < 1000 - 900 * int(isKoreanText(text)):
                    text = line[0] + ' ' + text + ' ' + line[2]
                    # text = text + ' ' + line[2]
            elif str(line[2]).strip() != '':
                text = line[2]  # 대표청구항
                # print('요약: ', len(line), text)
            elif str(line[0]).strip() != '':
                text = line[0]  # 제목
                # print('제목: ', len(line),text)
            else:
                print('No data in this line', idx)
                input()
            # For test
            # if idx == 55:
            #    print(line)
            #    print(text)
            #    input()
            """
            if line[3] == '?':
                label = line[4]
            else:
                label = line[3]
            """
            label = line[3]
            if isKoreanText(text) == True:
                xlsList_kr.append(text)
                label_kr.append(label)
                duplicateListKr.append('N')
            else:
                xlsList_en.append(text)
                label_en.append(label)
                duplicateListEn.append('N')
            idx += 1
            if idx >= no_docs_for_text:
                break  # For test ############################################################################
        else:
            if isKoreanText(text) == True:
                duplicateListKr.append('Y')
            else:
                duplicateListEn.append('Y')
    return xlsList_kr, xlsList_en, label_kr, label_en

# 대표청구항 우선
def getExcel(filename):
    global duplicateListKr
    global duplicateListEn
    print(filename)
    global root_dir
    xlsList_kr = []
    xlsList_en = []
    label_kr = []
    label_en = []
    fname = root_dir + '/' + filename
    wb = open_workbook(fname)
    FORMAT = ['발명의 명칭', '요약', '대표청구항', 'Label', 'SubLabel', '출원번호'] #5가지 컬럼만 선택 ##################################################
    xlsReader = []
    for s in wb.sheets():
        headerRow = s.row(0)
        columnIndex = [x for y in FORMAT for x in range(len(headerRow)) if y == headerRow[x].value]
        for row in range(1, s.nrows): #xls 파일 첫줄은 항목명칭 라인이라서 삭제
            currentRow = s.row(row)
            currentRowValues = [currentRow[x].value for x in columnIndex]
            xlsReader.append(currentRowValues)
        break   # 첫번재 시트만 처리한다.
    idx = 0
    file_num_list = []
    for line in xlsReader:
        if line[5] not in file_num_list:    # 중복 특허 제거
            file_num_list.append(line[5])
            # text = line[0] + ' ' + line[1] + ' ' + line[2] #########################################################
            if str(line[2]).strip() != '':
                text = line[2]  # 대표청구항
                if len(text) < 1000 - 900 * int(isKoreanText(text)):
                    text = line[0] + ' ' + text + ' ' + line[2]
                    # text = text + ' ' + line[2]
            elif str(line[1]).strip() != '':
                text = line[1]  # 요약
                # print('요약: ', len(line), text)
            elif str(line[0]).strip() != '':
                text = line[0]  # 제목
                # print('제목: ', len(line),text)
            else:
                print('No data in this line', idx)
                input()
            # For test
            # if idx == 55:
            #    print(line)
            #    print(text)
            #    input()
            """
            if line[3] == '?':
                label = line[4]
            else:
                label = line[3]
            """
            label = line[3]
            if isKoreanText(text) == True:
                xlsList_kr.append(text)
                label_kr.append(label)
                duplicateListKr.append('N')
            else:
                xlsList_en.append(text)
                label_en.append(label)
                duplicateListEn.append('N')
            idx += 1
            if idx >= no_docs_for_text:
                break  # For test ############################################################################
        else:
            if isKoreanText(text) == True:
                duplicateListKr.append('Y')
            else:
                duplicateListEn.append('Y')
    return xlsList_kr, xlsList_en, label_kr, label_en


# 어구를 자르고 ID로 변환하기
word_dic_kr = {"_MAX": 0}
word_dic_en = {"_MAX": 0}
"""
from konlpy.tag import Twitter
twitter = Twitter()
def tokenizer(text):
    # 단어의 기본형 사용
    malist = twitter.pos(text, norm=True, stem=True)
    words = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            words.append(word[0])
    return words
"""
word_data = []      # W2V 추가 학습을 위해 토크나이져를 통과하는 모든 문장들에게서 단어들을 모은다.
from konlpy.tag import Mecab
mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
def tokenizer(text):
    global word_data
    # 단어의 기본형 사용
    malist = mecab.pos(text)
    words = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외
        if ( word[1] in ['NNG', 'MAG', 'NNP', 'SL', 'NNG+NNG'] and len(word[0]) > 1 ):
            words.append(word[0])
    word_data += words
    return words

"""
def filter(text):
    # 단어의 기본형 사용
    malist = twitter.pos(text, norm=True, stem=True)
    words = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외
        if not word[1] in ["Josa", "Eomi", "Punctuation"]:
            words.append(word)
    return words
"""

def filter(text):
    # 단어의 기본형 사용
    malist = mecab.pos(text)
    words = []
    for word in malist:
        # 어미/조사/구두점 등은 대상에서 제외
        if ( word[1] in ['NNG', 'MAG', 'NNP', 'SL', 'NNG+NNG'] and len(word[0]) > 1 ):
            words.append(word)
    return words

#영문을 위한 토크나이져
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import re
def tokenizer4en_regex(sentence):
    global word_data
    filtered_sent = re.sub('[()-=.#/?:$}\\[\\]]', '', sentence.lower())

    pattern = r'''
            (?x)                    # set flag to allow verbose regexps
            (?:[a-z]\.)+            # abbreviations, e.g. U.S.A. 소문자로 앞서 변경한 관계로, a-z 소문자로 수정
            | \w+(?:-\w+)*          # words with optional internal hyphens
            | \$?\d+(?:\.\d+)?%?    # currency and percentages, e.g. $12.40, 82%
            | \.\.\.                # ellipsis
            | [][.,;"'?():-_`]      # these are separate tokens; includes ], [
            |(?:[+/\-@&*])          # special characters with meanings
            '''
    tokenizer = RegexpTokenizer(pattern)
    tokens = tokenizer.tokenize(filtered_sent)
    stopWords = set(stopwords.words('english'))
    filtered_words = [w for w in tokens if not w in stopWords]
    output = []
    for filword in filtered_words:
        if len(filword) >= 3:
            output.append(filword)
    word_data += output
    return output


def text_to_ids_kr(text):
    text = text.strip()
    words = tokenizer(text)
    result = []
    for n in words:
        n = n.strip()
        if n == "": continue
        if not n in word_dic_kr:
            wid = word_dic_kr[n] = word_dic_kr["_MAX"]
            word_dic_kr["_MAX"] += 1
            # print(wid, n)
        else:
            wid = word_dic_kr[n]
        result.append(wid)
    return result


def text_to_ids_en(text):
    words = tokenizer4en_regex(text)
    result = []
    for n in words:
        n = n.strip()
        if n == "": continue
        if not n in word_dic_en:
            wid = word_dic_en[n] = word_dic_en["_MAX"]
            word_dic_en["_MAX"] += 1
            # print(wid, n)
        else:
            wid = word_dic_en[n]
        result.append(wid)
    return result

# 딕셔너리에 단어 등록하기
def register_dic_kr(csvList):
    for list in csvList:
        #temp = list[0] + ' ' + list[1] + ' ' + list[2] # 하나의 리스트(제목, 요약, 대표청구항)를 합친 후 ################################
        temp = str(list)
        text_to_ids_kr(temp)  # 단어사전 갱신하고 wid로 변환후 리턴

def register_dic_en(csvList):
    for list in csvList:
        #temp = list[0] + ' ' + list[1] + ' ' + list[2] # 하나의 리스트(제목, 요약, 대표청구항)를 합친 후 #################################
        #temp = str(list).lower()        # lower 추가 2018-10-30
        temp = str(list)
        text_to_ids_en(temp)  # 단어사전 갱신하고 wid로 변환후 리턴


# 두 문장간의 공통 단어 추출
def get_com_words(text1, text2, KR_flag = True):
    if KR_flag == True:
        set1 = set(tokenizer(text1))
        set2 = set(tokenizer(text2))
    else:
        set1 = set(tokenizer4en_regex(text1))
        set2 = set(tokenizer4en_regex(text2))
    com_words = list(set1&set2)
    return com_words


# 문장 내에서 키워드 추출
def get_keyword(text_string, KR_flag = True):
    frequency = {}
    if KR_flag == True:
        match_pattern = tokenizer(text_string)
    else:
        match_pattern = tokenizer4en_regex(text_string)
    for word in match_pattern:
        count = frequency.get(word, 0)
        frequency[word] = count + 1
    #frequency_list = frequency.keys()
    topn = int(len(text_string)/50) #문장이 짧은 경우 적게 추출
    if topn > 10:
        topn = 10
    tmp = sorted(frequency.items(), key=lambda x: x[1], reverse=True)[:topn]
    keyword = []
    for wrd, val in tmp:
        if val > 1:
            keyword.append(wrd)
    if '상기' in keyword:
        keyword.remove('상기')
    key_sent = ''
    for wrd in match_pattern:
        if wrd in keyword:
            key_sent += (wrd + ' ')
            keyword.remove(wrd)
    return key_sent

# 특허 엑셀 시트 로드하기
def load_xls():
    csvList_kr = []
    csvList_en = []
    label_kr = []
    label_en = []
    directory = os.listdir(root_dir)
    tempList = []
    for fname in directory:
        if fname[-4:] == 'xlsx':
            tempList.append(fname)
    directory = tempList.copy()
    file_dic = {}
    idx = 0
    print(len(directory), directory, ' files found')
    for filename in directory:  # rawdata 디렉토리 내의 여려개의 csv파일을 읽어서 하나의 csvList array를 생성
        file_dic[idx] = filename
        temp_kr, temp_en, temp_label_kr, temp_label_en = getExcel(filename)
        csvList_kr += temp_kr
        csvList_en += temp_en
        label_kr += temp_label_kr
        label_en += temp_label_en
        idx += 1
        #break  # 파일 하나만 처리한다.
    return csvList_kr, csvList_en, label_kr, label_en


# [한글] 한라인 문장들 모두를 하나의 벡터로 임베딩
def emb_kr_docs(csvList_data):
    emb_lists = []
    for idx, lst in enumerate(csvList_data):
        text = str(lst)
        cnt = [0 for n in range(word_dic_kr["_MAX"])]
        ids = text_to_ids_kr(text)
        for wid in ids:
            cnt[wid] += 1
        emb_lists.append(cnt)
    return emb_lists

def emb_kr_query(query_sentence):
    emb_query = [0 for n in range(word_dic_kr["_MAX"])]  # Query문장도 임베딩
    ids = text_to_ids_kr(query_sentence)
    for wid in ids:
        emb_query[wid] += 1
    return emb_query

# [한글] IDF 구하기
def cal_kr_idf(emb_lists_kr, emb_query):
    print('[한글] IDF 구하기')
    idf_list = []
    total_num_doc = len(csvList_kr)
    for idx in range(word_dic_kr["_MAX"]):
        counter = 0
        for an_emb_list in emb_lists_kr:
            if an_emb_list[idx] > 0:
                counter += 1
        if emb_query[idx] > 0:  # Query문장도 반영
            counter += 1
        idf_list.append(total_num_doc / counter)
        if idx % 100 == 0:
            print('Calculate IDF ', idx)
    return idf_list

#Query문장도 TF
def calc_tf_emb_query(emb_query):
    sum_val = 0
    output = emb_query[:]
    for i in emb_query:
        sum_val += i
    for i in range(len(emb_query)):
        try:
            #emb_query[i] = emb_query[i] / sum_val * idf_list[i]  # emb_query x TF-IDF
            output[i] = emb_query[i] / sum_val  # emb_query x TF
        except:
            print('Error: ', sum_val, emb_query)
            input()
    return output

# Sentence Distance 구하기
def calc_dist(emb_query, emb_lists, krFlag=True):
    #dist = calc_dist_onehot(emb_query, emb_lists)      # OneHot vector를 이용한 방식
    dist = calc_dist_w2v(emb_query, emb_lists, krFlag)  # word2vector를 이용한 방식
    return dist

from scipy import linalg, mat, dot
def calc_dist_onehot(emb_query, emb_lists):
    sum_val = 0
    for i in emb_query:
        sum_val += i
    for i in range(len(emb_query)):
        try:
            # emb_query[i] = emb_query[i] / sum_val * idf_list[i]  # elist x TF-IDF
            emb_query[i] = emb_query[i] / sum_val  # elist x TF
        except:
            print('Error: ', sum_val, emb_query)
            input()
    m_emb_query = mat(emb_query)
    dist = {}
    idx = 0
    for elist in emb_lists:
        sum_val = 0
        for i in elist:
            sum_val += i
        for i in range(len(elist)):
            try:
                #elist[i] = elist[i] / sum_val * idf_list[i]  # elist x TF-IDF
                elist[i] = elist[i] / sum_val  # elist x TF
            except:
                print('Error: ', idx, sum_val, elist)
                input()
        m_elist = mat(elist)
        m_cos_sim = dot(m_elist, m_emb_query.T) / (linalg.norm(m_elist) * linalg.norm(m_emb_query))
        dist[idx] = float(m_cos_sim)
        #print(float(m_cos_sim))
        #input()
        if idx % 100 == 0:
            print('calculating distance ', idx)
        idx += 1
    return dist

from gensim.models import word2vec
import gensim
def calc_dist_w2v(emb_query, emb_lists, krFlag):
    global word_data
    # Load model
    model = word2vec.Word2Vec.load("wiki_w2v.model")
    W2V_SIZE = 100
    #model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)    # English only
    #W2V_SIZE = 300
    more_sentence = [word_data, word_data, word_data, word_data, word_data]     # min_count가 default 10이다. '바랙'이란 단어는 2번 나온다. 그래서 넉넉히 5번 반복한 것이다.
    model.build_vocab(more_sentence, update=True)             # 추가 학습 for load only
    model.train(more_sentence, total_examples=5, epochs=1)    # 추가 학습
    #print(model.wv['바랙'])
    #print(word_data)
    #input()
    sum_val = 0
    for i in emb_query:
        sum_val += i
    for i in range(len(emb_query)):
        try:
            # emb_query[i] = emb_query[i] / sum_val * idf_list[i]  # elist x TF-IDF
            emb_query[i] = emb_query[i] / sum_val  # elist x TF
        except:
            print('Error: ', sum_val, emb_query)
            input()
    m_emb_query = np.zeros(W2V_SIZE, dtype=np.float64)
    for i, value in enumerate(emb_query):
        if value != 0:
            if krFlag:
                #print([word for word, num in word_dic_kr.items() if num == i][0])  # 딕셔너리 내의 단어를 리턴
                m_emb_query += (model.wv[[word for word, num in word_dic_kr.items() if num == i][0]] * value)
            else:
                #print([word for word, num in word_dic_en.items() if num == i][0])  # 딕셔너리 내의 단어를 리턴
                m_emb_query += (model.wv[[word for word, num in word_dic_en.items() if num == i][0]] * value)
    dist = {}
    for idx, elist in enumerate(emb_lists):
        sum_val = 0
        for i in elist:
            sum_val += i
        for i in range(len(elist)):
            try:
                #elist[i] = elist[i] / sum_val * idf_list[i]  # elist x TF-IDF
                elist[i] = elist[i] / sum_val  # elist x TF
            except:
                print('Error: ', idx, sum_val, elist)
                input()
        m_elist = np.zeros(W2V_SIZE, dtype=np.float64)
        for i, value in enumerate(elist):
            if value != 0:
                if krFlag:
                    # print([word for word, num in word_dic_kr.items() if num == i][0])  # 딕셔너리 내의 단어를 리턴
                    m_elist += (model.wv[[word for word, num in word_dic_kr.items() if num == i][0]] * value)
                else:
                    # print([word for word, num in word_dic_en.items() if num == i][0])  # 딕셔너리 내의 단어를 리턴
                    m_elist += (model.wv[[word for word, num in word_dic_en.items() if num == i][0]] * value)
        m_cos_sim = dot(m_elist, m_emb_query.T) / (linalg.norm(m_elist) * linalg.norm(m_emb_query))
        dist[idx] = float(m_cos_sim)
        #print(idx, float(m_cos_sim))
        #input()
        if idx % 100 == 0:
            print('calculating distance ', idx)
    return dist


def get_normalized_dist(dist):
    nor_dist = {}
    sorted_rank = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    max_sim_val_kr = sorted_rank[0][1]
    sorted_rank = sorted(dist.items(), key=lambda x: x[1], reverse=False)
    min_sim_val_kr = sorted_rank[0][1]
    print('Max and Min ', max_sim_val_kr, min_sim_val_kr)
    if min_sim_val_kr < 0:
        vias_val = -1. * min_sim_val_kr
    else:
        vias_val = 0.
    for i in range(len(dist)):
        nor_dist[i] = (dist[i] + vias_val) / (max_sim_val_kr - min_sim_val_kr) * 100.0  # Normalization
    return nor_dist

# [영문] 한라인 문장들 모두를 하나의 벡터로 임베딩
def emb_en_docs(csvList_data):
    emb_lists = []
    for lst in csvList_data:
        # text = lst[0] + lst[1] + lst[2]  # 하나의 리스트(제목, 요약, 대표청구항)를 합친 후 ################################
        text = str(lst)
        cnt = [0 for n in range(word_dic_en["_MAX"])]
        ids = text_to_ids_en(text)
        for wid in ids:
            cnt[wid] += 1
        emb_lists.append(cnt)
    return emb_lists

def emb_en_query(query_sentence):
    emb_query = [0 for n in range(word_dic_en["_MAX"])]  # Query문장도 임베딩
    ids = text_to_ids_en(query_sentence)
    for wid in ids:
        emb_query[wid] += 1
    return emb_query

# [영문] IDF 구하기
def cal_en_idf(csvList_en, emb_query):
    print('[영문] IDF구하기')
    idf_list = []
    total_num_doc = len(csvList_en)
    for idx in range(word_dic_en["_MAX"]):
        counter = 0
        for an_emb_list in emb_lists_en:
            if an_emb_list[idx] > 0:
                counter += 1
        if emb_query[idx] > 0:  # Query문장도 반영
            counter += 1
        idf_list.append(total_num_doc / counter)
        if idx % 100 == 0:
            print('Calculate IDF ', idx)
    return idf_list

def read_query(fname_kr, fname_en):
    query_sent_kr = open(fname_kr, 'r').read()
    query_sent_en = open(fname_en, 'r').read()
    return query_sent_kr, query_sent_en



# 랭킹 순으로 결과 문장 저장
from xlwt import Workbook
from xlutils.copy import copy
def store_result(fname, fname_cmt, sorted_rank, csvList, query_sent, KR_flag = True):
    f = open(fname, 'w')
    fcmt = open(fname_cmt, 'w')
    for i in range(len(sorted_rank)):
        text = "%.4f%% " % (sorted_rank[i][1]) + ' ' + str(csvList[int(sorted_rank[i][0])]) + '\n'
        tmp_list = get_com_words(query_sent, csvList[int(sorted_rank[i][0])], KR_flag)
        if KR_flag:
            text_with_com_words = label_kr[sorted_rank[i][0]] + ' ' + str(get_keyword(str(csvList[int(sorted_rank[i][0])]), KR_flag)) + ' ' + str(tmp_list) + ' ' + text
        else:
            text_with_com_words = label_en[sorted_rank[i][0]] + ' ' + str(get_keyword(str(csvList[int(sorted_rank[i][0])]), KR_flag)) + ' ' + str(tmp_list) + ' ' + text
        f.write(text)
        #fcmt.write("%.4f%% " % (100. * (float(i) / float(len(sorted_rank)))))
        fcmt.write("%4d " % (i))
        fcmt.write(text_with_com_words)
    f.close()
    fcmt.close()
    # 엑셀파일 업데이트
    if KR_flag:
        # 중복 제거한 엑셀파일 만들기
        wb_read = open_workbook('./TestData/Consulting/DearGen/DearGen_KR_All.xlsx')
        s = wb_read.sheet_by_index(0)
        # For write
        wb_write = Workbook()
        Sheet1 = wb_write.add_sheet('Sheet1')
        clm_index = 0
        for i in range(s.nrows):
            if i == 0:
                for j in range(s.ncols):
                    Sheet1.write(i, j, s.cell(i, j).value)      # 첫번째 컬럼 (컬럼 이름)
            else:
                if duplicateListKr[i-1] == 'N':
                    for j in range(s.ncols):
                        Sheet1.write(clm_index + 1, j, s.cell(i, j).value)
                    clm_index += 1
        wb_write.save('./TestData/duplicationfree_kr.xls')
        #print('duplicationfree_kr.xls is created')
        #input()
        wb_read = open_workbook('./TestData/duplicationfree_kr.xls')
        wb_write = copy(wb_read)
        s = wb_write.get_sheet(0)
        for idx, val in sorted_rank:
            s.write(idx+1, 6, val)
        wb_write.save('./TestData/DearGen_KR_All_update.xls')
    else:
        wb_read = open_workbook('./TestData/Consulting/DearGen/DearGen_EN_All.xlsx')
        s = wb_read.sheet_by_index(0)
        # For write
        wb_write = Workbook()
        Sheet1 = wb_write.add_sheet('Sheet1')
        clm_index = 0
        for i in range(s.nrows):
            if i == 0:
                for j in range(s.ncols):
                    Sheet1.write(i, j, s.cell(i, j).value)      # 첫번째 컬럼 (컬럼 이름)
            else:
                if duplicateListEn[i-1] == 'N':
                    for j in range(s.ncols):
                        Sheet1.write(clm_index + 1, j, s.cell(i, j).value)
                    clm_index += 1
        wb_write.save('./TestData/duplicationfree_en.xls')
        wb_read = open_workbook('./TestData/duplicationfree_en.xls')
        wb_write = copy(wb_read)
        s = wb_write.get_sheet(0)
        for idx, val in sorted_rank:
            s.write(idx+1, 6, val)
        wb_write.save('./TestData/DearGen_EN_All_update.xls')
    return


# Update Dist with New query words
def update_dist_kr(fname_p, fname_n, base_val, emb_lists_kr, query_sent_kr, dist_kr):
    isEmpty = True
    if os.path.exists(fname_p) != True:
        print(fname_p, ' not found')
    else:
        isEmpty = False
        new_query_p = open(fname_p, 'r').read()
        if new_query_p != '':
            emb_new_query = emb_kr_query(new_query_p)
            dist_delta = calc_dist(emb_new_query, emb_lists_kr)
            #nor_dist_delta = get_normalized_dist(dist_delta)
            nor_dist_delta = dist_delta
            for i in range(len(nor_dist_delta)):
                dist_kr[i] += nor_dist_delta[i] / 100. * base_val * 0.1    # 10%만 반영
    if os.path.exists(fname_n) != True:
        print(fname_n, ' not found')
    else:
        isEmpty = False
        new_query_n = open(fname_n, 'r').read()
        if new_query_n != '':
            emb_new_query = emb_kr_query(new_query_n)
            dist_delta = calc_dist(emb_new_query, emb_lists_kr)
            #nor_dist_delta = get_normalized_dist(dist_delta)
            nor_dist_delta = dist_delta
            for i in range(len(nor_dist_delta)):
                dist_kr[i] -= nor_dist_delta[i] / 100. * base_val * 0.5    # 50%만 반영
    # sorted rank 한글
    if isEmpty == True:
        sorted_rank = {}
    else:
        #dist_kr = get_normalized_dist(dist_kr)
        dist_kr = dist_kr
        sorted_rank = sorted(dist_kr.items(), key=lambda x: x[1], reverse=True)
        store_result("./TestData/result_kr.txt", "./TestData/result_cmt_kr.txt", sorted_rank, csvList_kr, query_sent_kr, KR_flag = True)
    return sorted_rank


# Update Dist with New query words
def update_dist_en(fname_p, fname_n, base_val, emb_lists_en, query_sent_en, dist_en):
    isEmpty = True
    if os.path.exists(fname_p) != True:
        print(fname_p, ' not found')
    else:
        isEmpty = False
        new_query_p = open(fname_p, 'r').read()
        if new_query_p != '':
            emb_new_query = emb_en_query(new_query_p)
            dist_delta = calc_dist(emb_new_query, emb_lists_en)
            #nor_dist_delta = get_normalized_dist(dist_delta)
            nor_dist_delta = dist_delta
            for i in range(len(nor_dist_delta)):
                dist_en[i] += nor_dist_delta[i] / 100. * base_val * 0.1 # 10%만 반영
    if os.path.exists(fname_n) != True:
        print(fname_n, ' not found')
    else:
        isEmpty = False
        new_query_n = open(fname_n, 'r').read()
        if new_query_n != '':
            emb_new_query = emb_en_query(new_query_n)
            dist_delta = calc_dist(emb_new_query, emb_lists_en)
            #nor_dist_delta = get_normalized_dist(dist_delta)
            nor_dist_delta = dist_delta
            for i in range(len(nor_dist_delta)):
                dist_en[i] -= nor_dist_delta[i] / 100. * base_val * 0.5 # 50%만 반영
    # sorted rank 영문
    if isEmpty == True:
        sorted_rank = {}
    else:
        #dist_en = get_normalized_dist(dist_en)
        sorted_rank = sorted(dist_en.items(), key=lambda x: x[1], reverse=True)
        store_result("./TestData/result_en.txt", "./TestData/result_cmt_en.txt", sorted_rank, csvList_en, query_sent_en, KR_flag = False)
    return sorted_rank





# MAIN ###########################################################################################################
# 한글세션 시작

# 쿼리문장 읽어 오기
print('쿼리문장 읽어 오기')
query_sent_kr, query_sent_en = read_query("/home/hwang/PycharmProjects/SimSent/TestData/Consulting/query_디어젠_kr.txt", \
                                          "/home/hwang/PycharmProjects/SimSent/TestData/Consulting/query_디어젠_en.txt")
print(query_sent_kr, query_sent_en)
input()

# 특허 엑셀 시트 로드하기
print('xls파일 파싱하여 한글, 영문 리스트 각각 생성하기')
no_docs_for_text = 100000000
in_text = input("빠른 테스트를 위해 xls파일 내 앞의 일부 문서만 사용할 수 있습니다. 몇라인을 사용해서 테스트할지 숫자를 입력해 주세요.[모든 라인을 사용할 시에는 그냥 Enter]")
if in_text != '':
    no_docs_for_text = int(in_text)
print('no_docs_for_text : ', no_docs_for_text)

csvList_kr, csvList_en, label_kr, label_en = load_xls()
print('xls 한글 vs 영문: ', len(csvList_kr), len(csvList_en))

"""
trueCntKor = 0
falseCntKor = 0
ambiCntKor = 0
for lab in label_kr:
    if lab == 'T':
        trueCntKor += 1
    elif lab == 'F':
        falseCntKor += 1
    elif lab == '?':
        ambiCntKor += 1
    else:
        print('Unknown label')
        input()
print('Kor(P N Ambi): ', trueCntKor, falseCntKor, ambiCntKor)
trueCntEn = 0
falseCntEn = 0
ambiCntEn = 0
for lab in label_en:
    if lab == 'T':
        trueCntEn += 1
    elif lab == 'F':
        falseCntEn += 1
    elif lab == '?':
        ambiCntEn += 1
    else:
        print('Unknown label')
        input()
print('Eng(P N Ambi): ', trueCntEn, falseCntEn, ambiCntEn)
input()
"""
"""
# 단어 딕셔너리 만들기
print('한글 단어사전 등록하기')
register_dic_kr(csvList_kr)  # 엑셀 내 문장들을 단어사전에 등록
text_to_ids_kr(query_sent_kr)  # Query 문장도 단어사전에 등록
print('Kr Dic ', len(word_dic_kr))

# [한글] 한라인 문장들 모두를 하나의 벡터로 임베딩
print('[한글] 한라인 문장들 모두를 하나의 벡터로 임베딩')
emb_lists_kr = emb_kr_docs(csvList_kr)
emb_query_tmp = emb_kr_query(query_sent_kr)

#Query문장도 TF
idf_list = cal_kr_idf(emb_lists_kr, emb_query_tmp)
emb_query_kr = calc_tf_emb_query(emb_query_tmp)

# [한글] Sentence Distance 구하기
print('[한글] Sentence Distance 구하기')
#dist_kr = get_normalized_dist(calc_dist(emb_query_kr, emb_lists_kr))
dist_kr = calc_dist(emb_query_kr, emb_lists_kr)

#[한글] Customer Review
print('[한글] Customer Review')
print('[한글]랭킹 순으로 결과 문장 저장')
sorted_rank = sorted(dist_kr.items(), key=lambda x: x[1], reverse=True)
"""
"""
# Calculate Precision
TPosi = 0
FPosi = 0
idx = 0
for posi, val in sorted_rank:
    if idx >= trueCntKor:
        break
    if label_kr[posi] == 'T':
        TPosi += 1
    elif label_kr[posi] == '?':
        TPosi += 1
    elif label_kr[posi] == 'F':
        FPosi += 1
    else:
        print('Unknown label in sorted_rank', label_kr[posi])
        input()
    idx += 1
print('Result(TP FP): ', TPosi, FPosi)
TNega = 0
FNega = 0
idx = 0
for posi, val in sorted_rank:
    if idx < trueCntKor:
        idx += 1
        continue
    if label_kr[posi] == 'F':
        TNega += 1
    elif label_kr[posi] == '?':
        TNega += 1
    elif label_kr[posi] == 'T':
        FNega += 1
    else:
        print('Unknown label in sorted_rank', label_kr[posi])
        input()
    idx += 1
print('Result(TN FN): ', TNega, FNega)
precision = TPosi / (TPosi + FPosi)
recall = TPosi / (TPosi + FNega)
print('result(precision recall): ', precision, recall)
print('Boundary Value: ', sorted_rank[0], sorted_rank[trueCntKor-1], sorted_rank[trueCntKor+falseCntKor-1])
input()
"""
"""
store_result("result_kr.txt", "result_cmt_kr.txt", sorted_rank, csvList_kr, query_sent_kr, KR_flag = True)
q_count = 10    # 전체 문서 중 특정 위치에서 검토하는 특허 문서의 개수
print('Review intermediate result')
import shutil
index_val = int(float(len(csvList_kr))/2.0)
search_history = []
stop_flag = False
if os.path.exists("PositiveWords.txt") == True:
    os.remove("PositiveWords.txt")  # 직전에 비정상적인 종료에 대비
if os.path.exists("NegativeWords.txt") == True:
    os.remove("NegativeWords.txt")  # 직전에 비정상적인 종료에 대비
while True :
    true_count = 0
    search_history.append("%.4f%%" % (100. * (float(index_val) / float(len(csvList_kr)))))
    for i in range(q_count):
        text = "%.4f%% " % (sorted_rank[index_val][1]) + ' ' + str(csvList_kr[int(sorted_rank[index_val][0])])
        tmp_list = get_com_words(query_sent_kr, csvList_kr[int(sorted_rank[index_val][0])], KR_flag=True)
        text_with_com_words = str(
            get_keyword(str(csvList_kr[int(sorted_rank[index_val][0])]), KR_flag=True)) + ' ' + str(
            tmp_list) + ' ' + text
        print("%.4f%% " % (100. * (float(index_val) / float(len(csvList_kr)))), text_with_com_words)
        while True:
            text_in = input('찾으시는 특허가 맞습니까? [Yes(y) / No(n) / Quit(q)] :')
            if text_in == 'q':
                stop_flag = True
                break
            elif text_in == 'y':
                true_count += 1
                f = open('PositiveWords.txt', 'a')
                f.write(get_keyword(str(csvList_kr[int(sorted_rank[index_val][0])]), KR_flag=True))
                f.write('\n')
                f.close()
                break
            elif text_in == 'n':
                f = open('NegativeWords.txt', 'a')
                f.write(get_keyword(str(csvList_kr[int(sorted_rank[index_val][0])]), KR_flag=True))
                f.write('\n')
                f.close()
                break
            else:
                print("잘못된 입력입니다. 다시 입력해 주세요")

        if stop_flag == True:
            break

        index_val -= 1
        print('') # 한라인 비우기

    if stop_flag == True:
        break

    index_val += q_count
    base_val = sorted_rank[index_val][1]

    sorted_rank = update_dist_kr('PositiveWords.txt', 'NegativeWords.txt', base_val, emb_lists_kr, query_sent_kr, dist_kr)
    if sorted_rank == {}:
        print('[Error] Empty sorted_rank')
        break

    fname = 'PosiWords-' + str(int(100.*float(index_val)/float(len(csvList_kr)))) + '_' + str(i%5) + '_kr.txt'
    if os.path.exists("PositiveWords.txt") != True:
        print("No PositiveWords.txt")
    else:
        shutil.move("PositiveWords.txt", fname)
    fname = 'NegaWords-' + str(int(100.*float(index_val)/float(len(csvList_kr)))) + '_' + str(i%5) + '_kr.txt'
    if os.path.exists("NegativeWords.txt") != True:
        print("No NegativeWords.txt")
    else:
        shutil.move("NegativeWords.txt", fname)

    if index_val > len(csvList_kr)/2:
        window_size = len(csvList_kr) - index_val
    else:
        window_size = index_val
    if true_count > q_count/2:
        index_val += int(float(true_count-q_count/2.)/float(q_count/2.)*float(window_size)*0.5)
        if index_val >= len(csvList_kr):
            index_val = len(csvList_kr)
    elif true_count < q_count/2:
        index_val -= int(float(q_count/2. - true_count)/float(q_count/2.)*float(window_size)*0.5)
        if index_val < q_count:
            index_val = q_count
    print('Search History = ', search_history)

#메모리 해제
csvList_kr = []
emb_lists_kr = []
dist_kr = {}
word_dic_kr = {}
"""


#######################################################################################################
# 영어 세션 시작
# 단어 딕셔너리 만들기
word_data.clear()   # 추가 학습 데이터 클리어
print('영문 단어사전 등록하기')
register_dic_en(csvList_en)  # 엑셀 내 문장들을 단어사전에 등록
text_to_ids_en(query_sent_en)  # 입력 문장도 단어사전에 등록
print('En Dic ', len(word_dic_en))

# [영문] 한라인 문장들 모두를 하나의 벡터로 임베딩
print('[영문] 한라인 문장들 모두를 하나의 벡터로 임베딩')
emb_lists_en = emb_en_docs(csvList_en)
emb_query_tmp = emb_en_query(query_sent_en)

#영문 Query문장도 TF
idf_list = cal_en_idf(emb_lists_en, emb_query_tmp)
emb_query = calc_tf_emb_query(emb_query_tmp)

# [영문] Sentence Distance 구하기
import itertools
print('[영문] Sentence Distance 구하기')
#dist_en = get_normalized_dist(calc_dist(emb_query, emb_lists_en))
dist_en = calc_dist(emb_query, emb_lists_en, krFlag = False)

#[영어] Customer Review
print('[영어] Customer Review')
print('[영문]랭킹 순으로 결과 문장 저장')
sorted_rank = sorted(dist_en.items(), key=lambda x: x[1], reverse=True)
"""
# Calculate Precision
TPosi = 0
FPosi = 0
idx = 0
for posi, val in sorted_rank:
    if idx >= trueCntEn:
        break
    if label_en[posi] == 'T':
        TPosi += 1
    elif label_en[posi] == '?':
        TPosi += 1
    elif label_en[posi] == 'F':
        FPosi += 1
    else:
        print('[KR]Unknown label in sorted_rank', label_en[posi])
        input()
    idx += 1
print('Result(TP FP): ', TPosi, FPosi)
TNega = 0
FNega = 0
idx = 0
for posi, val in sorted_rank:
    if idx < trueCntEn:
        idx += 1
        continue
    if label_en[posi] == 'F':
        TNega += 1
    elif label_en[posi] == '?':
        TNega += 1
    elif label_en[posi] == 'T':
        FNega += 1
    else:
        print('Unknown label in sorted_rank', label_en[posi])
        input()
    idx += 1
print('Result(TN FN): ', TNega, FNega)
precision = TPosi / (TPosi + FPosi)
recall = TPosi / (TPosi + FNega)
print('result(precision recall): ', precision, recall)
print('Boundary Value: ', sorted_rank[0], sorted_rank[trueCntEn-1], sorted_rank[trueCntEn+falseCntEn-1])
input('pause..')
#for i in range(trueCntEn+falseCntEn):
#    if i < trueCntEn:
#        continue
#    print(i, sorted_rank[i])
#input()
"""
store_result("result_en.txt", "result_cmt_en.txt", sorted_rank, csvList_en, query_sent_en, KR_flag = False)
q_count = 10    # 전체 문서 중 특정 위치에서 검토하는 특허 문서의 개수
print('Review intermediate result')
import shutil
index_val = int(float(len(csvList_en))/2.0)
search_history = []
stop_flag = False
while True :
    true_count = 0
    search_history.append("%.4f%%" % (100. * (float(index_val) / float(len(csvList_en)))))
    for i in range(q_count):
        text = "%.4f%% " % (sorted_rank[index_val][1]) + ' ' + str(
            csvList_en[int(sorted_rank[index_val][0])])
        tmp_list = get_com_words(query_sent_en, csvList_en[int(sorted_rank[index_val][0])], KR_flag=False)
        text_with_com_words = str(
            get_keyword(str(csvList_en[int(sorted_rank[index_val][0])]), KR_flag=False)) + ' ' + str(
            tmp_list) + ' ' + text
        print("%.4f%% " % (100. * (float(index_val) / float(len(csvList_en)))), text_with_com_words)
        while True:
            text_in = input('찾으시는 특허가 맞습니까? [Yes(y) / No(n) / Quit(q)] :')
            if text_in == 'q':
                stop_flag = True
                break
            elif text_in == 'y':
                true_count += 1
                f = open('PositiveWords.txt', 'a')
                f.write(get_keyword(str(csvList_en[int(sorted_rank[index_val][0])]), KR_flag=False))
                f.write('\n')
                f.close()
                break
            elif text_in == 'n':
                f = open('NegativeWords.txt', 'a')
                f.write(get_keyword(str(csvList_en[int(sorted_rank[index_val][0])]), KR_flag=False))
                f.write('\n')
                f.close()
                break
            else:
                print("잘못된 입력입니다. 다시 입력해 주세요")

        if stop_flag == True:
            break

        index_val -= 1
        print('') # 한라인 비우기

    if stop_flag == True:
        break

    index_val += q_count
    base_val = sorted_rank[index_val][1]
    sorted_rank = update_dist_en('PositiveWords.txt', 'NegativeWords.txt', base_val, emb_lists_en, query_sent_en, dist_en)
    if sorted_rank == {}:
        print('[Error] Empty sorted_rank')
        break

    fname = 'PosiWords-' + str(int(100.*float(index_val)/float(len(csvList_en)))) + '_' + str(i%5) + '_en.txt'
    if os.path.exists("PositiveWords.txt") != True:
        print("No PositiveWords.txt")
    else:
        shutil.move("PositiveWords.txt", fname)
    fname = 'NegaWords-' + str(int(100.*float(index_val)/float(len(csvList_en)))) + '_' + str(i%5) + '_en.txt'
    if os.path.exists("NegativeWords.txt") != True:
        print("No NegativeWords.txt")
    else:
        shutil.move("NegativeWords.txt", fname)

    if index_val > len(csvList_kr)/2:
        window_size = len(csvList_kr) - index_val
    else:
        window_size = index_val
    if true_count > q_count/2:
        index_val += int(float(true_count-q_count/2.)/float(q_count/2.)*float(window_size)*0.5)
        if index_val >= len(csvList_en):
            index_val = len(csvList_en)
    elif true_count < q_count/2:
        index_val -= int(float(q_count/2. - true_count)/float(q_count/2.)*float(window_size)*0.5)
        if index_val < q_count:
            index_val = q_count
    print('Search History = ', search_history)

#메모리 해제
csvList_en = []
emb_lists_en = []
dist_en = {}
word_dic_en = {}

