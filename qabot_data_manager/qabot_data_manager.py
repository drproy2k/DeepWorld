########################################
# QAchatbot Data Manager
# 2018.1.11. Th Billy Inseong Hwang
# 하나카드 QA챗봇
# note : 영문자한글전환,순서변환만 함. 개체명사전 반영필요
# note : 영문자를한글로변환한문장도순서변환처리함. 한글만있는경우한번더순서변환처리해서개수를맞
########################################

import codecs
import os
from konlpy.tag import Mecab
root_dir = './data'

max_num_of_entity = 7 ##################### Global ############################

if not os.path.isdir(root_dir): #디렉토리 유무 확인
    os.mkdir(root_dir) #없으면 생성하라.
abc_kr_dic = {'A': '에이', 'B': '비', 'C': '씨', 'D': '디', 'E': '이', 'F': '에프', 'G': '지', 'H': '에이치', 'I': '아이', 'J': '제이', 'K': '케이',
              'L': '엘', 'M': '엠', 'N': '엔', 'O': '오', 'P': '피', 'Q': '큐', 'R': '알', 'S': '에스', 'T': '티', 'U': '유',
              'V': '브이', 'W': '더블유', 'X': '엑스', 'Y': '와이', 'Z': '제트',
              'a': '에이', 'b': '비', 'c': '씨', 'd': '디', 'e': '이', 'f': '에프', 'g': '지', 'h': '에이치', 'i': '아이', 'j': '제이', 'k': '케이',
              'l': '엘', 'm': '엠', 'n': '엔', 'o': '오', 'p': '피', 'q': '큐', 'r': '알', 's': '에스', 't': '티', 'u': '유',
              'v': '브이', 'w': '더블유', 'x': '엑스', 'y': '와이', 'z': '제트'}
word_bag = set(['지니'])
fp = codecs.open("qadataset.txt", "r", encoding="utf-8")
text = fp.read()
fp.close()
text = text.replace('\r', ' ')
#text = text.replace('\n', '')
#print(text)
print('Total Question = ', text.count('Q.'))
qst_data_list = []
ans_data_list = []
qa_list = text.split("Q.")
for qa_datum in qa_list :
    if qa_datum.find("A.") > 0 :
        temp_list = qa_datum.split("A.")
        qst_data_list.append(temp_list[0])
        ans_data_list.append(temp_list[1])

num_of_qsts = len(qst_data_list)

for i in range(num_of_qsts) :
    qst_data_list[i] = qst_data_list[i].strip()  #앞뒤 불필요한 공백 제거
    qst_data_list[i] = qst_data_list[i].replace('\n', '')
    qst_data_list[i] = qst_data_list[i].replace('?', '')
    ans_data_list[i] = ans_data_list[i].strip()  # 앞뒤 불필요한 공백 제거

#print(qst_data_list[1])
#print(ans_data_list[1])

#mecab = Mecab('/home/billy/mecab-ko-dic-2.0.1-20150920')
mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')

def perm(a): # permutation을 얻어내기 위한 함수를 정의한다. 인수로는 나열해야할 숫자들을 리스트로 받는다.
    length=len(a) # 나열해야할 리스트의 길이(개수)를 계산한다.
    if length==1: # 만약 나열해야할 리스트에 원소가 1개 밖에 없다면 그냥 인수로 받았던 리스트를 반환한다.
        return [a]
    else:
        result=[] # 결과가 저장 될 빈 리스트를 생성한다.
        for i in a: # 리스트 a의 원소들을 하나씩 i에 받는다.
            b=a.copy() # b에 인수로 받은 리스트를 복사한다.
            b.remove(i) # 맨 앞자리에 i가 오는 경우 일단 b에서 i를 제거하고
            b.sort() # i가 제거된 b를 오름차순으로 정렬한다.
            for j in perm(b): # 함수를 재귀적으로 사용하여 b에 속한 원소들을 나열하는 순열의 모든 경우를 리스트로 받는다.
                j.insert(0, i) # 다시 맨 앞자리에 i를 추가해 준다.
                if j not in result: # result에 j 가 존재하지 않는다면 result에 j를 추가한다.
                    result.append(j) # 이렇게 하면 같은 것이 있는 순열의 모든 경우도 나열하는 것이 가능하다.
        return(result)

def augmentation_morpphed(NMNS_text, num, label):
    augmented_morpphed_list = []
    if num>7 : num=7  #속도가 너무 느려지는 것을 방지 ######### Global max_num_of_entity has to be the same value ############################
    a = [x for x in range(0, int(num))]
    perm_lists = perm(a)
    words = NMNS_text.split(' ')
    for perm_list in perm_lists :
        temp = ''
        for i in perm_list :
            temp = temp + words[i] + ' '
        temp = temp.strip()
        augmented_morpphed_list.append(temp)
        augmented_morpphed_list.append(label)
    return augmented_morpphed_list

def abc_exist(text):
    abc_exist_flag = False
    for i in range(len(text)):
        ascii_code = ord(text[i])
        if ascii_code>=65 and ascii_code<=90 or ascii_code>=97 and ascii_code<=122 :
            abc_exist_flag = True
    return abc_exist_flag


for i in range(num_of_qsts) :
    tmp_string = qst_data_list[i]
    tmp_string = tmp_string.lower()
    morpphed_text = mecab.pos(tmp_string)
    NMNS_text = '' #한글용
    NMNS_text_en = '' #영문용
    abc_exist_flag = False
    word_count = 0
    word_count_en = 0
    for word_tag in morpphed_text:
        #print(word_tag)
        if (word_tag[1] in ['NNG', 'MAG', 'NNP', 'SL', 'NNG+NNG'] and len(word_tag[0]) >= 1):  # Check only Noun
            tmp_string = word_tag[0]
            tmp_string = tmp_string.replace(" ", "")
            word_bag.add(tmp_string)
            NMNS_text += tmp_string + ' '
            word_count += 1
            if abc_exist(tmp_string) == True: abc_exist_flag = True

    if word_count < max_num_of_entity:
        for word_tag in morpphed_text:
            # print(word_tag)
            if (word_tag[1] in ['VV+ETM', 'NP', 'VA+ETM', 'VV+ETM', 'XSN', 'MM', 'VV', 'VA', 'NNBC',  'SN', 'NR',
                                'VV+EC', 'VV+EP'] and len(word_tag[0]) >= 1):  # Check only Noun
                tmp_string = word_tag[0]
                tmp_string = tmp_string.replace(" ", "")
                word_bag.add(tmp_string)
                NMNS_text += tmp_string + ' '
                word_count += 1
                if abc_exist(tmp_string) == True: abc_exist_flag = True

    if word_count<3 : #모니터링
        print(morpphed_text)
        print(NMNS_text)
        print(word_count)

    if word_count < max_num_of_entity :
        NMNS_text += '@ '*(max_num_of_entity-word_count)
    NMNS_text = NMNS_text.rstrip()
    word_count = max_num_of_entity

    label = i.__str__()
    #print('label = ', label)
    augmented_morpphed_list = augmentation_morpphed(NMNS_text, word_count, label)
    #print(augmented_morpphed_list)
    fname = root_dir+'/'+'label'+label+'.txt'
    f = open(fname, 'w')
    for  word_txt in augmented_morpphed_list:
        f.write(word_txt+'\n')
    f.close()

    if abc_exist_flag == True:
        for word_tag in morpphed_text:
            # print(word_tag)
            if (word_tag[1] in ['NNG', 'MAG', 'NNP', 'SL', 'NNG+NNG'] and len(word_tag[0]) >= 1):  # Check only Noun
                tmp_string = word_tag[0]
                tmp_string = tmp_string.replace(" ", "")
                if abc_exist(tmp_string) == True:
                    temp_str = ''
                    #print(word_tag[0])
                    #print(len(word_tag[0]))
                    #print(word_tag[0][0])
                    for j in range(len(tmp_string)) :
                        if tmp_string[j] in abc_kr_dic.keys():
                            temp_str += abc_kr_dic[tmp_string[j]]
                        else:
                            temp_str += tmp_string[j]
                    NMNS_text_en += temp_str + ' '
                    word_bag.add(temp_str)
                else :
                    NMNS_text_en += tmp_string + ' '
                word_count_en += 1
                if tmp_string.isalpha() == True: abc_exist_flag = True

        if word_count_en < max_num_of_entity:
            for word_tag in morpphed_text:
                # print(word_tag)
                if (word_tag[1] in ['VV+ETM', 'NP', 'VA+ETM', 'VV+ETM', 'XSN', 'MM', 'VV',
                                    'VA', 'NNBC', 'SN', 'NR',
                                    'VV+EC', 'VV+EP'] and len(word_tag[0]) >= 1):  # Check only Noun
                    tmp_string = word_tag[0]
                    tmp_string = tmp_string.replace(" ", "")
                    if abc_exist(tmp_string) == True:
                        temp_str = ''
                        # print(word_tag[0])
                        # print(len(word_tag[0]))
                        # print(word_tag[0][0])
                        for j in range(len(tmp_string)):
                            temp_str += abc_kr_dic[tmp_string[j]]
                        NMNS_text_en += temp_str + ' '
                        word_bag.add(temp_str)
                    else:
                        NMNS_text_en += tmp_string + ' '
                    word_count_en += 1
                    if tmp_string.isalpha() == True: abc_exist_flag = True


        if word_count_en < 3:  # 모니터링
            print(morpphed_text)
            print(NMNS_text_en)
            print(word_count_en)

        if word_count_en < max_num_of_entity:
            NMNS_text_en += '@ ' * (max_num_of_entity - word_count_en)
        NMNS_text_en = NMNS_text_en.rstrip()
        word_count_en = max_num_of_entity

        label = i.__str__()
        #print('label = ', label)
        augmented_morpphed_list = augmentation_morpphed(NMNS_text_en, word_count_en, label)
        #print(augmented_morpphed_list)
        #print(word_bag)

        f = open(fname, 'a')  #앞서 쓴 문서에 추가해서 넣는다
        for word_txt in augmented_morpphed_list:
            f.write(word_txt + '\n')
        f.close()
    else : #영문자가 없으면 한글학습데이터를 중복해서 한번 더 써서 전체문장개수를 맞춘다
        f = open(fname, 'a')  #앞서 쓴 문서에 추가해서 넣는다
        for word_txt in augmented_morpphed_list:
            f.write(word_txt + '\n')
        f.close()

#단어 말뭉치 저장
fname = root_dir+'/'+'word_bag.txt'
f = open(fname, 'w')  #앞서 쓴 문서에 추가해서 넣는다
for word_txt in word_bag:
    f.write(word_txt + ' ')
f.close()


#
# Batch file generator
#
label_size  = 367
for batch in range(100):
    fw_name = root_dir + '/' + 'batch' + str(batch) + '.txt'
    fw = open(fw_name, 'w')

    for f in range(label_size):
        fname = root_dir + '/' + 'label' + str(f) + '.txt'
        fp = open(fname, 'r')
        data = fp.read()
        fp.close()
        sentence_list = data.split('\n')

        if batch > 0 :
            j = 0
            for encode_raw in sentence_list:
                j += 1
                if j >= batch*100*2 : break

        i = 0
        j = 0
        for encode_raw in sentence_list:

            if encode_raw == '' :
                break
            if j < batch*100*2:
                j += 1
            else:
                fw.write(encode_raw + '\n')
                i += 1
            if i >= 100*2 : break

    fw.close()
