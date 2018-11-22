################################################################################################
# Plain Trade-mark detector
# File Name : PlainTMdetector.py
# Date : 2018-05-21
# Name : Billy Inseong Hwang
#
import cv2
import numpy as np
import os, sys
IMG_SOURCE_WIDTH = 300
IMG_SOURCE_HEIGHT = 300
test_image_dir = 'data/issue_images_rs'
#test_image_dir = 'data/test_images'
#test_image_dir = 'data/mini_test_images'
forChar_image_dir = 'data/forChar_images'
korChar_image_dir = 'data/korChar_images'


def removeHorizontalLines(img):    # limit is a minimum length to be deleted
    height, width = img.shape
    bg_color = img[0,0]
    for y in range(height):
        obj_flag = False
        for x in range(width):
            if obj_flag == False and img[y,x] != bg_color:
                obj_flag = True
                x_start = x
            elif obj_flag == True and img[y,x] == bg_color:
                obj_flag = False
                x_end = x
                if (x_end - x_start) >= width*0.7:
                    img[y,x_start:x_end] = bg_color
                    break
    return img

def removeVerticalLines(img):    # limit is a minimum length to be deleted
    height, width = img.shape
    bg_color = img[0,0]
    for x in range(width):
        obj_flag = False
        for y in range(height):
            if obj_flag == False and img[y,x] != bg_color:
                obj_flag = True
                y_start = y
            elif obj_flag == True and img[y,x] == bg_color:
                obj_flag = False
                y_end = y
                if (y_end - y_start) >= height*0.52:
                    img[y_start:y_end,x] = bg_color
                    break
    return img


def get_outer_box(img_gray):
    init_val = img_gray[0,0]
    height, width = img_gray.shape
    min_x = width - 1
    min_y = height - 1
    max_x = 0
    max_y = 0
    for i in range(height):
        for j in range(width):
            if img_gray[i,j] != init_val:
                if min_x > j: min_x = j
                break
    for i in range(height):
        for j in range(width):
            if img_gray[i, width - 1 - j] != init_val:
                if max_x < width - 1 - j: max_x = width - 1 - j
                break
    for i in range(width):
        for j in range(height):
            if img_gray[j,i] != init_val:
                if min_y > j: min_y = j
                break
    for i in range(width):
        for j in range(height):
            if img_gray[height - 1 - j, i] != init_val:
                if max_y < height - 1 - j: max_y = height - 1 - j
                break
    return min_x, min_y, max_x, max_y

def preprocessing(img_gray):
    height, width  = img_gray.shape

    # noise erase
    """
    npa = np.asarray(img_gray)
    max_val = npa.max()
    min_val = npa.min()
    for y in range(0, height):
        for x in range(0, width):
            if img_gray[y, x] > (min_val+100):
                img_gray[y, x] = max_val
    """

    #blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret1, img_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    """
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    cv2.imwrite('blur.png', blur)
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(blur, kernel, iterations=1)
    erosion = cv2.erode(blur, kernel, iterations=1)
    morph_gradient = dilation - erosion
    cv2.imwrite('morph_gradient.png', morph_gradient)
    thresh = cv2.adaptiveThreshold(morph_gradient, 255, 1, 1, 11, 2)
    cv2.imwrite('adaptivethresh.png', thresh)
    kernel = np.ones((2, 2), np.uint8)  # 9x5 글자를 찾을때는 가로가 긴 커널
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('closing.png', thresh)
    """
    return img_gray

def get_obj_region_in_Xaxis(img_gray):
    height, width = img_gray.shape
    histo = []
    for x in range(0, width):
        temp = 0
        for y in range(0, height):
            temp += img_gray[y, x]
        histo.append(temp/IMG_SOURCE_HEIGHT)
    init_val = histo[0]
    obj_area = []
    obj_area_flag = False
    for idx, val in enumerate(histo):
        if idx == 0: continue
        if obj_area_flag == False and init_val != val:
            obj_area_flag = True
            start_point = idx
        elif obj_area_flag == True and init_val == val:
            obj_area_flag = False
            end_point = idx-1
            obj_area.append((start_point, end_point))
    return obj_area

def get_obj_region_in_Yaxis(img_gray):
    height, width = img_gray.shape
    histo = []
    for y in range(0, height):
        temp = 0
        for x in range(0, width):
            temp += img_gray[y, x]
        histo.append(temp/IMG_SOURCE_WIDTH)
    init_val = histo[0]
    obj_area = []
    obj_area_flag = False
    for idx, val in enumerate(histo):
        if idx == 0: continue
        if obj_area_flag == False and init_val != val:
            obj_area_flag = True
            start_point = idx
        elif obj_area_flag == True and init_val == val:
            obj_area_flag = False
            end_point = idx-1
            obj_area.append((start_point, end_point))
    return obj_area

def get_hor_sepatation_line(img_gray):
    height, width = img_gray.shape
    histo = []
    for y in range(0, height):
        temp = 0
        for x in range(0, width):
            temp += img_gray[y, x]
        histo.append(temp/IMG_SOURCE_WIDTH)
    init_val = histo[0]
    ready_flag = False
    div_pos = []
    start_point = 0
    end_point = IMG_SOURCE_WIDTH
    for idx, val in enumerate(histo):
        if ready_flag == False and init_val != val:
            ready_flag = True
            if start_point > 0:
                end_point = idx
                div_pos.append(int(start_point+end_point)//2)
        if ready_flag == True and init_val == val:
            start_point = idx
            ready_flag = False
    return div_pos

def get_ver_sepatation_line(img_gray):
    height, width = img_gray.shape
    histo = []
    for x in range(0, width):
        temp = 0
        for y in range(0, height):
            temp += img_gray[y, x]
        histo.append(temp/IMG_SOURCE_HEIGHT)
    init_val = histo[0]
    ready_flag = False
    div_pos = []
    start_point = 0
    end_point = IMG_SOURCE_WIDTH
    for idx, val in enumerate(histo):
        if ready_flag == False and init_val != val:
            ready_flag = True
            if start_point > 0:
                end_point = idx
                div_pos.append(int(start_point+end_point)//2)
        if ready_flag == True and init_val == val:
            start_point = idx
            ready_flag = False
    return div_pos

import re
import time

def isKorean(strOCRrslts):
    result = False
    for cur_str in strOCRresults:
        hanCount = len(re.findall(u'[\u3130-\u318F\uAC00-\uD7A3]+', cur_str))
        if hanCount > 0:
            result = True
            break
    return result

from PIL import Image
import pyocr
import pyocr.builders
import shutil
tools = pyocr.get_available_tools()
if len(tools) == 0:
    print("No OCR tool found")
    sys.exit(1)
# The tools are returned in the recommended order of usage
tool = tools[0]
print(tool)
cur_builder = pyocr.builders.TextBuilder( tesseract_layout = 6 )    # 6 - single uniform block of text, 7 - single text line

def lineOCR(img_gray):
    obj_areas = get_obj_region_in_Yaxis(img_gray)
    #print(obj_areas)
    height, width = img_gray.shape
    #print(height,width)
    strOCRresults = []
    for y_start,y_end in obj_areas:
        roi = img_gray[y_start:y_end+1, 0:width]
        bg_color = int(roi[0, 0])
        roi = cv2.copyMakeBorder(roi, 10, 10, 0, 0, cv2.BORDER_CONSTANT, value=bg_color)  # top, bottom, left, right padding
        pil_image = Image.fromarray(np.uint8(roi))  # convert numpy array to PIL image
        strOCRrslt = tool.image_to_string(pil_image, lang='kor+eng', builder = cur_builder)
        strOCRresults.append(strOCRrslt)
        print(strOCRrslt)
        cv2.imwrite('current_filteredimg.png', roi)
        input()
    return strOCRresults

##########################################################################################
# main
#
import pickle

directory = os.listdir(test_image_dir)
idx = 0
for filename in directory:
    if filename[-4:] != '.jpg':
        del directory[idx]
    idx += 1

print(len(directory), ' files found')

korChar_image_lists = []
forChar_image_lists = []
idx = 0
for filename in directory:
    print(idx, filename)
    idx += 1
    if filename is None:
        print("\nerror: image not read from file \n\n")
        os.system("pause")
        sys.exit()
    #filename = 'out_4020180012325.jpg'
    img = cv2.imread(test_image_dir + '/' + filename, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('current_img.png', img)

    # 1st step : check Korean char in the whole image
    pil_image = Image.fromarray(np.uint8(img))  # convert numpy array to PIL image
    #print('ready..')
    #input()
    strOCRrslt = tool.image_to_string(pil_image, lang='kor+eng', builder = cur_builder)
    strOCRresults = []
    strOCRresults.append(strOCRrslt)
    print(strOCRrslt)
    cv2.imwrite('current_filteredimg.png', img)
    input()
    if isKorean(strOCRresults) == True: continue

    # 2nd step : check it in the divided image
    img_gray = preprocessing(img)
    img_gray = removeVerticalLines(img_gray)
    x0, y0, x1, y1 = get_outer_box(img_gray)
    #print(x0,y0,x1,y1)
    roi = img_gray[y0:y1+1, x0:x1+1]   # get ROI
    bg_color = int(roi[0,0])
    img_gray = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=bg_color) # top, bottom, left, right padding with 10

    strOCRresults = lineOCR(img_gray)

    no_korean_flag = False
    if isKorean(strOCRresults) == False:
        obj_areas = get_obj_region_in_Xaxis(img_gray)
        height, width = img_gray.shape
        if len(obj_areas) >= 2:
            #print('cut the left block and retry')
            roi = img_gray[0:height, obj_areas[1][0]:obj_areas[len(obj_areas)-1][1]+1]    # cut left block
            bg_color = int(roi[0,0])
            roi = cv2.copyMakeBorder(roi, 0, 0, 10, 0, cv2.BORDER_CONSTANT,
                                          value=bg_color)  # top, bottom, left, right padding with 10

            strOCRresults = lineOCR(roi)

            if isKorean(strOCRresults) == False:
                #print('cut the right block and retry')
                obj_areas = get_obj_region_in_Xaxis(img_gray)
                height, width = img_gray.shape
                roi = img_gray[0:height, obj_areas[0][0]:obj_areas[len(obj_areas) - 2][1] + 1]  # cut right block
                bg_color = int(roi[0, 0])
                roi = cv2.copyMakeBorder(roi, 0, 0, 0, 10, cv2.BORDER_CONSTANT,
                                              value=bg_color)  # top, bottom, left, right padding with 10

                strOCRresults = lineOCR(roi)

                if isKorean(strOCRresults) == False:
                    #print('No korean charactor detected')
                    no_korean_flag = True
        else:
            #print('No korean charactor detected')
            no_korean_flag = True

    # split test images to kor and non-kor
    if no_korean_flag == True:
        #shutil.copy(test_image_dir + '/' + filename, forChar_image_dir + '/' + filename)
        forChar_image_lists.append(filename)
    else:
        #shutil.copy(test_image_dir + '/' + filename, korChar_image_dir + '/' + filename)
        korChar_image_lists.append(filename)

    #cv2.imwrite(test_image_dir + '/' + filename[0:-3]+'png', img_gray)

fi = open('output.dat', 'wb')
pickle.dump(korChar_image_lists, fi)
pickle.dump(forChar_image_lists, fi)
fi.close()
