#this will fetches images and calls
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from Algorithms import ImageConfig
import cv2


# loads images in a subfolder and returns image list
def load_images_from_folder(folder, d):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            labels.append(d)
    return images, labels


def load_images_from_folder2(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# Fetches images in the each subfolder (folders in the root) one by one and stors them in a list
def folder_fetcher(folder, analysis_doc_file, bin_size, train_ratio, list_instance):
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            print(root + d)
            images = load_images_from_folder(root + d)
            image_fetcher(images,d,analysis_doc_file,bin_size,train_ratio, list_instance)
            print('After:', ImageConfig.bin_size)


# convert BGR ? to HSV !!?
def convert_color_code(image, conv_code):
    cvt = cv2.cvtColor(image, conv_code)
    return cvt


# Selecting red pixcels from an image returning an image with red pixcells with intencityb dump
def masking4red(cvt, image):
    # !!? lower mask (0-10)
    lower_red = np.array([0, 120, 70])       # i think it should be [0, 120, 70]
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(cvt, lower_red, upper_red)

    # !!? upper mask (170-180)
    lower_red = np.array([170, 120, 70])     # should be [170, 120, 70]
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(cvt, lower_red, upper_red)

    # !!? join my masks
    mask = mask0 + mask1

    # !!? set my output img to zero everywhere escikit-learn jublixcept my mask
    output_image = image.copy()
    output_image[np.where(mask == 0)] = 0

    return output_image


def histogram_equalization(gray_image):
    equ_gray_image = cv2.equalizeHist(gray_image)
    return equ_gray_image


# this will gets the gray pixcels form image discards the black ones
def extract_target_pixels(equ_gray_image):
    row = equ_gray_image.shape[0]
    column = equ_gray_image.shape[1]

    equ_gray_image_nonzeros = []

    for row_val in range(row):
        for col_val in range(column):
            if equ_gray_image[row_val, col_val] != 0:
                equ_gray_image_nonzeros.append(equ_gray_image[row_val, col_val])

    return equ_gray_image_nonzeros


def f_matrix_histogram_generator(equ_gray_img_nonzeros,bin_param):
    featureM, bins, patches = plt.hist(equ_gray_img_nonzeros, bins=bin_param)
    return featureM


def histogram(equ_gray_img_nonzeros,bin_param):
    featureM = cv2.calcHist([equ_gray_img_nonzeros],[0],None,[bin_param],[0,256])
    #print("feture : ", featureM.flatten() , type(featureM))
    return featureM.flatten()


def np_array_generator(featureM,train_test_ratio,train_ratio,d,list_instance):
    if train_test_ratio <= train_ratio:
        # train_np_array=np.append([train_np_array], [FeatureM_16])
        list_instance.train_np_list.append(featureM)
        list_instance.train_label_list.append(float(d))
    else:
        list_instance.test_np_list.append(featureM)
        list_instance.test_label_list.append(float(d))


def generate_txt(d, analysis_doc_file,featureM):
    print(analysis_doc_file )
    file = open(analysis_doc_file, 'a')

    file.write(d+' ')
    for ind in range(len(featureM)):
        file.write(str(featureM[ind]) + ' ')
    file.write('\n')
    file.flush()
    print('now we printing')
    file.close()


def list_2_nparray (list):
    #np.set_printoptions(threshold=sys.maxsize)
    #nparray = np.empty((0))
    #nparray = np.asarray(list)
    #print('len list', len(np.asarray(list)))
    #print('len array',len(nparray))
    #for i in range(len(list)):
    #    nparray = np.append(nparray, list[i])
    #print('a',type(list),list)
    #print('b', nparray)
    return np.asarray(list)


# Fetches one image and calls perform conversion function.
def image_fetcher(images, d, analysis_doc_file, bin_size, train_ratio, list_instance):
    for i in range(len(images)):
      train_test_ratio = (i + 1) / (len(images))
      image = images[i]
      cvt = convert_color_code(image,cv2.COLOR_BGR2HSV)
      output_image=masking4red(cvt,image)
      gray_image=convert_color_code(output_image,cv2.COLOR_BGR2GRAY)
#      equ_gray_image=histogram_equalization(gray_image)
#      equ_gray_image_nonzeros=extract_target_pixels(equ_gray_image)
      equ_gray_image_nonzeros = extract_target_pixels(gray_image)
      # This craetes feture matrix based on histogram number of pixel ina single bin is a parameter
#      featureM = f_matrix_histogram_generator(equ_gray_image_nonzeros,bin_size)
      featureM = list_2_nparray(equ_gray_image_nonzeros)
      featureM_flattened = histogram(featureM, bin_size)
      print('flattened',featureM_flattened)
      #equ_gray_image_nonzeros_nparray = list_2_nparray(equ_gray_image_nonzeros)
      np_array_generator(featureM_flattened,train_test_ratio,train_ratio,d,list_instance)
      generate_txt(d,analysis_doc_file,featureM_flattened)


def images_to_grayscale(images, labels):
    converted_images = []
    for i in range(len(images)):
        img = images[i]
        cvt = convert_color_code(img, cv2.COLOR_BGR2HSV)
        output_image = masking4red(cvt, img)
        cv2.imwrite('RedMaskedImages/red_masked_image_'+str(i)+'.png', output_image)
        gray_image = convert_color_code(output_image, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('GrayScaledImages/gray_scale_image_'+str(i)+'.png', gray_image)
        converted_images.append(output_image)
    labels = np.array(labels)
    np.save('RedMaskedImages/all_labels.npy', labels)
    np.save('GrayScaledImages/all_labels.npy', labels)

    return converted_images


def load_all_image_tensors(folder):
    imgs = []
    lbls = []
    for root, dirs, files in os.walk(folder):
        for d in dirs:
            print(root + d)
            images, labels = load_images_from_folder(root + d, d)
            imgs += images
            lbls += labels
    img = images_to_grayscale(imgs, lbls)
    return img, lbls


def prep_one_image(img):
    cvt = convert_color_code(img, cv2.COLOR_BGR2HSV)
    output_image = masking4red(cvt, img)
    gray_image = convert_color_code(output_image, cv2.COLOR_BGR2GRAY)
    return gray_image