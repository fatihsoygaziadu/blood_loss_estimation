import os
import cv2
import numpy as np
from Algorithms import ImageConfig
from sklearn.externals import joblib
from ETL import save_numpy
import Data as d
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def load_np_arrays():
    train_np_array = np.load(ImageConfig.input_data_folder+"Train_data.npy")
    test_np_array = np.load(ImageConfig.input_data_folder + "Test_data.npy")
    train_label_np_array = np.load(ImageConfig.input_data_folder+"Train_label.npy")
    test_label_np_array = np.load(ImageConfig.input_data_folder+"Test_label.npy")

    np_process_ds = d.np_process_data(train_np_array, test_np_array, train_label_np_array, test_label_np_array)

    return np_process_ds


def load_all_np_arrays(bin_size, train_ratio):
    train_np_array = np.load(ImageConfig.input_data_folder + 'Train_data_'+str(bin_size)+
                           '_train'+str(int(train_ratio*100))+'_'+ImageConfig.saved_ts+'.npy')

    test_np_array = np.load(ImageConfig.input_data_folder + 'Test_data_'+str(bin_size)+
                          '_train'+str(int(train_ratio*100))+'_'+ImageConfig.saved_ts+'.npy')
    train_label_np_array = np.load(ImageConfig.input_data_folder + 'Train_label_'+str(bin_size)+
                                 '_train'+str(int(train_ratio*100))+'_'+ImageConfig.saved_ts+'.npy')
    test_label_np_array = np.load(ImageConfig.input_data_folder + 'Test_label_'+str(bin_size)+
                                '_train'+str(int(train_ratio*100))+'_'+ImageConfig.saved_ts+'.npy')

    np_process_ds = d.np_process_data(train_np_array, test_np_array, train_label_np_array, test_label_np_array)
    return np_process_ds


def save_model(classifier):
  # model_file=open(ImageConfig.model_file_name,'a+')
  # print(classifier)
  model_file = open(ImageConfig.model_file_name, 'a+b')
  joblib.dump(classifier,model_file)
  model_file.close()


def load_model():
  model_file = open(ImageConfig.model_file_name, 'rb')
  model_details=joblib.load(model_file)
  model_file.close()
  return model_details


def write_image_fields(string):
  print_image_report(string)


def print_image_report(one_line_report_str):
    file = open("Reports\\Final_Image_Report_" + ImageConfig.ts + ".txt", 'a')
    file.write(one_line_report_str + '\n')
    # file.write('\n')
    file.flush()
    file.close()
    ImageConfig.one_line_report_str=''


def read_txt_to_dataframe(file_name, top_n, criteria):
    # this method reads a txt file into pandas data frame then  its sorts the data according to given criteria and takes top_n
    # row to new dataframe returns that data frame
    df = pd.read_csv(file_name, delimiter='\s+', index_col=False)
    df = df.sort_values(by=criteria, ascending=False)
    df = df.head(top_n)
    return df


def algorithm_result_bar_chart(df):
    n_groups = len(df)
    rmse = df['RMSE']
    accuracy = df['Accuracy']

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = .35
    opacity = 0.8

    rects1 = plt.bar(index, rmse, bar_width,
                     alpha=opacity,
                     color='b',
                     label='RMSE')

    rects2 = plt.bar(index + bar_width, accuracy, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Accuracy')

    plt.xlabel('Algorithms')
    plt.ylabel('Results')
    plt.title('Best 10 Results according to Accuracy')
    plt.xticks(index + bar_width, df['Algorithm'])
    plt.legend()

    plt.tight_layout()
    plt.show()


def extract_new_features(bin_size, train_ratio):
    np_ds = load_all_np_arrays(bin_size, train_ratio)
    # adding sum of the red pixels number as a feature column
    # np_ds.train_data = sum_of_red_pixels(np_ds.train_data)
    # np_ds.test_data = sum_of_red_pixels(np_ds.test_data)

    # adding average color of the red pixels number as a feature column
    # print("Original:", np_ds.train_data[0])
    np_ds.train_data = avg_of_red_pixels(np_ds.train_data, bin_size)
    np_ds.test_data = avg_of_red_pixels(np_ds.test_data, bin_size)
    # print("Modified:", np_ds.train_data[0])

    save_numpy(np_ds, bin_size, ImageConfig.ts, train_ratio)


def sum_of_red_pixels(data_set):
    # adding sum of the red pixels number as a feature column
    temp = data_set.sum(axis=1)
    data_set = np.concatenate((data_set, temp[:, None]), axis=1)
    return data_set


def avg_of_red_pixels(org_data_set, bin_size):
    data_set = org_data_set.copy()
    # adding average of the red pixel number as a feature column
    coef = 256 / bin_size
    coef_addition = 256 / bin_size
    # print("Train:", data_set[0], "\n")
    # print("Test:", data_set[0], "\n\n\n\n")
    sum = 0
    average_red_color_arr = []
    i = 0
    for i in range(bin_size):
        # print("i={}, coef={}".format(i, coef))
        data_set[:, i] = data_set[:, i] * coef
        coef += coef_addition
    i = 0
    # print("\n\n\n\n")
    for i in range(len(data_set)):
        sum = data_set[i, :-1].sum()
        avg_red_color = sum / data_set[i][-1]
        # print("{}.image avgcolor ={}".format(i,avg_red_color))
        average_red_color_arr.append(avg_red_color)

    np_avg_red_color = np.array(average_red_color_arr)
    a = np.concatenate((org_data_set, np_avg_red_color[:, None]), axis=1)
    return a


def save_plot(objects, scores, yaxisname, title, save_file_name):

    import matplotlib.pyplot as plt; plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    objects = objects

    y_pos = np.arange(len(objects))
    performance = scores

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(yaxisname)
    plt.title(title)

    plt.savefig(save_file_name+'.png')


