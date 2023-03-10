import os
import image_processing
# import image_processing_backup
from Algorithms import ImageConfig
# if ratio is 0.7 means get %70 for tarining %30 of data goes for test
import numpy as np
from Data import list_process_data, np_process_data


def ETL_loader(folder, f_matrix_txt):
    list_instance = list_process_data([], [], [], [])

    print(f_matrix_txt)
    print(type(f_matrix_txt))

    print('Before:', ImageConfig.bin_size)
    image_processing.folder_fetcher(folder, f_matrix_txt,
                                    ImageConfig.bin_size, ImageConfig.train_ratio, list_instance)

    # image_processing_backup.folder_fetcher(folder, f_matrix_txt,
    #                                   ImageConfig.bin_size, ImageConfig.train_ratio, list_instance)

    npd_instance = transform_list_to_np_array(list_instance)

    save_numpy(npd_instance, ImageConfig.bin_size, ImageConfig.ts, ImageConfig.train_ratio)


def transform_list_to_np_array(list_instance):
    train_np_array = np.array(list_instance.train_np_list)
    train_label_np_array = np.array(list_instance.train_label_list)
    test_np_array = np.array(list_instance.test_np_list)
    test_label_np_array = np.array(list_instance.test_label_list)
    npd_instance = np_process_data(train_np_array, test_np_array, train_label_np_array, test_label_np_array)
    return npd_instance


def save_numpy(npd_instance, bin_size, ts, train_ratio):

    np.save('Train_data_'+str(bin_size)+'_train'+str(int(train_ratio*100))+'_'+ts+'.npy', npd_instance.train_data)
    np.save('Train_label_'+str(bin_size)+'_train'+str(int(train_ratio*100))+'_'+ts+'.npy', npd_instance.train_label)
    np.save('Test_data_'+str(bin_size)+'_train'+str(int(train_ratio*100))+'_'+ts+'.npy', npd_instance.test_data)
    np.save('Test_label_'+str(bin_size)+'_train'+str(int(train_ratio*100))+'_'+ts+'.npy', npd_instance.test_label)