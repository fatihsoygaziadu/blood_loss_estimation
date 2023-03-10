import numpy as np
import pandas as pd


class list_process_data:
    def __init__(self, train_np_list, test_np_list, train_label_list, test_label_list):
        self.train_np_list = train_np_list
        self.test_np_list = test_np_list
        self.train_label_list = train_label_list
        self.test_label_list = test_label_list

class np_process_data:
    def __init__(self, train_np_array, test_np_array, train_label_np_array, test_label_np_array):
        self.train_data = train_np_array
        self.test_data = test_np_array
        self.train_label = train_label_np_array
        self.test_label = test_label_np_array

        # self.encoded_train_label_np_array = encoded_train_label_np_array
        # self.encoded_test_label_np_array = encoded_test_label_np_array
