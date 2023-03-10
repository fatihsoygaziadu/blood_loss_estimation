from sklearn.model_selection import train_test_split
from Algorithms import ImageConfig


class image_report_data:
    def __init__(self, description, bin_size,train_ratio, ML_accuracy, RMSE, mean_score, std_dev, ML_elapsed_time, cross_val_elapsed_time):
        self.description = description
        self.bin_size = bin_size
        self.train_ratio = train_ratio
        self.ML_accuracy = ML_accuracy
        self.RMSE = RMSE
        self.mean_score = mean_score
        self.std_dev = std_dev
        self.ML_elapsed_time = ML_elapsed_time
        self.cross_val_elapsed_time = cross_val_elapsed_time


def data_label_splitter (all_data, all_label, train_ratio):
    data_train, data_test, label_train, label_test = train_test_split(all_data,all_label, test_size = (1-train_ratio))


def append_image_report(report_str):
    ImageConfig.one_line_report_str += report_str
