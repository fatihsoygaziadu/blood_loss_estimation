# This module controls ML operations such as loads required data, executes Ml algorithms and executes
# cross validated ML algorithms
# should contain just controlling duties (calling other functions) not the function itself
import time
import IO as iop
import Process as proc
import Utility as iu
from Algorithms import ImageConfig
from Algorithms import MyANN
from Logger import log_print
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from dbn.models import SupervisedDBNClassification
from sklearn.base import BaseEstimator
from sklearn.multiclass import OneVsRestClassifier

# Calls related functions to execute ML algos
# it gets bin size (we don't want to import config file to all modules),
# train ratio (we don't want to import config file to all modules)
# name of classifier and classifier instance (config file calls classifier's constructor therefore
# we will have and instance e.g. instance of SVC() constructor)

# all ml controllers executes related ml process and calculates rmse and accurayc and prints result to file and log


# this function controls the flags and runs the ml algoritms controllers
def run_controller(one, ovr, grid, cross, nn, conv, dbn , algos, train_ratio, bin_size, k_fold, njobs, epochs, batch_size):
    f = True

    if one:
        for clf in algos:
            # clf.one_classifier.set_params(n_jobs=njobs)
            one_clf_controller(clf, f, train_ratio, bin_size)
            f = False
    if ovr:
        for clf in algos:
            ovr_clf_controller(clf, f, train_ratio, bin_size)
            f = False
    if cross:
        for clf in algos:
            cross_val_controller(clf, f, train_ratio, bin_size, k_fold)
            f = False
    if nn:
        for clf in algos:
            if isinstance(clf, MyANN):
                # clf.one_classifier.set_params(n_jobs=njobs)
                classical_ann_controller(clf, f, train_ratio, bin_size, epochs, batch_size)

    if conv:
        for clf in algos:
            # clf.one_classifier.set_params(n_jobs=njobs)
            cnn_controller()
    if dbn:
        for clf in algos:
            if isinstance(clf, SupervisedDBNClassification):
                dbn_controller(clf, train_ratio, bin_size)

    if grid:
        for clf in algos:
            if isinstance(clf, GridSearchCV):
                grid_search_controller(clf, f, train_ratio, bin_size, k_fold)
                f = False



# this controller for a algorithm with one set of hyper parameters and thats it
def one_clf_controller(clf, f, train_ratio, bin_size):
    if f:
        iop.write_image_fields(ImageConfig.field_names_str)

    if bin_size == 0:
        ImageConfig.bin_size = bin_size = clf.best_bin_size

    if train_ratio == 0:
        ImageConfig.train_ratio = train_ratio = clf.best_train_ratio

    log_print('******************** Starting to a New Epoch **********************************************************')
    log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)

    # setting classifier string to make more sense when we are looking to log files or image report file
    classifier_str = clf.classifier_str + "_one"
    # Writing data type, bin size, and train ratio into final image report file
    iu.append_image_report(
        ImageConfig.folder_data_type.ljust(40) + str(bin_size).ljust(40) + str(train_ratio).ljust(40))
    # loading fractional data
    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)

    print(np_process_ds.test_data.shape, np_process_ds.train_data.shape)

    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    # executing ML algorithm
    model, pred_test_label, elapsed_time = proc.execute_one_clf(np_process_ds, clf.one_classifier)
    # calculating rmse and accuracy
    pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)
    # calling print function to writing log
    time.sleep(2)
    proc.result_print(pred_test_label, np_process_ds.test_label, pred_acc, pred_rmse, elapsed_time, model,
                      classifier_str, lab_enc)
    # Final image report writes
    iop.print_image_report(ImageConfig.one_line_report_str)


def ovr_clf_controller(clf, f, train_ratio, bin_size):
    if f:
        iop.write_image_fields(ImageConfig.field_names_str)

    if bin_size == 0:
        ImageConfig.bin_size = bin_size = clf.best_bin_size

    if train_ratio == 0:
        ImageConfig.train_ratio = train_ratio = clf.best_train_ratio

    log_print('******************** Starting to a New Epoch **********************************************************')
    log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)

    # setting classifier string to make more sense when we are looking to log files or image report file
    classifier_str = clf.classifier_str + "_ovr"
    # Writing data type, bin size, and train ratio into final image report file
    iu.append_image_report(
        ImageConfig.folder_data_type.ljust(40) + str(bin_size).ljust(40) + str(train_ratio).ljust(40))
    # loading fractional data
    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)

    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    # executing ML algorithm
    model, pred_test_label, elapsed_time = proc.execute_one_clf(np_process_ds, clf.ovr_classifier)
    # calculating rmse and accuracy
    pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)
    # calling print function to writing log
    proc.result_print(pred_test_label, np_process_ds.test_label, pred_acc, pred_rmse, elapsed_time, model,
                      classifier_str, lab_enc)
    # Final image report writes
    iop.print_image_report(ImageConfig.one_line_report_str)


# this controller for a algorithm with one set of hyper parameter and its executes fitting with cross valdiation
# k folds can be controller by user
def cross_val_controller(clf, f, train_ratio, bin_size, k_fold):
    if f:
        iop.write_image_fields(ImageConfig.field_names_str)

    if bin_size == 0:
        ImageConfig.bin_size = bin_size = clf.best_bin_size

    if train_ratio == 0:
        ImageConfig.train_ratio = train_ratio = clf.best_train_ratio

    log_print('******************** Starting to a New Epoch **********************************************************')
    log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)

    classifier_str = clf.classifier_str + "_cross"
    # line sends str regarding data type, bin size and
    # train ratio by using append_image_report function of Image_Util module

    iu.append_image_report(
        ImageConfig.folder_data_type.ljust(40) + str(bin_size).ljust(40) + str(train_ratio).ljust(40))

    # (same as above) calls load_all_np_arrays function in IO_Processor. Gets bin_size and train_ratio
    # we sending these arguments becouse numpy files stored with their bin size and train ratio
    # np_process_ds not linke to anythink it implicitly get type of function return and this is our strycture (process_data)
    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)

    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    model, pred_test_label, accuracy, rmse, elapsed_time = proc.execute_cross_val(np_process_ds, clf.one_classifier,
                                                                                  k_fold)
    # pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)
    proc.result_print(pred_test_label, np_process_ds.test_label, accuracy, rmse, elapsed_time, model,
                      classifier_str, lab_enc)

    iop.print_image_report(ImageConfig.one_line_report_str)


# k folds can be controller by user
# below method makes grid search and finds best parameter set from the algoritm grid_classifier
def grid_search_controller(clf, f, train_ratio, bin_size, k_fold):
    if f:
        iop.write_image_fields(ImageConfig.field_names_str)

    if bin_size == 0:
        ImageConfig.bin_size = bin_size = clf.best_bin_size

    if train_ratio == 0:
        ImageConfig.train_ratio = train_ratio = clf.best_train_ratio

    log_print('******************** Starting to a New Epoch **********************************************************')
    log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)

    classifier_str = clf.classifier_str + "_grid"
    # line sends str regarding data type, bin size and
    # train ratio by using append_image_report function of Image_Util module

    iu.append_image_report(
        ImageConfig.folder_data_type.ljust(40) + str(bin_size).ljust(40) + str(train_ratio).ljust(40))
    clf.grid_classifier.cv = k_fold
    clf.grid_classifier.n_jobs = -1
    # classifier_str = classifier_str + "_cv_" + str(k)
    # (same as above) calls load_all_np_arrays function in IO_Processor. Gets bin_size and train_ratio
    # we sending these arguments becouse numpy files stored with their bin size and train ratio
    # np_process_ds not linke to anythink it implicitly get type of function return and this is our strycture (process_data)
    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)
    # print(np_process_ds.train_label)
    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    model, pred_test_label, elapsed_time = proc.execute_grid_search(np_process_ds, clf.grid_classifier)
    pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)

    proc.result_print(pred_test_label, np_process_ds.test_label, pred_acc, pred_rmse, elapsed_time, model,
                      classifier_str, lab_enc)
    iop.print_image_report(ImageConfig.one_line_report_str)


def classical_ann_controller(clf, f, train_ratio, bin_size, epochs, batch_size):
    if train_ratio == 0.6 and bin_size == 1 and epochs == 12 and batch_size == 7:
        iop.write_image_fields(ImageConfig.field_names_for_ann_str)

    if bin_size == 0:
        ImageConfig.bin_size = bin_size = clf.best_bin_size

    if train_ratio == 0:
        ImageConfig.train_ratio = train_ratio = clf.best_train_ratio

    log_print('******************** Starting to a New Epoch **********************************************************')
    log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)

    # setting classifier string to make more sense when we are looking to log files or image report file
    classifier_str = clf.classifier_str + "_e"+str(epochs)+"_b"+str(batch_size)
    # Writing data type, bin size, and train ratio into final image report file
    iu.append_image_report(
        ImageConfig.folder_data_type.ljust(40) + str(bin_size).ljust(40) + str(train_ratio).ljust(40))
    # loading fractional data
    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)

    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    # executing ML algorithm
    model, pred_test_label, elapsed_time = proc.execute_classical_ann(np_process_ds, clf.one_classifier, epochs, batch_size)

    #iop.save_model(model)

    # calculating rmse and accuracy

    pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)
    # calling print function to writing log

    proc.result_print(pred_test_label, np_process_ds.test_label, pred_acc, pred_rmse, elapsed_time, model,
                      classifier_str, lab_enc)
    iu.append_image_report(str(clf.node_numbers).ljust(40)+str(clf.layer_number).ljust(40)+str(clf.activations).ljust(40)+str(epochs).ljust(40)+str(batch_size).ljust(40))
    # Final image report writes
    iop.print_image_report(ImageConfig.one_line_report_str)


def cnn_controller():
    # np_process_ds = iop.load_np_arrays()
    # lab_enc = proc.label_encoder(np_process_ds)
    proc.execute_cnn()


def dbn_controller(clf, train_ratio, bin_size):
    log_print('******************** Starting to a New Epoch **********************************************************')
    log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)

    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)

    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    # executing ML algorithm
    model, pred_test_label, elapsed_time = proc.execute_dbn(np_process_ds, clf)

    # iop.save_model(model)

    # calculating rmse and accuracy

    pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)
    # calling print function to writing log
    proc.result_print(pred_test_label, np_process_ds.test_label, pred_acc, pred_rmse, elapsed_time, model, "dbn", lab_enc)



def grid_search_cv_controller(bin_size, train_ratio, classifier_str, classifier, njobs):
    # line sends str regarding data type, bin size and
    # train ratio by using append_image_report function of Image_Util module

    iu.append_image_report(
        ImageConfig.folder_data_type.ljust(40) + str(bin_size).ljust(40) + str(train_ratio).ljust(40))
    classifier.cv = 6
    classifier.n_jobs = njobs
    classifier_str = classifier_str + "_" + str(classifier.cv) + "_thread_" + str(njobs)
    # (same as above) calls load_all_np_arrays function in IO_Processor. Gets bin_size and train_ratio
    # we sending these arguments becouse numpy files stored with their bin size and train ratio
    # np_process_ds not linke to anythink it implicitly get type of function return and this is our strycture (process_data)
    np_process_ds = iop.load_all_np_arrays(bin_size, train_ratio)
    # print(np_process_ds.train_label)
    # Encoding fractional labels
    lab_enc = proc.label_encoder(np_process_ds)

    model, pred_test_label, elapsed_time = proc.execute_grid_search(np_process_ds, classifier)
    pred_rmse, pred_acc = proc.calculate_rmse(pred_test_label, np_process_ds.test_label, lab_enc)

    proc.result_print(pred_acc, pred_rmse, elapsed_time, classifier_str)
    iop.print_image_report(ImageConfig.one_line_report_str)
