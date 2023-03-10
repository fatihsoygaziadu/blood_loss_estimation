import Controller as ctrl
from Algorithms import *
from ETL import ETL_loader
from image_processing import *
from Logger import log_print
from dbn import SupervisedDBNClassification
import Config
from IO import extract_new_features
import numpy as np

np.random.seed(1337)

# Plot settings. plot_top_n is the number of algorithms
plot_top_n = 10
plot_criteria = 'Accuracy'
# for choosing the train ratio. it means train_ration*10 sample in train set
# when it stays 0 the best ones used by each algorithm fetched from config
# choose [0.6, 0.7, 0.8]
train_ratio = 0.7

# choose [1, 8, 16, 32, 64, 256]
# when it stays 0 the best ones used by each algorithm fetched from config
bin_size = 8


# For cross validation or grid search the cv parameter
# it must be appropriate for train ratio
k_fold = 2
njobs = -1
batch_size = 6

epochs = 1000

etl = False
# execution of a classifier with one set of hyper parameters without cross validation, choose ..._one_classifier
one_classifier = False
# execution of a classifier with one set of hyper parameters with one vs rest classifier method without cross validation, choose ..._one_classifier
one_versus_rest = False
# execution of a classifier with one set of hyper parameters and cross validation of data, choose ..._one_classifier
cross_validation = False
# execution of a classifier with gridsearchcv library and finds best hyperparameters amoung many sets of hyper parameters with cross validation, choose ..._grid_classifier
grid_search = False
#
neural_networks = True
#
conv_networks = False
#
deep_belief_nets = False

# Usable algorithms
svc = MySVC()
rfc = MyRFC()
knc = MyKNC()
bnb = MyBNB()
gnb = MyGNB()
abc = MyABC()
dt = MyDT()
sgdc = MySGDC()
gbc = MyGBC()
logr = MyLogR()
xgbc = MyXGBC()

# ANN Parameters and ann model object instance
layer_number = 5
node_numbers = [10, 64, 64, 64, 12]
activation_functions = ['relu', 'relu','tanh', 'relu', 'softmax']
input_dim = bin_size+2
optimizer = 'adam'
init_mode = 'uniform'
ann = MyANN(input_dim=input_dim, optimizer=optimizer, layer_number=layer_number, node_numbers=node_numbers,
            activations=activation_functions, init_mode=init_mode)

# DBN Parameters and dbn model object instance
dbn = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                  learning_rate_rbm=0.0002,
                                  learning_rate=0.2,
                                  n_epochs_rbm=10,
                                  n_iter_backprop=100,
                                  batch_size=12,
                                  activation_function='sigmoid',
                                  dropout_p=0.002)

# selected algorithms to run
algorithms = [ann]


# SETS UP THE SYSTEM TO RUN ACCORDING TO PREDEFINED OPTIONS
ctrl.run_controller(one_classifier, one_versus_rest, grid_search, cross_validation, neural_networks, conv_networks,
                    deep_belief_nets,
                    algorithms, train_ratio, bin_size, k_fold, njobs, epochs, batch_size)




# ML_for_all()

# etl_for_all()


# for bin_size in [1, 8, 16,32, 64, 256]:
#     for train_ratio in [0.6, 0.7, 0.8]:
#         extract_new_features(bin_size, train_ratio)


# bar char visualization method. it takes a data frame
# algorithm_result_bar_chart(read_txt_to_dataframe("Final_Image_Report_2019-07-09_16-36-07.txt", plot_top_n, plot_criteria))


def ML_for_all():
    # for ImageConfig.train_ratio in [0.7]:
    # for ImageConfig.bin_size in [16]:
    for train_ratio in [0.6, 0.7, 0.8]:
        for bin_size in [1, 8, 16, 32, 64, 256]:
            for epochs in [12, 24, 100]:
                for batch_size in [7, 14]:
                    for layer_number in [1, 2, 3]:
                        for node_numbers in [[10, 40, 12], [10, 80, 12], [10, 120, 12]]:
                            for activation_functions in [['relu', 'sigmoid', 'softmax'], ['relu', 'tanh', 'softmax'],
                                                         ['sigmoid', 'tanh', 'softmax']]:
                                log_print('******************** Starting to a New Epoch ********************** ')
                                log_print('Bin size: ', bin_size, ' Train ratio: ', train_ratio)
                                # This line is for logging prints current bin size and train ratio
                                # print('Bin size:',ImageConfig.bin_size,'Train ratio:', ImageConfig.train_ratio)
                                # controller calls apropriate clasifier with bin size, ratio, clasifier string for name of clasifier,
                                # clasifier is hyperparameters
                                # ml_numpy_controller(ImageConfig.bin_size,ImageConfig.train_ratio,classifier_str,classifier)
                                ann1 = MyANN(input_dim=input_dim, optimizer=optimizer, layer_number=layer_number, node_numbers=node_numbers,
                                             activations=activation_functions, init_mode=init_mode)
                                algorithms = [ann1]
                                ctrl.run_controller(one_classifier, one_versus_rest, grid_search, cross_validation,
                                                    neural_networks, conv_networks, algorithms, train_ratio, bin_size,
                                                    k_fold, njobs, epochs, batch_size)


def etl_for_all():
    # for ImageConfig.train_ratio in np.arange(0.6,0.81,0.1):
    # this is for train-test ratio
    for ImageConfig.train_ratio in [0.6, 0.7, 0.8]:
        # for ImageConfig.train_ratio in [0.6]:
        # for ImageConfig.bin_size in [16]:
        # this is for bin sizes
        for ImageConfig.bin_size in [1, 8, 16, 32, 64, 256]:
            log_print('Bin size: ', ImageConfig.bin_size, ' Train ratio: ', ImageConfig.train_ratio)
            # print('Bin size:', ImageConfig.bin_size, 'Train ratio:', ImageConfig.train_ratio)
            ETL_loader(ImageConfig.root_folder, ImageConfig.f_matrix_txt)



# load_all_image_tensors('RawImages/')
#
#
# images = load_images_from_folder2('GrayScaledImages')
# labels = np.load('GrayScaledImages/all_labels.npy')
#
# print(len(labels), len(images))


# executes model without cross validation
# Process.execute_one_model(RFCConfig.rfc_one_classifier, data, lab_enc)


# executes model with cross validation
# Process.execute_cross_val(RFCConfig.rfc_one_classifier, data, 3, lab_enc)


# execute a model with one vs rest classifier and cross validation
# Process.one_vs_rest(data, RFCConfig.rfc_one_classifier, lab_enc, 3)


# Statistical method for each feature's lower and upper bound to detect outliers
# Process.detect_outlier(data.train_data)


# Execute model with Grid Search Cv
# Process.execute_grid_search_cv(SVCConfig.svc_grid_classifier, data, lab_enc, True, 12)


# creating box plot of our train data
# Process.create_box_plot(data.train_data)
# Process.detect_outlier(data.train_data)

# Process.plot_data(data)
