from datetime import datetime


class ImageConfig:
    # this is folder that contains all raw data
    # root_folder = 'C:\\Users\\hus\\Desktop\\Feray_Hoca\\no_water\\'
    folder_data_type = 'Water'
    # root_folder_prefix = 'D:\\GitHub\\ImageClassification\\'
    root_folder_prefix = 'C:\\Users\\hus\\PycharmProjects\\DursunImageClassification_Final'

    root_folder = root_folder_prefix + folder_data_type + '\\'

    # this is folder that contains all input (processed) data (contains numpy files
    # 1 train
    # 1 train label
    # 1 test
    # 1 test label file)
    # input_data_folder = "D:\\GitHub\\ImageClassification\\drsn_data_v2\\"
    input_data_folder = "C:\\Users\\hus\\PycharmProjects\\DursunImageClassification_Final\\drsn_data_v2\\"

    # TO DO remove this we dont use this we will save with logging
    model_file_name = "optim_models.txt"

    # this is default value for train ratio
    # if this is 0.6 therefore 60% of raw data goes for training and 40% goes for test
    train_ratio = 0.7
    # this is bin for histogram
    # if it is 1 takes all 256 grayscale color intencities into one bin
    # if it is 8 takes 256/8 = 32 grayscale color intencities into one bin
    # 16 for 16 color, 32 for 8 color, 64 for 4 color and 256 for 1 color (all color intensities - full view)
    bin_size = 8

    # timestamp taken from the host system and appends to every file for each start run,
    # therefore each start saves many processed files with same ts value
    ts = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    # this is required to read (load) all input files (proccessed previously) from the imput folder
    # but each set of numpy files has specific ts, without this appliction produces file not file error
    saved_ts = '2020-04-10_14-00-02'  # '2020-02-21_17-32-49' #'2019-06-06_09-24-55'

    # name of the text file tht contains all proccessed data
    # we want to save all proccessed results in a single txt file as well
    f_matrix_txt = 'Feature_Matrix_' + str(bin_size) + '_train' + str(int(train_ratio * 100)) + '_' + ts + '.txt'

    # this is for k-fold split for cross validation, this rate is used for validation part (train,validation,test in crossval)
    # kf_split must be small-equal to train ratio there is no data more than this ratio
    # we multiply with 10 because we gave a float number to train ratio
    kf_split = int(train_ratio * 10)

    # Excel header string. Change here what ever you want store for Excel investigation
    field_names_str = 'Description'.ljust(40) + 'BinSize'.ljust(40) + 'TrainRatio'.ljust(40) + 'Algorithm'.ljust(40) + \
                      'Accuracy'.ljust(40) + 'RMSE'.ljust(40) + 'ElapsedTime'.ljust(40)

    # Excel header string. Change here what ever you want store for Excel investigation
    field_names_for_ann_str = 'Description'.ljust(40) + 'BinSize'.ljust(40) + 'TrainRatio'.ljust(
        40) + 'Algorithm'.ljust(40) + \
                              'Accuracy'.ljust(40) + 'RMSE'.ljust(40) + 'ElapsedTime'.ljust(40) + 'Nodes'.ljust(
        40) + 'Layers'.ljust(40) + 'ActivationFunc'.ljust(40) + 'Epocs'.ljust(40) + 'BatchSize'.ljust(40)

    # this will gathers all fild values according to above filed names
    one_line_report_str = ''


class SVC_Parameters:
    # best hyperparameters
    C = .68
    cache_size = 400
    class_weight = None
    coef0 = 0.35
    decision_function_shape = 'ovo'
    degree = 3
    gamma = 1
    kernel = 'linear'
    max_iter = 1000
    probability = True
    random_state = 17
    shrinking = True
    tol = 0.0001
    verbose = False

    # best data parameters
    svc_classifier_str = "svc_clf"
    svc_best_bin_size = 32
    svc_best_train_ratio = .8  # .7

    # this is user predefined range for SVM classifier
    # if you do not mention any hyper parameters in here grid search takes its default values
    svc_grid_parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': (0.78, 0.88, 0.98, 0.68),
                           'gamma': (1, 2, 3, 'scale'), 'decision_function_shape': ('ovo', 'ovr'),
                           'shrinking': (True, False), 'max_iter': [10000], 'coef0': [0.35, 0.4, 0.45, 0.5, 0.55]}


class RFC_Parameters:
    # best hyperparameters
    n_estimators = 200
    criterion = 'gini'
    max_depth = 4
    min_samples_split = 0.1
    min_samples_leaf = 2
    min_weight_fraction_leaf = 0.002
    max_features = 'sqrt'
    max_leaf_nodes = None
    min_impurity_decrease = 0.0000001
    # min_impurity_split = 0.00000001
    bootstrap = True
    oob_score = False
    n_jobs = None
    random_state = 42
    verbose = False
    warm_start = True
    class_weight = None

    # best data parameters
    rfc_classifier_str = "rfc_clf"
    rfc_best_bin_size = 32  # 8
    rfc_best_train_ratio = 0.8  # 0.7
    # this is user predefined range for Random Forest classifier
    # if you do not mention any hyper parameters in here grid search takes its default values
    rfc_grid_parameters = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 7, None],
                           'min_weight_fraction_leaf': [0.0, 0.1], 'n_estimators': [200, 500, 1000], 'n_jobs': [-1],
                           'oob_score': [True, False], 'warm_start': [True], 'verbose': [False],
                           'bootstrap': [True], 'max_features': ['auto', 'sqrt'],
                           'min_samples_leaf': [1, 2, 4], 'min_samples_split': [.1, .2, .5],
                           }


class KNC_Parameters:
    # best hyperparameters
    n_neighbors = 3
    weights = 'distance'
    metric = 'euclidean'
    algorithm = 'ball_tree'
    leaf_size = 1
    p = 1
    # best data parameters
    knc_classifier_str = "knc_clf"
    knc_best_bin_size = 64
    knc_best_train_ratio = .6
    # this is user predefined range for KNN classifier
    # if you do not mention any hyper parameters in here grid search takes its default values
    knc_grid_parameters = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance'], 'p': [1, 2],
                           'metric': ['euclidean', 'manhattan', 'minkowski'],
                           'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'leaf_size': [1, 3, 5]
                           }


class BNB_Parameters:
    # best hyperparameters
    alpha = 1
    binarize = 0.0
    fit_prior = True
    class_prior = None

    # best data parameters
    bnb_classifier_str = "bnb_clf"
    bnb_best_bin_size = 8
    bnb_best_train_ratio = 0.8
    # this is user predefined range for BernoulliNB classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    bnb_grid_parameters = {'alpha': [1], 'binarize': [0.0], 'fit_prior': [True, False], 'class_prior': ['None']}


class MNB_Parameters:
    # best hyperparameters
    alpha = 1
    fit_prior = True
    class_prior = None

    # best data parameters
    mnb_classifier_str = "mnb_clf"
    mnb_best_bin_size = 8
    mnb_best_train_ratio = 0.8
    # this is user predefined range for MultinominalNB classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    mnb_grid_parameters = {'alpha': [1], 'fit_prior': [True, False], 'class_prior': ['None']}


class GNB_Parameters:
    # best hyperparameters
    priors = None
    var_smoothing = 0.000000001

    # best data parameters
    gnb_classifier_str = "gnb_clf"
    gnb_best_bin_size = 8
    gnb_best_train_ratio = 0.8
    # this is user predefined range for GaussianNB classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    gnb_grid_parameters = {'var_smoothing': [0.00000001, 0.0000001, 0.000001]}


class ABC_Parameters:
    # best hyperparameters
    base_estimator = None
    n_estimators = 1000
    learning_rate = 0.9
    algorithm = 'SAMME.R'
    random_state = 42

    # best data parameters
    abc_classifier_str = "abc_clf"
    abc_best_bin_size = 256
    abc_best_train_ratio = 0.8
    # this is user predefined range for AdaBoost classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    abc_grid_parameters = {'base_estimator': [None], 'n_estimators': [100, 200, 500, 1000],
                           'learning_rate': [.005, 0.9, 1.0],
                           'algorithm': ['SAMME.R', 'SAMME'], 'random_state': [17, 42, 11]}


class DT_Parameters:
    # best hyperparameters
    criterion = 'entropy'
    splitter = 'random'
    max_depth = 6
    min_sample_split = 3
    min_samples_leaf = 1
    min_weight_fraction_leaf = 0.1
    max_features = None
    random_state = None
    max_leaf_nodes = None
    min_impurity_decrease = 0.2
    class_weight = None
    presort = False
    min_impurity_split = 0.0000001

    # best data parameters
    dt_classifier_str = "dt_clf"
    dt_best_bin_size = 32
    dt_best_train_ratio = 0.8
    # this is user predefined range for AdaBoost classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    dt_grid_parameters = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random'],
                          'max_depth': [None, 1, 3, 5, 6], 'min_samples_leaf': [1, 2, 3],
                          'min_weight_fraction_leaf': [0, 0.1, 0.2, 0.3],
                          'max_features': [None], 'random_state': [None], 'max_leaf_nodes': [None],
                          'min_impurity_decrease': [0.2, .3, .5],
                          'min_impurity_split': [0.0001, 0.001, 0.00001, 0.0000001],
                          'class_weight': [None], 'presort': [True, False]}


class GBC_Parameters:
    # best hyperparameters
    loss = 'deviance'
    learning_rate = 1
    n_estimators = 1000
    subsample = 1.0
    criterion = 'friedman_mse'
    min_samples_split = 10
    min_samples_leaf = 4
    min_weight_fraction_leaf = 0.0
    max_depth = 100
    min_impurity_decrease = 2.0
    min_impurity_split = 0.001
    init = None
    random_state = None
    max_features = None
    verbose = 0
    max_leaf_nodes = None
    warm_start = False
    presort = 'auto'
    validation_fraction = 0.01
    n_iter_no_change = None
    tol = 0.01

    # best data parameters
    gbc_classifier_str = "gbc_clf"
    gbc_best_bin_size = 8
    gbc_best_train_ratio = 0.8
    # this is user predefined range for GradientBoosting classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    gbc_grid_parameters = {'loss': ['deviance'], 'learning_rate': [0.5, 0.1, 0.9, 0.6],
                           'n_estimators': [100, 200, 500, 1000], 'criterion': ['friedman_mse'],
                           'min_samples_split': [1, 2, 3], 'min_samples_leaf': [2, 3, 1],
                           'min_weight_fraction_leaf': [0.0], 'max_depth': [3, 4, 6],
                           'min_impurity_decrease': [0.0], 'min_impurity_split': [0.0000001, .00001],
                           'warm_start': [False, True], 'presort': ['auto', False, True],
                           'validation_fraction': [0.1], 'n_iter_no_change': [1000], 'tol': [0.0001]}


class IF_Parameters:
    # best hyperparameters
    n_estimators = 1000
    max_samples = 'auto'
    contamination = 0.03
    max_features = 0.8
    # n_features = 3.0
    bootstrap = False
    n_jobs = None
    behaviour = 'new'
    random_state = None
    verbose = 0
    warm_start = False

    # best data parameters
    if_classifier_str = "if_clf"
    if_best_bin_size = 8
    if_best_train_ratio = 0.8
    # this is user predefined range for IsolationForest classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    if_grid_parameters = {n_estimators: [200], max_samples: ['auto'], contamination: [0.0001], random_state: [17],
                          max_features: [0.01], bootstrap: [False], n_jobs: [None], behaviour: ['new'],
                          verbose: [10], warm_start: [False]}


class SGDC_Parameters:
    # best hyperparameters
    alpha = 0.001
    average = True
    class_weight = None
    early_stopping = False
    epsilon = 0.5
    eta0 = 0.0
    fit_intercept = True
    l1_ratio = 0.15
    learning_rate = 'optimal'
    loss = 'hinge'
    max_iter = 1000
    n_iter_no_change = 5
    n_jobs = None
    penalty = 'l2'
    power_t = 0.3
    random_state = None
    shuffle = True
    tol = 0.001
    validation_fraction = 0.1
    verbose = 0
    warm_start = True

    # best data parameters
    sgdc_classifier_str = "sgdc_clf"
    sgdc_best_bin_size = 16
    sgdc_best_train_ratio = 0.7
    # this is user predefined range for SGDClassifier classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    sgdc_grid_parameters = {'alpha': [0.0001, 0.001, 0.00001], 'average': [False, True], 'class_weight': [None],
                            'early_stopping': [False], 'epsilon': [0.1, 0.5, 0.4], 'eta0': [0.0, 0.1],
                            'fit_intercept': [True],
                            'l1_ratio': [0.15, 0.10, .2], 'learning_rate': ['optimal'],
                            'loss': ['hinge', 'log', 'perceptron'], 'max_iter': [1000],
                            'n_iter_no_change': [5], 'n_jobs': [-1], 'penalty': ['l2'], 'power_t': [0.5, 0.3],
                            'random_state': [None], 'shuffle': [True], 'tol': [0.001, 0.1],
                            'validation_fraction': [0.1, 0.2], 'verbose': [False], 'warm_start': [True]}


class LinR_Parameters:
    # best hyperparameters
    fit_intercept = True
    normalize = True
    copy_X = True
    n_jobs = True

    # best data parameters
    linr_regressor_str = "linr_reg"
    linr_best_bin_size = 64
    linr_best_train_ratio = 0.8
    # this is user predefined range for LinearRegression classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    linr_grid_parameters = {fit_intercept: [True, False],
                            normalize: [False, True],
                            copy_X: [False, True],
                            n_jobs: [False, True]}


class LogR_Parameters:
    # best hyperparameters
    penalty = 'l2'
    dual = False
    tol = 1e-4
    C = 1.0
    fit_intercept = True
    intercept_scaling = 1
    class_weight = None
    random_state = None
    solver = 'lbfgs'
    max_iter = 100
    multi_class = 'multinomial'
    verbose = 0
    warm_start = False
    n_jobs = None
    l1_ratio = None

    # best data parameters
    logr_classifier_str = "logr_clf"
    logr_best_bin_size = 64
    logr_best_train_ratio = 0.8
    # this is user predefined range for LinearRegression classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    logr_grid_parameters = {'penalty': ['l1', 'l2'], 'dual': [False], 'tol': [1e-4], 'C': [1.0],
                            'fit_intercept': [True], 'intercept_scaling': [1],
                            'class_weight': [None], 'random_state': [None], 'solver': ['warn'], 'max_iter': [1000],
                            'multi_class': ['warn'],
                            'verbose': [0], 'warm_start': [True], 'n_jobs': [-1], 'l1_ratio': [None]}


class XGBC_Parameters:
    # Base parameters for xgb. parameters can change by booster type     # best hyperparameters
    booster = 'gbtree'
    silent = None
    verbosity = 1
    nthread = None
    objective = 'binary:logistic'

    # gbtree booster parameters
    learning_rate = 0.1
    gamma = 0
    max_depth = 3
    min_child_weight = 1
    max_delta_step = 0
    subsample = 1
    colsample_bytree = 1
    colsample_bylevel = 1
    colsample_bynode = 1
    reg_alpha = 0
    reg_lambda = 1
    # tree_method = 'exact'
    sketch_eps = 0.3
    scale_pos_weight = 1
    updater = 'grow_colmaker', 'prune'
    refresh_leaf = 1
    process_type = 'default'
    grow_policy = 'depthwise'
    max_leaves = 0
    max_bins = 256
    num_parallel_tree = 1

    # dart booster parameters
    sample_type = 'uniform'
    normalize_type = 'tree'
    rate_drop = 0.0
    one_drop = 0
    skip_drop = 1

    # gblinear booster parameters
    reg_lambda = 1
    reg_alpha = 0
    updater = 'shotgun'
    feature_selector = 'cyclic'
    top_k = 0

    # best data parameters
    xgbc_classifier_str = "xgbc_clf"
    xgbc_best_bin_size = 8
    xgbc_best_train_ratio = 0.7
    # this is user predefined range for LinearRegression classifier
    # if you do not mention any hyper parameters in here grid search takes its default values

    xgbc_grid_parameters = {'max_depth': [3], 'learning_rate': [0.1], 'nthread': [8, -1], 'verbosity': [0],
                            'silent': [None],
                            'objective': ["binary:logistic", 'reg:squarederror'], 'booster': ['gbtree', 'dart'],
                            'gamma': [0.5, 0.3, 0], 'min_child_weight': [1], 'max_delta_step': [0], 'subsample': [1],
                            'colsample_bytree': [1], 'colsample_bylevel': [1], 'colsample_bynode': [1],
                            'reg_alpha': [0],
                            'reg_lambda': [1], 'scale_pos_weight': [1]}


class ANN_Parameters:
    ann_classifier_str = "ann_clf"
    ann_nOfL = 3
    ann_nOfN = [(ImageConfig.bin_size + 2), 512, 12]
    ann_aF = ['relu', 'relu', 'softmax']
    ann_oF = 'adam'
    ann_inputDim = ImageConfig.bin_size + 2
    ann_initMode = 'RandomNormal'

    ann_best_bin_size = 8
    ann_best_train_ratio = 0.7
    ann_best_layer_number = 3
    ann_best_epoc_number = 12
    ann_best_activation_function = "relu"
    ann_best_node_number = 9
    ann_grid_parameters = {'batch_size': [7, 14], 'epochs': [12, 6],
                           'init_mode': ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                                         'glorot_uniform', 'he_normal', 'he_uniform'],
                           'optimizer': ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
                           }


class DBN_Parameters:
    dbn_classifier_str = "dbn_clf"
    dbn_hidden_layers_structure = [100, 100]
    dbn_learning_rate_rbm = 0.1
    dbn_learning_rate = 0.013
    dbn_n_epochs_rbm = 100
    dbn_n_iter_backprop = 100
    dbn_batch_size = 6
    dbn_activation_function = 'sigmoid'
    dbn_dropout_p = 0.3
    dbn_verbose = 0
    dbn_dbn_best_bin_size = 8
    dbn_dbn_best_train_ratio = 0.7
