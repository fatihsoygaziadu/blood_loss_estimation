from Config import *
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier, LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn import neighbors, tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from xgboost import XGBClassifier
import keras
from keras import regularizers
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.layers import InputLayer
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from dbn import SupervisedDBNClassification


class MySVC:
    def __init__(self):
        # name of the classifier
        self.classifier_str = SVC_Parameters.svc_classifier_str
        self.best_train_ratio = SVC_Parameters.svc_best_train_ratio
        self.best_bin_size = SVC_Parameters.svc_best_bin_size

        # This calls classifier for one set of hyperparameters no range of values
        # calls SVM's SVC constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = SVC(C=SVC_Parameters.C, cache_size=SVC_Parameters.cache_size,
                                  class_weight=SVC_Parameters.class_weight, coef0=SVC_Parameters.coef0,
                                  decision_function_shape=SVC_Parameters.decision_function_shape,
                                  degree=SVC_Parameters.degree, gamma=SVC_Parameters.gamma,
                                  kernel=SVC_Parameters.kernel,
                                  max_iter=SVC_Parameters.max_iter, probability=SVC_Parameters.probability,
                                  random_state=SVC_Parameters.random_state, shrinking=SVC_Parameters.shrinking,
                                  tol=SVC_Parameters.tol, verbose=SVC_Parameters.verbose)

        # This calls classifier for range of hyperparameters
        # calls SVM's SVC constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fethed from SVM library
        self.grid_classifier = GridSearchCV(SVC(), SVC_Parameters.svc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyRFC:
    def __init__(self):
        # name of the classifier
        self.classifier_str = RFC_Parameters.rfc_classifier_str
        self.best_train_ratio = RFC_Parameters.rfc_best_train_ratio
        self.best_bin_size = RFC_Parameters.rfc_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls RandomForestClassier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = RandomForestClassifier(n_estimators=RFC_Parameters.n_estimators,
                                                     criterion=RFC_Parameters.criterion,
                                                     max_depth=RFC_Parameters.max_depth,
                                                     min_samples_split=RFC_Parameters.min_samples_split,
                                                     min_samples_leaf=RFC_Parameters.min_samples_leaf,
                                                     min_weight_fraction_leaf=RFC_Parameters.min_weight_fraction_leaf,
                                                     max_features=RFC_Parameters.max_features,
                                                     max_leaf_nodes=RFC_Parameters.max_leaf_nodes,
                                                     min_impurity_decrease=RFC_Parameters.min_impurity_decrease,
                                                     bootstrap=RFC_Parameters.bootstrap,
                                                     oob_score=RFC_Parameters.oob_score, n_jobs=RFC_Parameters.n_jobs,
                                                     random_state=RFC_Parameters.random_state,
                                                     verbose=RFC_Parameters.verbose,
                                                     warm_start=RFC_Parameters.warm_start,
                                                     class_weight=RFC_Parameters.class_weight)
        # This calls classifier for range of hyperparameters
        # calls RandomForestClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from sklearn.ensemble library
        self.grid_classifier = GridSearchCV(RandomForestClassifier(), RFC_Parameters.rfc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyKNC:
    def __init__(self):
        # name of the classifier
        self.classifier_str = KNC_Parameters.knc_classifier_str
        self.best_train_ratio = KNC_Parameters.knc_best_train_ratio
        self.best_bin_size = KNC_Parameters.knc_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls neighbors.KNeighborsClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = neighbors.KNeighborsClassifier(n_neighbors=KNC_Parameters.n_neighbors,
                                                             weights=KNC_Parameters.weights,
                                                             metric=KNC_Parameters.metric,
                                                             algorithm=KNC_Parameters.algorithm,
                                                             leaf_size=KNC_Parameters.leaf_size, p=KNC_Parameters.p)
        # This calls classifier for range of hyperparameters
        # calls neighbors.KNeighborsClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from neighbors library
        self.grid_classifier = GridSearchCV(neighbors.KNeighborsClassifier(), KNC_Parameters.knc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyBNB:
    def __init__(self):
        # name of the classifier
        self.classifier_str = BNB_Parameters.bnb_classifier_str
        self.best_train_ratio = BNB_Parameters.bnb_best_train_ratio
        self.best_bin_size = BNB_Parameters.bnb_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls naive_bayes.BernoulliNB's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = BernoulliNB(alpha=BNB_Parameters.alpha, binarize=BNB_Parameters.binarize,
                                          fit_prior=BNB_Parameters.fit_prior,
                                          class_prior=BNB_Parameters.class_prior)

        # This calls classifier for range of hyperparameters
        # calls naive_bayes.BernoulliNB's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from naive_bayes library
        self.grid_classifier = GridSearchCV(BernoulliNB(), BNB_Parameters.bnb_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyMNB:
    def __init__(self):
        # name of the classifier
        self.classifier_str = MNB_Parameters.mnb_classifier_str
        self.best_train_ratio = MNB_Parameters.mnb_best_train_ratio
        self.best_bin_size = MNB_Parameters.mnb_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls naive_bayes.MultinomialNB's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = MultinomialNB(alpha=MNB_Parameters.alpha, fit_prior=MNB_Parameters.fit_prior,
                                            class_prior=MNB_Parameters.class_prior)

        # This calls classifier for range of hyperparameters
        # calls naive_bayes.MultinomialNB's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from naive_bayes library
        self.grid_classifier = GridSearchCV(MultinomialNB(), MNB_Parameters.mnb_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyGNB:
    def __init__(self):
        # name of the classifier
        self.classifier_str = GNB_Parameters.gnb_classifier_str
        self.best_train_ratio = GNB_Parameters.gnb_best_train_ratio
        self.best_bin_size = GNB_Parameters.gnb_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls naive_bayes.GaussianNB's constructor it initialise with given hyerparameters that taken from our algorithms_data module

        self.one_classifier = GaussianNB(priors=GNB_Parameters.priors, var_smoothing=GNB_Parameters.var_smoothing)

        # This calls classifier for range of hyperparameters
        # calls naive_bayes.GaussianNB's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from naive_bayes library

        self.grid_classifier = GridSearchCV(GaussianNB(), GNB_Parameters.gnb_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyABC:
    def __init__(self):
        # name of the classifier
        self.classifier_str = ABC_Parameters.abc_classifier_str
        self.best_train_ratio = ABC_Parameters.abc_best_train_ratio
        self.best_bin_size = ABC_Parameters.abc_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls ensemble.AdaBoostClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        be = MySVC()

        self.base_estimator = be.one_classifier
        self.one_classifier = AdaBoostClassifier(base_estimator=ABC_Parameters.base_estimator,
                                                 n_estimators=ABC_Parameters.n_estimators,
                                                 learning_rate=ABC_Parameters.learning_rate,
                                                 algorithm=ABC_Parameters.algorithm,
                                                 random_state=ABC_Parameters.random_state)

        # This calls classifier for range of hyperparameters
        # calls ensemble.AdaBoostClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from ensemble library
        self.grid_classifier = GridSearchCV(AdaBoostClassifier(), ABC_Parameters.abc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyDT:
    def __init__(self):
        # name of the classifier
        self.classifier_str = DT_Parameters.dt_classifier_str
        self.best_train_ratio = DT_Parameters.dt_best_train_ratio
        self.best_bin_size = DT_Parameters.dt_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls tree.DecisionTreeClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module

        self.one_classifier = tree.DecisionTreeClassifier(criterion=DT_Parameters.criterion,
                                                          splitter=DT_Parameters.splitter,
                                                          max_depth=DT_Parameters.max_depth,
                                                          min_samples_leaf=DT_Parameters.min_samples_leaf,
                                                          min_weight_fraction_leaf=DT_Parameters.min_weight_fraction_leaf,
                                                          max_features=DT_Parameters.max_features,
                                                          random_state=DT_Parameters.random_state,
                                                          max_leaf_nodes=DT_Parameters.max_leaf_nodes,
                                                          min_impurity_decrease=DT_Parameters.min_impurity_decrease,
                                                          class_weight=DT_Parameters.class_weight,
                                                          presort=DT_Parameters.presort)

        # This calls classifier for range of hyperparameters
        # calls tree.DecisionTreeClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from ensemble library
        self.grid_classifier = GridSearchCV(tree.DecisionTreeClassifier(), DT_Parameters.dt_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyGBC:
    def __init__(self):
        # name of the classifier
        self.classifier_str = GBC_Parameters.gbc_classifier_str
        self.best_train_ratio = GBC_Parameters.gbc_best_train_ratio
        self.best_bin_size = GBC_Parameters.gbc_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls ensemble.GradientBoostingClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module

        self.one_classifier = GradientBoostingClassifier(loss=GBC_Parameters.loss,
                                                         learning_rate=GBC_Parameters.learning_rate,
                                                         n_estimators=GBC_Parameters.n_estimators,
                                                         subsample=GBC_Parameters.subsample,
                                                         criterion=GBC_Parameters.criterion,
                                                         min_samples_split=GBC_Parameters.min_samples_split,
                                                         min_samples_leaf=GBC_Parameters.min_samples_leaf,
                                                         min_weight_fraction_leaf=GBC_Parameters.min_weight_fraction_leaf,
                                                         max_depth=GBC_Parameters.max_depth,
                                                         min_impurity_decrease=GBC_Parameters.min_impurity_decrease,
                                                         init=GBC_Parameters.init,
                                                         random_state=GBC_Parameters.random_state,
                                                         max_features=GBC_Parameters.max_features,
                                                         verbose=GBC_Parameters.verbose,
                                                         max_leaf_nodes=GBC_Parameters.max_leaf_nodes,
                                                         warm_start=GBC_Parameters.warm_start,
                                                         presort=GBC_Parameters.presort,
                                                         validation_fraction=GBC_Parameters.validation_fraction,
                                                         n_iter_no_change=GBC_Parameters.n_iter_no_change,
                                                         tol=GBC_Parameters.tol)

        # This calls classifier for range of hyperparameters
        # calls ensemble.GradientBoostingClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from ensemble library

        self.grid_classifier = GridSearchCV(GradientBoostingClassifier(), GBC_Parameters.gbc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyIF:
    def __init__(self):
        # name of the classifier
        self.classifier_str = IF_Parameters.if_classifier_str
        self.best_train_ratio = IF_Parameters.if_best_train_ratio
        self.best_bin_size = IF_Parameters.if_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls ensemble.IsolationForest's constructor it initialise with given hyerparameters that taken from our algorithms_data module

        self.one_classifier = IsolationForest(n_estimators=IF_Parameters.n_estimators,
                                              max_samples=IF_Parameters.max_samples,
                                              contamination=IF_Parameters.contamination,
                                              max_features=IF_Parameters.max_features,
                                              bootstrap=IF_Parameters.bootstrap,
                                              n_jobs=IF_Parameters.n_jobs,
                                              behaviour=IF_Parameters.behaviour,
                                              random_state=IF_Parameters.random_state,
                                              verbose=IF_Parameters.verbose,
                                              warm_start=IF_Parameters.warm_start
                                              )

        # This calls classifier for range of hyperparameters
        # calls ensemble.IsolationForest's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from ensemble library
        self.grid_classifier = GridSearchCV(IsolationForest(), IF_Parameters.if_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MySGDC:
    def __init__(self):
        # name of the classifier
        self.classifier_str = SGDC_Parameters.sgdc_classifier_str
        self.best_train_ratio = SGDC_Parameters.sgdc_best_train_ratio
        self.best_bin_size = SGDC_Parameters.sgdc_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls SGDClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = SGDClassifier(alpha=SGDC_Parameters.alpha,
                                            average=SGDC_Parameters.average,
                                            class_weight=SGDC_Parameters.class_weight,
                                            early_stopping=SGDC_Parameters.early_stopping,
                                            epsilon=SGDC_Parameters.epsilon,
                                            eta0=SGDC_Parameters.eta0,
                                            fit_intercept=SGDC_Parameters.fit_intercept,
                                            l1_ratio=SGDC_Parameters.l1_ratio,
                                            learning_rate=SGDC_Parameters.learning_rate,
                                            loss=SGDC_Parameters.loss,
                                            max_iter=SGDC_Parameters.max_iter,
                                            n_iter_no_change=SGDC_Parameters.n_iter_no_change,
                                            n_jobs=SGDC_Parameters.n_jobs,
                                            penalty=SGDC_Parameters.penalty,
                                            power_t=SGDC_Parameters.power_t,
                                            random_state=SGDC_Parameters.random_state,
                                            shuffle=SGDC_Parameters.shuffle,
                                            tol=SGDC_Parameters.tol,
                                            validation_fraction=SGDC_Parameters.validation_fraction,
                                            verbose=SGDC_Parameters.verbose,
                                            warm_start=SGDC_Parameters.warm_start
                                            )
        # This calls classifier for range of hyperparameters
        # calls SGDClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from * library
        self.grid_classifier = GridSearchCV(SGDClassifier(), SGDC_Parameters.sgdc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyLinR:
    def __init__(self):
        # name of the regression
        self.classifier_str = LinR_Parameters.linr_regressor_str
        self.best_train_ratio = LinR_Parameters.linr_best_train_ratio
        self.best_bin_size = LinR_Parameters.linr_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls LinearRegression's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        self.one_classifier = LinearRegression(fit_intercept=LinR_Parameters.fit_intercept,
                                               normalize=LinR_Parameters.normalize,
                                               copy_X=LinR_Parameters.copy_X,
                                               n_jobs=LinR_Parameters.n_jobs)

        # This calls classifier for range of hyperparameters
        # calls LinearRegression's constructor it initialise with given hyerparameters that taken from our algorithms_data module
        # missing (undefined) hyperparameters fetched from * library
        self.grid_classifier = GridSearchCV(LinearRegression(), LinR_Parameters.linr_grid_parameters)
        # linr_ovr_classifier = OneVsRestClassifier(linr_one_classifier)


class MyLogR:
    def __init__(self):
        # name of the regression
        self.classifier_str = LogR_Parameters.logr_classifier_str
        self.best_train_ratio = LogR_Parameters.logr_best_train_ratio
        self.best_bin_size = LogR_Parameters.logr_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls LogisticRegression's constructor it initialise with given hyerparameters that taken from our algorithms_data module

        self.one_classifier = LogisticRegression(penalty=LogR_Parameters.penalty,
                                                 dual=LogR_Parameters.dual,
                                                 tol=LogR_Parameters.tol,
                                                 C=LogR_Parameters.C,
                                                 fit_intercept=LogR_Parameters.fit_intercept,
                                                 intercept_scaling=LogR_Parameters.intercept_scaling,
                                                 class_weight=LogR_Parameters.class_weight,
                                                 random_state=LogR_Parameters.random_state,
                                                 solver=LogR_Parameters.solver,
                                                 max_iter=LogR_Parameters.max_iter,
                                                 multi_class=LogR_Parameters.multi_class,
                                                 verbose=LogR_Parameters.verbose,
                                                 warm_start=LogR_Parameters.warm_start,
                                                 n_jobs=LogR_Parameters.n_jobs,
                                                 l1_ratio=LogR_Parameters.l1_ratio)

        self.grid_classifier = GridSearchCV(LogisticRegression(), LogR_Parameters.logr_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


class MyXGBC:
    def __init__(self):
        # name of the regression
        self.classifier_str = XGBC_Parameters.xgbc_classifier_str
        self.best_train_ratio = XGBC_Parameters.xgbc_best_train_ratio
        self.best_bin_size = XGBC_Parameters.xgbc_best_bin_size
        # This calls classifier for one set of hyperparameters no range of values
        # calls XGBClassifier's constructor it initialise with given hyerparameters that taken from our algorithms_data module

        self.one_classifier = XGBClassifier(booster=XGBC_Parameters.booster,
                                            verbosity=XGBC_Parameters.verbosity,
                                            nthread=XGBC_Parameters.nthread,
                                            objective=XGBC_Parameters.objective,
                                            learning_rate=XGBC_Parameters.learning_rate,
                                            gamma=XGBC_Parameters.gamma,
                                            max_depth=XGBC_Parameters.max_depth,
                                            max_delta_step=XGBC_Parameters.max_delta_step,
                                            subsample=XGBC_Parameters.subsample,
                                            colsample_bytree=XGBC_Parameters.colsample_bytree,
                                            colsample_bylevel=XGBC_Parameters.colsample_bylevel,
                                            colsample_bynode=XGBC_Parameters.colsample_bynode,
                                            reg_alpha=XGBC_Parameters.reg_alpha,
                                            reg_lambda=XGBC_Parameters.reg_lambda,
                                            scale_pos_weight=XGBC_Parameters.scale_pos_weight)

        self.grid_classifier = GridSearchCV(XGBClassifier(), XGBC_Parameters.xgbc_grid_parameters)
        self.ovr_classifier = OneVsRestClassifier(self.one_classifier)


def create_ann_model(optimizer, input_dim, init_mode, activations, node_numbers, layer_number):
    model = Sequential()
    for i in range(layer_number):
        if i == 0:
            model.add(Dense(node_numbers[i], input_dim=input_dim, kernel_initializer=init_mode, activation=activations[i]))
            # print("ilk layer")
        elif i == layer_number - 1:
            model.add(Dense(node_numbers[i], activation=activations[i]))
            # print("Son layer".format(i))

        else:
            model.add(Dense(node_numbers[i], activation=activations[i]))
            # print("{}.layer".format(i))

    # Softmax nörondan çıkan output a göre hangi class a ait olduğunun olasılığını hesaplar. ActivationFunction bir
    # nörünun aktif edilip edilmeyeceğini belirler. Gelen inputun değerine göre alakalı mı alakasız mı olduguna bakar
    # bir anlamda
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class MyANN:
    def __init__(self, input_dim, optimizer, init_mode, activations, node_numbers, layer_number):
        self.classifier_str = ANN_Parameters.ann_classifier_str
        self.best_train_ratio = ANN_Parameters.ann_best_train_ratio
        self.best_bin_size = ANN_Parameters.ann_best_bin_size
        self.best_layer_number = ANN_Parameters.ann_best_layer_number
        self.best_epoc_number = ANN_Parameters.ann_best_epoc_number
        self.best_activation_function = ANN_Parameters.ann_best_activation_function
        self.best_node_number = ANN_Parameters.ann_best_node_number

        self.input_dim = input_dim
        self.optimizer = optimizer
        self.init_mode = init_mode
        self.activations = activations
        self.node_numbers = node_numbers
        self.layer_number = layer_number

        #        self.one_classifier = KerasClassifier(build_fn=create_ann_model, optimizer=mOptimizer, numberOfLayers=mNumberOfLayers, numberOfNodes=mNumberOfNodes, epochs=12, batch_size=7, activation=mActivationFunc)
        self.one_classifier = create_ann_model(optimizer=self.optimizer,
                                               input_dim=self.input_dim,
                                               init_mode=self.init_mode,
                                               activations=self.activations,
                                               node_numbers=self.node_numbers,
                                               layer_number=self.layer_number
                                               )
        self.grid_classifier = GridSearchCV(self.one_classifier, ANN_Parameters.ann_grid_parameters)


class MyDBN:
    def __init__(self):
        self.classifier_str = DBN_Parameters.dbn_classifier_str
        self.one_classifier = SupervisedDBNClassification(
                                  hidden_layers_structure=DBN_Parameters.dbn_hidden_layers_structure,
                                  learning_rate_rbm=DBN_Parameters.dbn_learning_rate_rbm,
                                  learning_rate=DBN_Parameters.dbn_learning_rate,
                                  n_epochs_rbm=DBN_Parameters.dbn_n_epochs_rbm,
                                  n_iter_backprop=DBN_Parameters.dbn_n_iter_backprop,
                                  batch_size=DBN_Parameters.dbn_batch_size,
                                  activation_function=DBN_Parameters.dbn_activation_function,
                                  dropout_p=DBN_Parameters.dbn_dropout_p,
                                  verbose=DBN_Parameters.dbn_verbose)
