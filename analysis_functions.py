"""
@author: DURSUN
"""
import numpy as np
import pandas as pd
from sklearn import neighbors, tree, ensemble, utils
from sklearn.metrics import mean_squared_error, make_scorer, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import datetime
import math
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)



def RMSE(test_data, test_label, model):
    test_predictions = model.predict(test_data)
    model_mse = mean_squared_error(test_label, test_predictions)
    return np.sqrt(model_mse)


def calculate_default_score(train_data, train_label, test_data, test_label, model):
    x_train, y_train, x_test, y_test = shuffle_data(train_data, train_label)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    predicted_labels = model.predict(test_data)
    predicted_labels = pd.DataFrame(predicted_labels)
    #test_prediction_score = accuracy_score(test_label,test_prediction)
    print(model)
    print("\t Score:", score)
    #print("\t Pred Score:",test_prediction_score)
    # print("\t Orig Labels:", test_label)
    # print("\t Pred Labels:", predicted_labels)
    rmse = RMSE(test_data, test_label, model)
    print("\t RMSE:", rmse)
    print(classification_report(test_label, predicted_labels))
    print("------------------------------------------------------------------")
    return model


def calculate_gridsearchCV(train_data, train_label, test_data, test_label, model, params):

    x_train, y_train, x_test, y_test = shuffle_data(train_data, train_label)
    grid_params = params
    gs = GridSearchCV(
        model,
        grid_params,
        verbose=1,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        iid='True'
    )
    # np.set_printoptions(precision=2)
    # print(y_train.shape)
    gs.fit(x_train, y_train)
    score = gs.score(x_test,y_test)
    predicted_labels = gs.predict(test_data)
    print("Orgi Labels:", test_label)
    print("Pred Labels:", predicted_labels)
    clf_report = classification_report(test_label, predicted_labels)
    print("Best Parameters: ", gs.best_params_)
    print("Score:", score)
    print("Classification Report:\n", clf_report)
    rmse = RMSE(test_data, test_label, gs)
    print("RMSE:", rmse)
    print("------------------------------------------------")
    return gs


def shuffle_data(train_data, train_label):

    #print(train_label)
    shuffled_data, shuffled_label = utils.shuffle(train_data, train_label, random_state=42)
    # print(shuffled_label)
    data_df = pd.DataFrame(shuffled_data)
    label_df = pd.DataFrame(shuffled_label)
    #print(label_df)

    #print(len(data_df), len(label_df))
    #print(42*0.8)

    x_train = data_df[0:33]
    x_test = data_df[33:]

    y_train = label_df[0:33]
    y_test = label_df[33:]
    return x_train, y_train, x_test, y_test


def calculate_ada_boost(train_data, train_label, test_data, test_label, model):

    x_train, y_train, x_test, y_test = shuffle_data(train_data, train_label)
    ada_clf = AdaBoostClassifier(model)
    ada_clf.fit(x_train, y_train)
    ada_score = ada_clf.score(x_test, y_test)
    predicted_labels = ada_clf.predict(test_data)

    print("Ada-Bosst Score:", ada_score)
    print("Orig Labels:", test_label)
    print("Pred Labels:", predicted_labels)
    print("------------------------------------------------")


train_data = np.load('Train_data_8_train70_2019-05-28_15-21-30.npy')
train_label = np.load('Train_label_8_train70_2019-05-28_15-21-30.npy')

test_data = np.load('Test_data_8_train70_2019-05-28_15-21-30.npy')
test_label = np.load('Test_label_8_train70_2019-05-28_15-21-30.npy')

test_data = pd.DataFrame(test_data)
test_label = pd.DataFrame(test_label)


#print(train_data.shape)
#print(train_label.shape)
#np.set_printoptions(precision=2)
start = datetime.datetime.now()

#calculate_ada_boost(train_data, train_label, test_data, test_label, ensemble.RandomForestClassifier())

# calculate_default_score(train_label, train_label, test_data, test_label, tree.DecisionTreeClassifier())

# calculate_default_score(train_data, train_label, test_data, test_label, neighbors.KNeighborsClassifier())

# calculate_gridsearchCV(train_data, train_label, test_data, test_label, neighbors.KNeighborsClassifier(), params={
#         'n_neighbors': [3, 5, 11, 19, 21],
#         'weights': ['uniform', 'distance'],
#         'metric': ['euclidean', 'manhattan', 'minkowski'],
#         'algorithm': ['ball_tree', 'kd_tree', 'brute'],
#         'leaf_size': [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90],
#         'p': [3, 5, 7, 9]
#     })

# calculate_default_score(train_data, train_label, test_data, test_label, BernoulliNB())
#calculate_gridsearchCV(train_data, train_label, test_data, test_label, 'BernNB')

# calculate_default_score(train_data, train_label, test_data, test_label, MultinomialNB())
#calculate_gridsearchCV(train_data, train_label, test_data, test_label, 'MultiNB')

#calculate_default_score(train_data, train_label, test_data, test_label, GaussianNB())

#calculate_gridsearchCV(train_data, train_label, test_data, test_label, 'GausNB')

# calculate_default_score(train_data, train_label, test_data, test_label, RandomForestClassifier())

# calculate_gridsearchCV(train_data, train_label, test_data, test_label, RandomForestClassifier(), params={'bootstrap': [True], 'class_weight': ['balanced'], 'criterion': ['gini'],
#                           'max_depth': [6, 7], 'max_features': ['auto'], 'max_leaf_nodes': [7, 9], 'min_impurity_decrease': [0.0, 0.001],
#                           'min_impurity_split': [None], 'min_samples_leaf': [1, 2],
#                           'min_samples_split': [2, 4, 6], 'min_weight_fraction_leaf': [0.0, 0.1], 'n_estimators': [100, 200], 'n_jobs': [-1],
#                           'oob_score': [False, True],'random_state': [13, 15, 17, 42], 'verbose': [1, 0], 'warm_start': [False, True]
#                           })



# print(x_train.shape)
# print(y_train.shape)

# start = datetime.datetime.now()
# scaler = preprocessing.MinMaxScaler()
# X_train = pd.DataFrame(scaler.fit_transform(train_data))
# #Random shuffle training data
# X_train.sample(frac=1)
# X_test = pd.DataFrame(scaler.transform(test_data))


# from sklearn.decomposition import PCA
# pca = PCA(n_components=2, svd_solver='full')
# X_train_PCA = pca.fit_transform(X_train)
# X_train_PCA = pd.DataFrame(X_train_PCA)
# X_train_PCA.index = X_train.index
#
# X_test_PCA = pca.transform(X_test)
# X_test_PCA = pd.DataFrame(X_test_PCA)
# X_test_PCA.index = X_test.index

#
# plt.plot(X_train_PCA)
# plt.show()

#
# rf_clf = RFConfig.rf_one_classifier
# rf_clf.fit(x_train, y_train)
# rf_one_score = rf_clf.score(x_test, y_test)
# rf_one_predicted_labels = rf_clf.predict(test_data)
# print("RF Score:", rf_one_score)
# print("RF RMSE", Functions.calculate_rmse(test_data, test_label, rf_clf))


# abc_clf = ABCConfig.abc_one_classifier
# abc_clf.fit(x_train, y_train)
# score = abc_clf.score(x_test, y_test)
# predicted_labels = abc_clf.predict(train_data)
#
# print(score)
# print(predicted_labels)


# svc_clf = SVMConfig.svm_one_classifier
# svc_clf.fit(x_train, y_train)
# svc_one_score = svc_clf.score(x_test, y_test)
# svc_one_predicted_labels = svc_clf.predict(test_data)
# print("SVC Score:", svc_one_score)
# print("SVC RMSE", Functions.calculate_rmse(test_data, test_label, svc_clf))

# for i in range(6):
#     for j in range(42):
#         pass
#
# feature_mean = []
#
# feature_variance = []

# np.set_printoptions(precision=2)
# for i in range(8):
#     mean = np.mean(train_data[:, i])
#     variance = np.var(train_data[:, i])
#     feature_mean.append(mean)
#     feature_variance.append(variance)
#
# gau_dist =1
# for i in range(8):
#     gau_dist *= 1/((feature_variance[i]**1/2) * np.sqrt(2 * np.pi)) * np.exp(-(bins - feature_mean[i])**2 / (2 * feature_variance[i]**2))

# end = datetime.datetime.now()
# elapsed_time = end - start
# print("Elapsed Time:", elapsed_time)


end = datetime.datetime.now()
elapsed = end - start
print("Elapsed Time:", elapsed)

