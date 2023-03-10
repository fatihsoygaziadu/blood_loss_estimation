from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from tensorflow_core.python.tools.saved_model_cli import preprocess_inputs_arg_string
import tensorflow as tf
import image_processing
import Algorithms
import matplotlib.pyplot as plt
from sklearn import utils
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import Utility as ut
from Logger import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, train_test_split, \
    StratifiedKFold
from sklearn.pipeline import make_pipeline
import numpy as np
import time
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers import InputLayer
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import image
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.wrappers.scikit_learn import KerasClassifier

warnings.filterwarnings('ignore')


def execute_one_clf(ds, classifier):
    # declare and initialize time for calculating elapsed time end of function
    start_time = time.time()

    # making pipeline for scaling ever time
    clf_pipeline = make_pipeline(StandardScaler(), classifier)

    # fit the pipeline with train data and it's labels
    model = clf_pipeline.fit(ds.train_data, ds.train_label)

    # predicting train and unseen test data
    predicted_test_label = clf_pipeline.predict(ds.test_data)

    # Elapsed Time
    elapsed_time = time.time() - start_time

    return model, predicted_test_label, elapsed_time


def execute_grid_search(ds, classifier):
    # declare and initialize time for calculating elapsed time end of function
    start_time = time.time()

    # making pipeline for scaling ever time
    clf_pipeline = make_pipeline(StandardScaler(), classifier)

    # fit the pipeline with train data and it's labels
    model = clf_pipeline.fit(ds.train_data, ds.train_label)

    # predicting train and unseen test data
    predicted_test_label = clf_pipeline.predict(ds.test_data)

    # rmse calculation with function
    elapsed_time = time.time() - start_time

    return classifier, predicted_test_label, elapsed_time


def execute_cross_val(ds, classifier, k_split):
    start_time = time.time()

    clf_pipeline = make_pipeline(StandardScaler(), classifier)
    # X, y = data_joiner(ds)

    # cross validate and the other method returns a list of number
    # which can be neg_mean_score or accuracy by changing scoring parameter
    # scoring=None means that accuracy used from estimator score method 4
    # otherwise cross validation calculates the given score type
    cv = cross_validate(clf_pipeline, ds.train_data, ds.train_label, cv=k_split, return_estimator=True,
                        scoring=('accuracy', 'neg_mean_squared_error'), n_jobs=-1)
    pipe = cv['estimator'][0]
    accuracy = cv['test_accuracy'].mean()

    rmse = np.sqrt(- cv['test_neg_mean_squared_error'].mean())
    # cv_scores = cross_val_score(clf_pipeline, X, y, cv=k_split, scoring='r2')
    # print(sorted(cv.keys()))
    # print(accuracy)
    elapsed_time = time.time() - start_time
    predicted_test_label = pipe.predict(ds.test_data)
    # print(predicted_test_label)

    return clf_pipeline, predicted_test_label, accuracy, rmse, elapsed_time


def execute_classical_ann(ds, classifier, epochs, batch_size):
    start_time = time.time()

    # mScaler = StandardScaler()
    # ds.train_data = mScaler.fit_transform(ds.train_data, ds.train_label)
    # ds.test_data = mScaler.fit_transform(ds.test_data, ds.test_label)

    scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    ds.train_data = scaler.fit_transform(ds.train_data)
    ds.test_data = scaler.fit_transform(ds.test_data)

    model = classifier

    model.fit(ds.train_data, ds.train_label, epochs=epochs, batch_size=batch_size, verbose=0,
              shuffle=False)
    predicted_test_label = classifier.predict_classes(ds.test_data)
    # predicted_test_label = classifier.evaluate(ds.test_data, ds.test_label)
    # rmse calculation with function
    elapsed_time = time.time() - start_time

    return classifier, predicted_test_label, elapsed_time


def execute_cnn():
    start_time = time.time()

    image_generator = ImageDataGenerator(rescale=1/255, validation_split=0.3)

    train_dataset = image_generator.flow_from_directory(batch_size=24,
                                                        directory='RawImages',
                                                        shuffle=True,
                                                        target_size=(120, 252),
                                                        subset='training',
                                                        class_mode='categorical')

    test_dataset = image_generator.flow_from_directory(batch_size=24,
                                                       directory='RawImages',
                                                       shuffle=True,
                                                       target_size=(120, 252),
                                                       subset='validation',
                                                       class_mode='categorical')

    model = Sequential()
    model.add(Conv2D(24, (5, 5), activation='relu', input_shape=(120, 252, 3)))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 4)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    # model.add(Dense(32, activation='sigmoid'))
    # model.add(Dropout(0.5))
    model.add(Dense(12, activation='softmax'))
    model.compile(Adam(lr=0.002), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_dataset, epochs=7, shuffle=True)
    
    res = model.evaluate(test_dataset, verbose=0)
    print("test loss:\n", res[0])
    print("test acc:\n", res[1])


def execute_dbn(ds, classifier):
    start_time = time.time()

    mScaler = StandardScaler()
    ds.train_data = mScaler.fit_transform(ds.train_data, ds.train_label)
    ds.test_data = mScaler.fit_transform(ds.test_data, ds.test_label)

    # scaler = MinMaxScaler(copy=True, feature_range=(-1, 1))
    # ds.train_data = scaler.fit_transform(ds.train_data)
    # ds.test_data = scaler.fit_transform(ds.test_data)

    model = classifier

    model.fit(ds.train_data, ds.train_label)

    predicted_test_label = classifier.predict(ds.test_data)
    # predicted_test_label = classifier.evaluate(ds.test_data, ds.test_label)
    # rmse calculation with function
    elapsed_time = time.time() - start_time

    return classifier, predicted_test_label, elapsed_time


def calculate_rmse(test_predictions, test_label, le):
    acc = accuracy_score(test_label, test_predictions)
    test_label = le.inverse_transform(test_label)

    # print("Decoded Test label", test_label)
    test_predictions = le.inverse_transform(test_predictions)
    # print("Decoded prediction label", test_predictions)

    model_mse = mean_squared_error(test_label, test_predictions)
    rmse = np.sqrt(model_mse)
    return rmse, acc


def result_print(predicted_test_label, original_test_label, pred_accuracy, pred_rmse, elapsed_time, model, algorithm,
                 enc):
    log_print('Classifier: ', model)
    if isinstance(model, GridSearchCV):
        log_print("Best params:", model.best_params_)
    log_print("Original  Labels:", enc.inverse_transform(original_test_label))
    log_print("Predicted Labels:", enc.inverse_transform(predicted_test_label))

    ut.append_image_report(algorithm.ljust(40))

    log_print('Prediction Accuracy: ', pred_accuracy)
    ut.append_image_report(str(pred_accuracy).ljust(40))

    log_print('Prediction RMSE: ', pred_rmse)
    ut.append_image_report(str(pred_rmse).ljust(40))

    log_print("Execution Elapsed Time: ", elapsed_time)
    ut.append_image_report(str(elapsed_time).ljust(40))
    # print("-------------------------------------------")
    time.sleep(2)
    print(classification_report(y_pred=predicted_test_label, y_true=original_test_label))


def label_encoder(ds):
    lab_enc = LabelEncoder()
    ds.train_label = lab_enc.fit_transform(ds.train_label)
    ds.test_label = lab_enc.fit_transform(ds.test_label)
    return lab_enc


def plot_image(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if ims.shape[-1] != 3:
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


# def execute_my_cross_val(ds, classifier, k_split):
#     start_time = time.time()
#
#     clf_pipeline = make_pipeline(StandardScaler(), classifier)
#
#     x, y = ds.train_data, ds.train_label
#     # cross validation operations
#     skf = StratifiedKFold(k_split, shuffle=True, random_state=42)
#     scores = np.zeros()
#     for train, test in skf.split(x, y):
#         clf_pipeline.fit(x[train], y[train])
#         # clf_pipeline.
#     predicted_test_label = clf_pipeline.predict(ds.test_data)
#
#     # Elapsed Time
#     elapsed_time = time.time() - start_time
#
#     return clf_pipeline, predicted_test_label, elapsed_time

# ####################################################################################
def data_joiner(ds):
    data = np.concatenate((ds.train_data, ds.test_data), axis=0)
    label = np.concatenate((ds.train_label, ds.test_label), axis=0)
    return data, label


def create_box_plot(data):
    df = pd.DataFrame(data, columns=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8'])
    boxplot = df.boxplot(column=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8'], grid=False)
    plt.show()


def plot_data(ds):
    # isinstance(x, pd.DataFrame):
    df = pd.DataFrame(ds.train_data, columns=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8'])
    df['label'] = pd.DataFrame(ds.train_label)

    for i in range(12):
        lower = i * 7
        upper = lower + 7
        d = df.iloc[lower:upper, 0:8]

        d.boxplot(column=['bin1', 'bin2', 'bin3', 'bin4', 'bin5', 'bin6', 'bin7', 'bin8'], grid=False)

        plt.title("{}th class".format(i))

        plt.show()


# Isolation forest
def outlier_detection(algorithm, ds):
    start_time = time.time()
    # The algorithm fits the data and finds anomaly an removes them from data set
    algorithm.fit(ds.train_data)
    train_prediction = algorithm.predict(ds.train_data)
    test_prediction = algorithm.predict(ds.test_data)

    train_outlier = []
    test_outlier = []

    for i in range(len(train_prediction)):
        out = train_prediction[i]
        if out == -1:
            train_outlier.append(i)
            ds.train_data = np.delete(ds.train_data, i, 0)
            ds.train_label = np.delete(ds.train_label, i, 0)

    for i in range(len(test_prediction)):
        out = test_prediction[i]
        if out == -1:
            test_outlier.append(i)
            ds.test_data = np.delete(ds.test_data, i, 0)
            ds.test_label = np.delete(ds.test_label, i, 0)

    print("\nOutlier Indexes:", train_outlier, " - ", test_outlier)

    elapsed_time = time.time() - start_time
    print("\nElapsed Time:", elapsed_time, "sec.\n")
    return ds


def voting_classifiers(ds):
    from sklearn.ensemble import VotingClassifier
    from sklearn.svm import SVC
    gbc_clf = Algorithms.MyGBC.one_classifier
    ada_clf = Algorithms.MyABC.one_classifier
    rnd_clf = Algorithms.MyRFC.one_classifier
    svm_clf = Algorithms.MySVC.one_classifier

    bnb_clf = Algorithms.MyBNB.one_classifier
    dt_clf = Algorithms.MyDT.one_classifier
    gnb_clf = Algorithms.MyGNB.one_classifier
    knn_clf = Algorithms.MyKNC.one_classifier

    voting_clf = VotingClassifier(
        estimators=[('ada_clf', ada_clf),
                    ('rf', rnd_clf),
                    ('svc', svm_clf),
                    ('gbc_clf', gbc_clf),
                    ('bnb_clf', bnb_clf),
                    ('dt_clf', dt_clf),
                    ('gnb_clf', gnb_clf),
                    ('knn_clf', knn_clf)
                    ],
        voting='soft'
    )
    voting_clf.fit(ds.train_data, ds.train_label)

    for clf in (gbc_clf, ada_clf, rnd_clf, svm_clf, bnb_clf, dt_clf, gnb_clf, knn_clf, voting_clf):
        clf.fit(ds.train_data, ds.train_label)
        y_pred = clf.predict(ds.test_data)
        print(clf.__class__.__name__, accuracy_score(ds.test_label, y_pred))
        # print(clf.__class__.__name__, y_pred)


def one_vs_rest(ds, classifier, le, k_split):
    ovrc = OneVsRestClassifier(classifier)
    clf_pipeline = make_pipeline(StandardScaler(), ovrc)

    cv_scores = cross_val_score(clf_pipeline, ds.train_data, ds.train_label, cv=k_split)

    clf_pipeline.fit(ds.train_data, ds.train_label)
    s = clf_pipeline.score(ds.test_data, ds.test_label)

    p = clf_pipeline.predict(ds.test_data)
    print("CV Accuracy: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
    print("Score:", s)
    df = pd.DataFrame()
    df['Original'] = le.inverse_transform(ds.test_label)
    df_pred = le.inverse_transform(p)
    df['Predicted'] = df_pred

    # print(classifier.best_params_)

    # print results
    print(df)


# K_means
def un_supervised_learning(ds):
    k_means = KMeans(n_clusters=2, random_state=0).fit(ds.train_data)
    pred_train_label = k_means.labels_
    print("OLD Train data size:", len(ds.train_data), " OLD Train Label size:", len(ds.train_label), "\n")
    print("K_Means Train Labels:\n", pred_train_label)

    train_outliers = []
    for i in range(len(pred_train_label)):
        if pred_train_label[i] == 1:
            train_outliers.append(i)

    train_outliers = train_outliers[::-1]
    # print(train_outliers)
    for i in range(len(train_outliers)):
        ds.train_data = np.delete(ds.train_data, train_outliers[i], 0)
        ds.train_label = np.delete(ds.train_label, train_outliers[i], 0)
    print("\nNEW Train data size:", len(ds.train_data), " NEW Train Label size:", len(ds.train_label))

    k_means_2 = KMeans(n_clusters=2, random_state=0).fit(ds.test_data)
    pred_test_label = k_means_2.labels_
    print("\nOLD Test data size:", len(ds.test_data), " OLD Test Label size:", len(ds.test_label), "\n")
    print("K_Means Test Labels:\n", pred_test_label)

    test_outliers = []
    for i in range(len(pred_test_label)):
        if pred_test_label[i] == 1:
            test_outliers.append(i)

    test_outliers = test_outliers[::-1]
    # print(train_outliers)
    for i in range(len(test_outliers)):
        ds.test_data = np.delete(ds.test_data, test_outliers[i], 0)
        ds.test_label = np.delete(ds.test_label, test_outliers[i], 0)
    print("\nNEW Train data size:", len(ds.test_data), " NEW Train Label size:", len(ds.test_label))

    return ds


def shuffle_data(data):
    # print(train_label)
    shuffled_data, shuffle_label = utils.shuffle(data.train_data, data.train_label, random_state=42)
    # print(shuffled_label)
    # data_df = pd.DataFrame(shuffled_data)
    # label_df = pd.DataFrame(shuffled_label)
    # print(label_df)

    # print(len(data_df), len(label_df))
    # print(42*0.8)

    x_train = shuffled_data
    y_train = shuffle_label

    return x_train, y_train


def gaussian_dist(test_data):
    p = []
    for i in range(len(test_data)):
        mu, sigma = np.mean(test_data), np.std(test_data[:, i])

        # Create the bins and histogram1
        count, bins, ignored = plt.hist(test_data[:, i], 8, density=True)

        p_temp = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- ((bins - mu) ** 2) / (2 * sigma ** 2))
        print(p_temp)
        p.append(p_temp)
        # Plot the distribution curve
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- ((bins - mu) ** 2) / (2 * sigma ** 2)), linewidth=4, color='b')

        plt.show()


def detect_outlier(data_1):
    w, h = len(data_1[:, 0]), 8
    outliers = [[0 for x in range(w)] for y in range(h)]

    threshold = 3
    for i in range(8):
        # Create the bins and histogram1

        mean_1 = np.mean(data_1[:, i])
        std_1 = np.std(data_1[:, i])
        max_1 = np.amax(data_1[:, i])
        min_1 = np.amin(data_1[:, i])
        # p_temp = 1 / (std_1 * np.sqrt(2 * np.pi)) * np.exp(- ((8 - mean_1) ** 2) / (2 * std_1 ** 2))

        # for j in range(len(data_1[:, i])):
        #     y = data_1[j][i]
        #     z_score = (y - mean_1) / std_1
        #     if np.abs(z_score) > threshold:
        #         print(i, j, y)

        sorted(data_1[:, i])
        q1, q3 = np.percentile(data_1[:, i], [25, 75])
        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        data = (std_1, mean_1, upper_bound, lower_bound)
        fig1, ax1 = plt.subplots()
        t = "{} th bin".format(i)
        ax1.set_title(t)
        ax1.boxplot(data)

        plt.show()

        print(i, "th feature lower bound", lower_bound)
        print(i, "th feature upper bound", upper_bound)
