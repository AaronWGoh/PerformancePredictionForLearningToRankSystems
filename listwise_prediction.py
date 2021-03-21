import errno
import os
import sys
import time
from itertools import chain, repeat, islice
from optparse import OptionParser
from io import StringIO
# from torch.utils.data import DataLoader

import lightgbm as lgb
import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
import joblib

import kerastuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Lambda, dot, Activation, concatenate
from torch.utils.data import TensorDataset
from sklearn.model_selection import PredefinedSplit
import sklearn

from kerastuner.tuners import RandomSearch, Hyperband
from kerastuner import HyperParameters

import transformer

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from hyperband import HyperbandSearchCV

import eval
import kldiv_metric
import mse_metric
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
import math

np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=1000, precision=5)


class redirected_stdout:
    def __init__(self):
        self._stdout = None
        self._string_io = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._string_io = StringIO()
        return self

    def __exit__(self, type, value, traceback):
        sys.stdout = self._stdout

    @property
    def string(self):
        return self._string_io.getvalue()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def attention_3d_block(hidden_states, dense_layer_size):
    """
    Many-to-one attention mechanism for Keras.
    @param dense_layer_size: layer size for the attention vector output
    @param hidden_states: 3D tensor with shape (batch_size, time_steps, input_dim).
    @return: 2D tensor with shape (batch_size, dense_layer_size)
    """
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(dense_layer_size, use_bias=False, activation='tanh', name='attention_vector')(
        pre_activation)
    return attention_vector


def _validate_flags(features_dir, predictions_dir):
    return features_dir and predictions_dir and \
           os.path.isabs(features_dir) and os.path.isabs(predictions_dir)


def _confirm_file_exists(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def filter_data(X_nested, labels, scores, num_doc=10):
    """
    Removes query-documents with less than num_doc

    :param scores: list of NDCG scores of the y values on a document-query list
    :param X_nested: X_nested is a dictionary where key is a query and the value is the vector representing
    document-query data More specifically, the document-query data has N rows where N is # of documents with
    associated query and M columns where the first M-1 columns are document-query pair features and the last Mth
    column is the score
    :param labels: labels the file for initial relevance values
    :return:
    """
    for query in X_nested.copy().keys():
        if len(X_nested[query]) < num_doc:
            del X_nested[query]
            del labels[query]
            del scores[query]


@timeit
def return_data(dense_feature_filename, predict_filename, eval_metrics=None, n=10, add_noise=True, sigma=0.1):
    """
    Convert data into vectors for deep learning input and returns X, y

    :param add_noise: whether to add noise based on https://openreview.net/pdf?id=Ut1vF_q_vC
    :param n: eval metric up until n
    :param eval_metrics: Set of evaluation metric for y (NDCG, Precision, RR, ERR)
    :param dense_feature_filename: initial yahoo data (densified)
    :param predict_filename: score prediction from LTR algorithm (i.e. RankSVM)
    :return: X_nested, y_nested
    X_nested is a dictionary where key is a query and the value is the vector representing document-query data
        More specifically, the document-query data has N rows where N is # of documents with associated query
        and M columns where the first M-1 columns are document-query pair features and the last Mth column is the score

    y_nested is a dictionary with key being the query id and the value being the NDCG score from the query-document
    scores
    """
    if eval_metrics is None:
        eval_metrics = {"NDCG", "RR", "Precision", "ERR"}

    X_nested, y_nested_by_metric = {}, {}

    scores = {}  # Key is query, value is (score_prediction_from_LTR_alg, document_id)
    sort_scores = {}  # sorted version of scores (sorts document within query)
    labels = {}
    print("---------START " + dense_feature_filename + " DATA READ MANIPULATION----------")
    print("---------DENSE FEATURE FILE DIR:", dense_feature_filename + "---------")
    print("---------PREDICT FILE DIR:", predict_filename + "---------")

    dense_file = open(dense_feature_filename)
    predict_file = open(predict_filename)

    prediction_lines = predict_file.readlines()
    feature_lines = dense_file.readlines()
    print("LENGTH OF PREDICTIONS", len(prediction_lines))
    print("LENGTH OF FEATURE LINES", len(feature_lines))
    count, document_count = 0, 0
    last_query = None
    for line in feature_lines:
        value = line.split()
        if len(value) == 0:
            continue
        relevance_label, query, feature_infos = float(value[0]), value[1].split(':')[1], value[2:]
        if last_query and last_query != query:
            document_count = 0

        # document_query_data is array where index should be feature # feature_value is value
        document_query_data = np.zeros(len(value[2:]))
        for feature_info in feature_infos:
            value = feature_info.split(':')
            feature, feature_value = int(value[0]), float(value[1])
            if add_noise:
                document_query_data[feature] = (feature_transformation(float(feature_value), sigma=sigma))
            else:
                document_query_data[feature] = (float(feature_value))

        document_score = float(prediction_lines[count])
        # document_query_data = np.append(document_query_data,
        #                                 document_score)  # Add prediction score to end of document-query features
        if query in X_nested:
            X_nested[query] = np.concatenate((X_nested[query], [document_query_data]), axis=0)
            scores[query] = np.concatenate((scores[query], [(document_score, document_count)]), axis=0)
            labels[query] = np.concatenate((labels[query], [relevance_label]))
        else:
            X_nested[query] = np.array([document_query_data])
            scores[query] = np.array([(document_score, document_count)])
            labels[query] = np.array([relevance_label])

        last_query = query
        count += 1
        document_count += 1

    filter_data(X_nested, labels, scores)  # Filters based on # of documents for X_nested

    doc_num_info = {}
    sorted_X_nested = {}
    for query in scores.keys():
        sort_scores[query] = sorted(scores[query], key=lambda item: -item[0])
        for score, document in sort_scores[query]:
            document = int(document)
            if query not in sorted_X_nested:
                doc_num_info[query] = 1
                sorted_X_nested[query] = np.array([X_nested[query][document,]])
            else:
                doc_num_info[query] += 1
                sorted_X_nested[query] = np.concatenate((sorted_X_nested[query], [X_nested[query][document,]]), axis=0)

    for eval_metric in eval_metrics:
        y_nested_by_metric[eval_metric] = {}

    for query in labels.keys():
        if "NDCG" in eval_metrics:
            y_nested_by_metric["NDCG"][query] = eval.ndcg_at_k(labels[query], n)
        if "Precision" in eval_metrics:
            binarized_labels = eval.binarize_relevance(labels[query])
            sorted_scores = sorted(scores[query], key=lambda item: -item[0])
            indicies_sorted_by_score = [int(item[1]) for item in sorted_scores]
            y_nested_by_metric["Precision"][query] = eval.ranking_precision_score(binarized_labels,
                                                                                  indicies_sorted_by_score,
                                                                                  n)
        if "RR" in eval_metrics:
            y_nested_by_metric["RR"][query] = eval.reciprocal_rank(labels[query], n)
        if "ERR" in eval_metrics:
            y_nested_by_metric["ERR"][query] = eval.err(labels[query], n)

    dense_file.close()
    predict_file.close()

    print("Y METRIC", y_nested_by_metric)
    print("---------END " + dense_feature_filename + " DATA READ MANIPULATION----------")
    return sorted_X_nested, y_nested_by_metric, doc_num_info


# noinspection PyPep8Naming
@timeit
def write_data_to_file(x_nested, y_nested_by_metric, x_output_file, y_output_file, n=10):
    """
    Writes out the data to the two files

    :param x_output_file: file to output the X data
        File outputed as follows:
        Q:<query_id>
        1.4 2.21 3.32 ....
        .....
        4.2 2.4 9.32 ....
        Q:<query_id>
        ....

    :param y_output_file: file (the evaluation metric appended at the end) to output the y data
    :param x_nested: X data to input
    :param y_nested_by_metric: y data to input
    :return:
    """
    print("---------START " + x_output_file + " DATA WRITE----------")
    _confirm_file_exists(x_output_file)
    _confirm_file_exists(y_output_file)

    X_writer = open(x_output_file, "w")
    y_writers = {}
    if y_output_file[-4:] == '.txt':
        y_output_file = y_output_file[:-4]

    for key in y_nested_by_metric.keys():
        file_end = "_" + key + "_at_" + str(n) + ".txt"
        y_writers[key] = open(y_output_file + file_end, "w")

    count, num_docs = len(x_nested), 0
    for query, rows in x_nested.items():
        X_writer.write("Q:" + str(query) + "\n")
        for row in rows:
            for value in row[:-1]:
                X_writer.write(str(value) + ' ')
            X_writer.write(str(row[-1]) + '\n')

        for key, y_labels in y_nested_by_metric.items():
            if count == num_docs - 1:
                y_writers[key].write('Q:' + str(query) + ' ' + str(y_labels[query]))
                y_writers[key].close()
            else:
                y_writers[key].write('Q:' + str(query) + ' ' + str(y_labels[query]) + "\n")
        count += 1
    X_writer.close()
    print("---------END " + x_output_file + " DATA WRITE----------")


@timeit
def read_data_from_file(x_input_file, y_input_file, eval_metrics=None, k=10):
    """
    Reads data from file created in method write_data_to_file

    :param eval_metrics:
    :param x_input_file: File containing X_nested data
    :param y_input_file: File containing y_nested data
    :return: X_nested, y_nested
    """
    print("---------START  " + x_input_file + " DATA READ----------")
    if eval_metrics is None:
        eval_metrics = {"NDCG", "RR", "Precision", "ERR"}

    X_reader = open(x_input_file, "r")
    X_nested = {}
    doc_num_info = {}
    for line in X_reader.readlines():
        if len(line.split(':')) > 1:
            query = line.split(':')[1].strip()
        else:
            raw_feature_data = np.array(line.split(' '))
            feature_data = raw_feature_data.astype(np.float)
            if query not in X_nested:
                doc_num_info[query] = 1
                X_nested[query] = np.array([feature_data])
            else:
                X_nested[query] = np.concatenate((X_nested[query], [feature_data]), axis=0)
                doc_num_info[query] += 1

    y_nested = {}
    y_readers = {}
    if y_input_file[-4:] == '.txt':
        y_input_file = y_input_file[:-4]
    for eval_metric in eval_metrics:
        end_file = "_" + eval_metric + "_at_" + str(k) + ".txt"
        print("Y FILE DIR READ:", y_input_file + end_file)
        y_readers[eval_metric] = open(y_input_file + end_file, "r")
        y_nested[eval_metric] = {}

    for key, y_reader in y_readers.items():
        for line in y_reader.readlines():
            value = line.split()
            query_info, y_value = value[0], float(value[1])
            query = query_info.split(':')[1].strip()
            y_nested[key][query] = y_value
        y_reader.close()

    X_reader.close()
    print("---------END " + x_input_file + " DATA READ----------")
    return X_nested, y_nested, doc_num_info


def _pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def _pad(iterable, size, padding=None):
    return islice(_pad_infinite(iterable, padding), size)


@timeit
def _limit_sorted_entries(x_as_list, num_limit, orig_length=None):
    X = x_as_list.copy()
    res = []
    for index, query_documents in enumerate(X):
        res.append(np.array(query_documents[:num_limit]))
        if orig_length and orig_length[index] >= num_limit:
            orig_length[index] = num_limit

    return np.array(res)


@timeit
def pad_and_flatten_data(x_nested, max_length):
    """
    Flattens X_nested by query and pads remaining positions by max_length

    :param x_nested:
    :param max_length: longest possible length of a flattened document-query pair batch
    :return: Returns flattened array
    """
    converted_X_nested = []
    for query, rows in x_nested.items():
        padded_row = np.array(list(_pad(rows.flatten(), max_length, 0.0)))
        converted_X_nested.append(padded_row)

    converted_X_nested = np.array(converted_X_nested).astype(float)
    return converted_X_nested


@timeit
def pad_rows(x_nested, max_number_document_on_query):
    new_dict = {}
    original_lengths = []
    for query, rows in x_nested.items():
        new_rows = rows
        original_lengths.append(len(rows))
        while len(new_rows) < max_number_document_on_query:
            new_rows = np.concatenate((new_rows, [np.zeros(rows.shape[1])]), axis=0)
        new_dict[query] = new_rows

    return new_dict, original_lengths


@timeit
def convert_dictionary_to_array(dict_to_convert, reverse=False):
    if reverse:
        new_arr = [np.flip(np.array(rows), 1) for rows in dict_to_convert.values()]
    else:
        new_arr = [np.array(rows) for rows in dict_to_convert.values()]
    new_arr = np.array(new_arr)
    return new_arr


@timeit
def convert_array_to_tensor(array_to_convert):
    return tf.convert_to_tensor(array_to_convert, dtype=tf.float32)


def feature_transformation(element, sigma=0.1):
    """ Transforms feature based on Paper (https://openreview.net/pdf?id=Ut1vF_q_vC)"""
    mu = 0
    return np.log(1 + np.abs(element)) * np.sign(element) + np.random.normal(mu, sigma)


@timeit
def _append_one_hot_encode(x, train_orig_length):
    x = x.copy()
    depth = tf.cast(x.shape[1], tf.int32)
    final_result = [None for _ in range(x.shape[0])]
    i = 0
    for index, query_document_pairs in enumerate(x):
        indicies = tf.constant(query_document_pairs[:train_orig_length[index], -1], dtype=tf.int32)  # -1 for position
        one_hot_array = tf.one_hot(indicies, depth, dtype=tf.float32).numpy()
        offset = len(query_document_pairs) - train_orig_length[index]
        pad = np.zeros((offset, one_hot_array.shape[1]))
        one_hot_array = np.concatenate((one_hot_array, pad))
        result = np.concatenate((query_document_pairs, one_hot_array), axis=1)
        final_result[i] = result
        i += 1

    return final_result


@timeit
def _evaluate_model(x_test, y_test, model, listwise_metric, verbose=True, filewrite=True,
                    dir='model_outputs', filename='model_output', loss=None, scores=None, results=None,
                    cluster_num=None, non_keras=False):
    _confirm_file_exists(dir)
    _confirm_file_exists(dir + '/' + filename)
    if scores is None:
        scores = model.evaluate(x_test, np.asarray(y_test))
    if results is None:
        results = model.predict(x_test)
    predictions = []
    has_metric = True
    for index, result in enumerate(results):
        if isinstance(result, (list, np.ndarray)):
            y_pred = result[0]
        else:
            y_pred = result
        if np.isinf(y_pred) or np.isnan(y_pred):
            has_metric = False
        predictions.append(y_pred)

    predictions = np.array(predictions).flatten()
    model_name = filename.split('_')[0]
    if verbose:
        if model and not non_keras:
            print(model.metrics_names)
        if scores != -1:
            print(scores)
        print("RESULTS")
    if loss == 'mse':
        dir += '/mse'
    elif loss == 'kullback_leibler_divergence':
        dir += '/kld'
    if filewrite:
        with open(dir + '/' + filename + '.txt', "a+") as text_file:
            print(dir + '/' + filename)
            if cluster_num is not None:
                print("LISTWISE METRIC:\t" + listwise_metric + "\t" + cluster_num, file=text_file)
                print("Average Y_Prediction:\t{}".format(np.average(predictions)), file=text_file)
                print("Std Y_Prediction:\t{}".format(np.std(predictions)), file=text_file)
                print("Average Y_test:\t{}".format(np.average(y_test)), file=text_file)
                print("Std Y_test:\t{}".format(np.std(y_test)), file=text_file)
            else:
                print("LISTWISE METRIC:\t" + listwise_metric, file=text_file)
            if model and not non_keras:
                for i in range(len(model.metrics_names)):
                    if model.metrics_names[i] != 'loss' and scores != -1:
                        print(model.metrics_names[i] + ":\t{}".format(scores[i]), file=text_file)
            if has_metric:
                try:
                    r2_res = r2_score(y_test, predictions)
                except:
                    print("Y TEST", y_test)
                    print("PREDICTIONS", predictions)
                if r2_res == 0:
                    print("Y_TEST", y_test)
                    print("PREDICTIONS", predictions)
                print("R2 Score:\t{}".format(r2_res), file=text_file)
                print("RMSE:\t{}".format(math.sqrt(mean_squared_error(y_test, predictions))), file=text_file)
                print("MAE:\t{}".format(math.sqrt(mean_absolute_error(y_test, predictions))), file=text_file)
            print("\n", file=text_file)


def custom_asymmetric_train(y_true, y_pred):
    y_true = 0.000001 if y_true == 0 else y_true
    residual = y_true * tf.log(y_true / y_pred)
    grad = -2 * residual
    hess = 2.0
    return grad, hess


def custom_asymmetric_valid(y_true, y_pred):
    residual = (y_true - y_pred).astype("float")
    loss = np.where(residual < 0, (residual ** 2) * 10.0, residual ** 2)
    return "custom_asymmetric_eval", np.mean(loss), False


def _light_gbm_parameter_tuning(hp):
    lgb_model = lgb.LGBMRegressor(num_leaves=hp.Int('num_leaves', min_value=6, max_value=50, step=4),
                                  min_child_samples=hp.Int('min_child_samples', min_value=100, max_value=500, step=20),
                                  min_child_weight=hp.Choice('min_child_weight', values=[1e-5, 1e-3, 1e-2, 1e-1, 1.]),
                                  max_depth=hp.Int('max_depth', min_value=10, max_value=100, step=10),
                                  reg_alpha=hp.Choice('reg_alpha', values=[1e-5, 1e-3, 1e-2, 1e-1, 0., 1.]),
                                  reg_lambda=hp.Choice('reg_alpha', values=[1e-5, 1e-3, 1e-2, 1e-1, 0., 1.]))
    return lgb_model


@timeit
def light_gbm_model(x_train_nested, y_train_nested, x_test_nested, y_test_nested, x_valid_nested, y_valid_nested,
                    total_num_features_per_row, listwise_metric, loss='mse', metrics=None, verbose=True,
                    dir='model_outputs', operational_dir='operations'):
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler']
    print("LISTWISE METRIC", listwise_metric)
    flattened_X_train = pad_and_flatten_data(x_train_nested, total_num_features_per_row)
    flattened_X_test = pad_and_flatten_data(x_test_nested, total_num_features_per_row)
    print("LIGHT GBM X_TRAIN", flattened_X_train.shape)
    print("LIGHT GBM X_TEST SHAPE", flattened_X_test.shape)
    y_train = np.array([value for value in y_train_nested.values()])
    y_test = np.array([value for value in y_test_nested.values()])

    model_name = 'light_gbm_'
    try:
        print("ATTEMPTS TO LOAD MODEL")
        model = joblib.load(operational_dir + '/models/' + 'light_gbm_' + loss + '_' + listwise_metric + '.h5')
    except (OSError, IOError, EOFError):
        print("FAILS TO LOAD MODEL")
        hp = HyperParameters()
        hp.Choice('total_num_features_per_row', values=[total_num_features_per_row])
        hp.Choice('loss', values=[loss])
        flattened_X_valid = pad_and_flatten_data(x_valid_nested, total_num_features_per_row)
        y_valid = np.array([value for value in y_valid_nested.values()])

        X_combined = np.append(flattened_X_train, flattened_X_valid, 0)
        y_combined = np.append(y_train, y_valid, 0)
        print("X COMB SHAPE", X_combined.shape)
        print("Y COMB SHAPE", y_combined.shape)
        combined_indices = [0 for _ in range(flattened_X_train.shape[0])]
        valid_indices = [1 for _ in range(flattened_X_valid.shape[0])]
        combined_indices.extend(valid_indices)
        tuner = kt.tuners.Sklearn(
            oracle=kt.oracles.Hyperband(
                hyperparameters=hp,
                objective=kt.Objective('score', direction='min'),
                max_epochs=20,
                factor=3),
            cv=PredefinedSplit(combined_indices),
            scoring=sklearn.metrics.make_scorer(mean_squared_error),
            hypermodel=_light_gbm_parameter_tuning,
            directory=operational_dir + '/hyperparameter_tuning',
            project_name=model_name + listwise_metric + '_' + loss + '_project',
            overwrite=False)
        tuner.search(X_combined, y_combined)
        model = tuner.get_best_models(num_models=1)[0]
        filename = dir + '/hyperparameter_tuning/' + model_name + listwise_metric + '_' + loss + '_file.txt'
        _confirm_file_exists(filename)
        with open(filename, "w") as text_file:
            print("Results:", file=text_file)
            with redirected_stdout() as out:
                tuner.results_summary()
                result = out.string
            print(result, file=text_file)
            print('\n', file=text_file)

        print()
        print(model_name + listwise_metric + '_' + loss)
        print("Hyperparam Results:")
        tuner.results_summary()
        print()
        joblib.dump(model, operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')

    results = np.array(model.predict(flattened_X_test))
    calculate_based_on_cluster(flattened_X_test, y_test, results, listwise_metric, 'light_gbm', loss,
                               dir=dir)
    _evaluate_model(flattened_X_test, y_test, model, listwise_metric,
                    filename=model_name + loss,
                    loss=loss, verbose=verbose, dir=dir,
                    results=results, scores=-1, non_keras=True)


@timeit
def feed_forward_model(x_train_nested, y_train_nested, x_test_nested, y_test_nested, x_valid_nested, y_valid_nested,
                       total_num_features_per_row, listwise_metric, losses=None, metrics=None, dir='model_outputs',
                       operational_dir='operations'):
    if losses is None:
        losses = ['mse', 'kullback_leibler_divergence']
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler_divergence']
    flattened_X_train = pad_and_flatten_data(x_train_nested, total_num_features_per_row)
    flattened_X_test = pad_and_flatten_data(x_test_nested, total_num_features_per_row)
    flattened_X_val = pad_and_flatten_data(x_valid_nested, total_num_features_per_row)

    print("TRAIN", flattened_X_train.shape)
    print("TEST", flattened_X_test.shape)
    print("VALID", flattened_X_val.shape)

    y_train_list = np.array([value for value in y_train_nested.values()])
    y_test_list = np.array([value for value in y_test_nested.values()])
    y_val_list = np.array([value for value in y_valid_nested.values()])

    print("TRAIN LEN", y_train_list.shape)
    print("TEST LEN", y_test_list.shape)
    print("VAL LEN", y_val_list.shape)

    for loss in losses:
        print("--------------START LOSS", loss + "---------------------")
        _feed_forward_model(flattened_X_train, y_train_list,
                            flattened_X_test, y_test_list,
                            flattened_X_val, y_val_list,
                            total_num_features_per_row,
                            listwise_metric,
                            loss=loss, metrics=metrics,
                            dir=dir, operational_dir=operational_dir)
        print("--------------END LOSS", loss + "---------------------")


def _feed_forward_model_tuning(hp):
    inputs = tf.keras.Input(shape=(hp.Choice('total_num_features_per_row', values=[1]),), name='features')
    x = layers.Dense(units=hp.Int('first_neuron',
                                  min_value=32,
                                  max_value=768,
                                  step=64),
                     activation=hp.Choice('first_activation', values=['softmax', 'relu', 'sigmoid']), name='dense_1')(
        inputs)
    x = layers.Dense(units=hp.Int('second_neuron',
                                  min_value=32,
                                  max_value=768,
                                  step=64),
                     activation=hp.Choice('second_activation', values=['softmax', 'relu', 'sigmoid']), name='dense_2')(
        x)
    x = layers.Dense(units=hp.Int('third_neuron',
                                  min_value=32,
                                  max_value=768,
                                  step=64),
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']), name='dense_3')(x)
    outputs = layers.Dense(1, name='predictions',
                           activation=hp.Choice('output_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate',
                  values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


# inputs = tf.keras.Input(shape=(total_num_features_per_row,), name='features')
# x = layers.Dense(728, activation='softmax', name='dense_1')(inputs)
# x = layers.Dense(128, activation='softmax', name='dense_2')(x)
# x = layers.Dense(64, activation='softmax', name='dense_3')(x)
# outputs = layers.Dense(1, name='predictions', activation='softmax')(x)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#               loss=loss,
#               metrics=['mean_squared_error', 'kullback_leibler_divergence'])
@timeit
def _feed_forward_model(flattened_X_train, y_train, flattened_X_test, y_test,
                        flattened_X_valid, y_valid, total_num_features_per_row, listwise_metric,
                        verbose=True, loss='mse', metrics=None, dir='model_outputs',
                        operational_dir='operations'):
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler_divergence']
    try:
        model = tf.keras.models.load_model(
            operational_dir + '/models/feedforward_' + loss + '_' + listwise_metric + '.h5')
        model.summary()
    except (OSError, IOError):
        hp = HyperParameters()
        hp.Choice('total_num_features_per_row', values=[total_num_features_per_row])
        hp.Choice('loss', values=[loss])
        tuner = Hyperband(
            _feed_forward_model_tuning,
            hyperparameters=hp,
            objective='val_mean_squared_error',
            max_epochs=20,
            factor=3,
            executions_per_trial=1,
            directory=operational_dir + '/hyperparameter_tuning',
            project_name='feedforward_' + listwise_metric + '_' + loss + '_project',
            overwrite=False)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3)
        tuner.search(flattened_X_train, y_train,
                     epochs=20,
                     validation_data=(flattened_X_valid, y_valid),
                     callbacks=[stop_early])
        model = tuner.get_best_models(num_models=1)[0]
        filename = dir + '/hyperparameter_tuning/feedforward_' + listwise_metric + '_' + loss + '_file.txt'
        _confirm_file_exists(filename)
        with open(filename, "w") as text_file:
            print("Results:", file=text_file)
            with redirected_stdout() as out:
                tuner.results_summary()
                result = out.string
            print(result, file=text_file)
            print("\nBest Model Summary:", file=text_file)
            with redirected_stdout() as out:
                model.summary()
                model_summary = out.string
            print(model_summary, file=text_file)
            print('\n', file=text_file)

        print("Best Model Summary:")
        model.summary()
        print()
        print('Feedforward_' + listwise_metric + '_' + loss)
        print("Hyperparam Results:")
        tuner.results_summary()
        print()
        model.save(operational_dir + '/models/feedforward_' + loss + '_' + listwise_metric + '.h5')

    results = model.predict(flattened_X_test)
    _evaluate_model(flattened_X_test, y_test, model, listwise_metric,
                    filename='feedforward_' + loss,
                    loss=loss, verbose=verbose, dir=dir,
                    results=results)
    calculate_based_on_cluster(flattened_X_test, y_test, results, listwise_metric, 'feedforward', loss,
                               dir=dir)

    return model


@timeit
def lstm_model(X_train_nested, y_train_nested, X_test_nested, y_test_nested, X_valid_nested, y_valid_nested,
               max_number_document_on_query, listwise_metric, reverse=True, losses=None, metrics=None,
               dir='model_outputs', with_attention=False, operational_dir='operations'):
    if losses is None:
        losses = ['mse', 'kullback_leibler_divergence']
    if metrics is None:
        metrics = ['mean_squared_error']
    y_train_list = np.array([value for value in y_train_nested.values()])
    y_test_list = np.array([value for value in y_test_nested.values()])
    y_valid_list = np.array([value for value in y_valid_nested.values()])

    X_train_padded, train_orig_length = pad_rows(X_train_nested, max_number_document_on_query)
    X_test_padded, test_orig_length = pad_rows(X_test_nested, max_number_document_on_query)
    X_valid_padded, valid_orig_length = pad_rows(X_valid_nested, max_number_document_on_query)

    X_train_arr = convert_dictionary_to_array(X_train_padded, reverse=reverse)
    X_test_arr = convert_dictionary_to_array(X_test_padded, reverse=reverse)
    X_valid_arr = convert_dictionary_to_array(X_valid_padded, reverse=reverse)

    X_train_arr = _limit_sorted_entries(X_train_arr, 10, train_orig_length)
    X_valid_arr = _limit_sorted_entries(X_valid_arr, 10, valid_orig_length)
    X_test_arr = _limit_sorted_entries(X_test_arr, 10, test_orig_length)

    X_train = convert_array_to_tensor(X_train_arr)
    X_test = convert_array_to_tensor(X_test_arr)
    X_valid = convert_array_to_tensor(X_valid_arr)

    for loss in losses:
        print("--------------START LOSS", loss + "---------------------")
        _lstm_model(X_train, y_train_list, X_test, y_test_list, X_valid, y_valid_list, listwise_metric,
                    loss=loss, metrics=metrics, dir=dir, with_attention=with_attention, operational_dir=operational_dir)
        print("--------------END LOSS", loss + "---------------------")


def _lstm_model_tuning(hp):
    input_shape = (hp.Choice('num_batches', values=[10]), hp.Choice('num_features', values=[700]))
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = layers.LSTM(hp.Int('first_lstm',
                           min_value=32,
                           max_value=768,
                           step=64), return_sequences=True, input_shape=input_shape, name='lstm_1')(i)
    x = layers.LSTM(hp.Int('second_lstm',
                           min_value=32,
                           max_value=768,
                           step=64), input_shape=input_shape, name='lstm_2')(x)
    x = layers.Dense(1, name='dense_1',
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


def _lstm_model_with_attention_tuning(hp):
    input_shape = (hp.Choice('num_batches', values=[10]), hp.Choice('num_features', values=[700]))
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = layers.LSTM(hp.Int('first_lstm',
                           min_value=32,
                           max_value=768,
                           step=64), return_sequences=True, input_shape=input_shape, name='lstm_1')(i)
    x = layers.LSTM(hp.Int('second_lstm',
                           min_value=32,
                           max_value=768,
                           step=64), return_sequences=True, input_shape=input_shape, name='lstm_2')(x)
    x = attention_3d_block(x, hp.Int('output_dense_size', min_value=32,
                                     max_value=768,
                                     step=64))
    x = layers.Dense(1, name='dense_1',
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


# input_shape = (X_train.shape[1], X_train.shape[2])
# i = tf.keras.Input(shape=input_shape, name='main_input')
#
# # model.add(layers.Embedding(input_dim=1000, input_length=inputshape, output_dim=701))
#
# x = layers.LSTM(256, return_sequences=True, input_shape=input_shape, name='lstm_1')(i)
# x = layers.LSTM(256, input_shape=input_shape, name='lstm_2')(x)
# x = layers.Dense(1, name='dense_1')(x)
# model = tf.keras.Model(inputs=[i], outputs=[x])
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#               loss=loss,
#               metrics=[metrics])
#
# model.fit(X_train, y_train,
#           batch_size=128,
#           epochs=20,
#           validation_steps=10,
#           steps_per_epoch=10,
#           validation_data=(X_valid, y_valid))
@timeit
def _lstm_model(X_train, y_train, X_test, y_test, X_valid, y_valid, listwise_metric, verbose=True,
                loss='mse', metrics=None, dir='model_outputs', with_attention=False, operational_dir='operational'):
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler_divergence']
    if with_attention:
        model_name = 'lstm_with_attention_'
    else:
        model_name = 'lstm_'
    try:
        model = tf.keras.models.load_model(
            operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')
        model.summary()
    except (OSError, IOError):
        hp = HyperParameters()
        hp.Choice('num_batches', values=[X_train.shape[1]])
        hp.Choice('num_features', values=[X_train.shape[2]])
        hp.Choice('loss', values=[loss])
        if with_attention:
            tuner = Hyperband(
                _lstm_model_with_attention_tuning,
                hyperparameters=hp,
                objective='val_mean_squared_error',
                max_epochs=20,
                factor=3,
                executions_per_trial=1,
                directory=operational_dir + '/hyperparameter_tuning',
                project_name=model_name + listwise_metric + '_' + loss + '_project',
                overwrite=False)
        else:
            tuner = Hyperband(
                _lstm_model_tuning,
                hyperparameters=hp,
                objective='val_mean_squared_error',
                max_epochs=20,
                factor=3,
                executions_per_trial=1,
                directory=operational_dir + '/hyperparameter_tuning',
                project_name=model_name + listwise_metric + '_' + loss + '_project',
                overwrite=False)

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3)
        tuner.search(X_train, y_train,
                     epochs=20,
                     validation_data=(X_valid, y_valid),
                     callbacks=[stop_early])
        model = tuner.get_best_models(num_models=1)[0]
        filename = dir + '/hyperparameter_tuning/' + model_name + listwise_metric + '_' + loss + '_file.txt'
        _confirm_file_exists(filename)
        with open(filename, "w") as text_file:
            print("Results:", file=text_file)
            with redirected_stdout() as out:
                tuner.results_summary()
                result = out.string
            print(result, file=text_file)
            print("\nBest Model Summary:", file=text_file)
            with redirected_stdout() as out:
                model.summary()
                model_summary = out.string
            print(model_summary, file=text_file)
            print('\n', file=text_file)

        print("Best Model Summary:")
        model.summary()
        print()
        print('LSTM_' + listwise_metric + '_' + loss)
        print("Hyperparam Results:")
        tuner.results_summary()
        print()

        model.save(operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')

    results = model.predict(X_test)
    _evaluate_model(X_test, y_test, model, listwise_metric,
                    filename=model_name + loss,
                    loss=loss, verbose=verbose, dir=dir,
                    results=results)
    calculate_based_on_cluster(X_test, y_test, results, listwise_metric, model_name[:-1], loss, dir=dir)
    return model


@timeit
def gru_model(X_train_nested, y_train_nested, X_test_nested, y_test_nested, X_valid_nested, y_valid_nested,
              max_number_document_on_query, listwise_metric, reverse=True, losses=None, metrics=None,
              dir='model_outputs', with_attention=False, operational_dir='operational'):
    if losses is None:
        losses = ['mse', 'kullback_leibler_divergence']
    if metrics is None:
        metrics = ['mean_squared_error']
    y_train_list = np.array([value for value in y_train_nested.values()])
    y_test_list = np.array([value for value in y_test_nested.values()])
    y_valid_list = np.array([value for value in y_valid_nested.values()])

    X_train_padded, train_orig_length = pad_rows(X_train_nested, max_number_document_on_query)
    X_test_padded, test_orig_length = pad_rows(X_test_nested, max_number_document_on_query)
    X_valid_padded, valid_orig_length = pad_rows(X_valid_nested, max_number_document_on_query)

    X_train_arr = convert_dictionary_to_array(X_train_padded, reverse=reverse)
    X_test_arr = convert_dictionary_to_array(X_test_padded, reverse=reverse)
    X_valid_arr = convert_dictionary_to_array(X_valid_padded, reverse=reverse)

    X_train_arr = _limit_sorted_entries(X_train_arr, 10, train_orig_length)
    X_valid_arr = _limit_sorted_entries(X_valid_arr, 10, valid_orig_length)
    X_test_arr = _limit_sorted_entries(X_test_arr, 10, test_orig_length)

    X_train = convert_array_to_tensor(X_train_arr)
    X_test = convert_array_to_tensor(X_test_arr)
    X_valid = convert_array_to_tensor(X_valid_arr)

    for loss in losses:
        print("--------------START LOSS", loss + "---------------------")
        _gru_model(X_train, y_train_list, X_test, y_test_list, X_valid, y_valid_list,
                   listwise_metric, loss=loss, metrics=metrics, dir=dir, with_attention=with_attention,
                   operational_dir=operational_dir)
        print("--------------END LOSS", loss + "---------------------")


def _gru_model_tuning(hp):
    input_shape = (hp.Choice('num_batches', values=[10]), hp.Choice('num_features', values=[700]))
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = layers.GRU(hp.Int('first_gru',
                          min_value=32,
                          max_value=768,
                          step=64), return_sequences=True, input_shape=input_shape, name='gru_1')(i)
    x = layers.GRU(hp.Int('second_gru',
                          min_value=32,
                          max_value=768,
                          step=64), input_shape=input_shape, name='gru_2')(x)
    x = layers.Dense(1, name='dense_1',
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


def _gru_model_with_attention_tuning(hp):
    input_shape = (hp.Choice('num_batches', values=[10]), hp.Choice('num_features', values=[700]))
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = layers.GRU(hp.Int('first_gru',
                          min_value=32,
                          max_value=768,
                          step=64), return_sequences=True, input_shape=input_shape, name='gru_1')(i)
    x = layers.GRU(hp.Int('second_gru',
                          min_value=32,
                          max_value=768,
                          step=64), return_sequences=True, input_shape=input_shape, name='gru_2')(x)
    x = attention_3d_block(x, hp.Int('output_dense_size', min_value=32,
                                     max_value=768,
                                     step=64))
    x = layers.Dense(1, name='dense_1',
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


# input_shape = (X_train.shape[1], X_train.shape[2])
# i = tf.keras.Input(shape=input_shape, name='main_input')
# # model.add(layers.Embedding(input_dim=1000, input_length=inputshape, output_dim=701))
#
# x = layers.Bidirectional(layers.GRU(256, return_sequences=True),
#                          input_shape=input_shape, name='gru_1')(i)
# x = layers.Bidirectional(layers.GRU(256, name='gru_2'))(x)
# x = layers.Dense(1, name='dense_1')(x)
# model = tf.keras.Model(inputs=[i], outputs=[x])
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#               loss=loss,
#               metrics=[metrics])
#
# model.fit(X_train, y_train,
#           batch_size=128,
#           epochs=20,
#           validation_steps=10,
#           steps_per_epoch=10,
#           validation_data=(X_valid, y_valid))
@timeit
def _gru_model(X_train, y_train, X_test, y_test, X_valid, y_valid, listwise_metric, verbose=True,
               loss='mse', metrics=None, dir='model_outputs', with_attention=False, operational_dir='operational'):
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler_divergence']
    if with_attention:
        model_name = 'gru_with_attention_'
    else:
        model_name = 'gru_'
    try:
        model = tf.keras.models.load_model(
            operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')
        model.summary()
    except (OSError, IOError):
        hp = HyperParameters()
        hp.Choice('num_batches', values=[X_train.shape[1]])
        hp.Choice('num_features', values=[X_train.shape[2]])
        hp.Choice('loss', values=[loss])
        if with_attention:
            tuner = Hyperband(
                _gru_model_with_attention_tuning,
                hyperparameters=hp,
                objective='val_mean_squared_error',
                max_epochs=20,
                factor=3,
                executions_per_trial=1,
                directory=operational_dir + '/hyperparameter_tuning',
                project_name=model_name + listwise_metric + '_' + loss + '_project',
                overwrite=False)
        else:
            tuner = Hyperband(
                _gru_model_tuning,
                hyperparameters=hp,
                objective='val_mean_squared_error',
                max_epochs=20,
                factor=3,
                executions_per_trial=1,
                directory=operational_dir + '/hyperparameter_tuning',
                project_name=model_name + listwise_metric + '_' + loss + '_project',
                overwrite=False)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3)
        tuner.search(X_train, y_train,
                     epochs=20,
                     validation_data=(X_valid, y_valid),
                     callbacks=[stop_early])
        model = tuner.get_best_models(num_models=1)[0]
        filename = dir + '/hyperparameter_tuning/' + model_name + listwise_metric + '_' + loss + '_file.txt'
        _confirm_file_exists(filename)
        with open(filename, "w") as text_file:
            print("Results:", file=text_file)
            with redirected_stdout() as out:
                tuner.results_summary()
                result = out.string
            print(result, file=text_file)
            print("\nBest Model Summary:", file=text_file)
            with redirected_stdout() as out:
                model.summary()
                model_summary = out.string
            print(model_summary, file=text_file)
            print('\n', file=text_file)

        print("Best Model Summary:")
        model.summary()
        print()
        print(model_name + listwise_metric + '_' + loss)
        print("Hyperparam Results:")
        tuner.results_summary()
        print()

        model.save(operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')

    results = model.predict(X_test)
    _evaluate_model(X_test, y_test, model, listwise_metric,
                    filename=model_name + loss,
                    loss=loss, verbose=verbose, dir=dir,
                    results=results)
    calculate_based_on_cluster(X_test, y_test, results, listwise_metric, model_name[:-1], loss, dir=dir)
    return model


@timeit
def attention_model(X_train_nested, y_train_nested, X_test_nested, y_test_nested, X_valid_nested, y_valid_nested,
                    max_number_document_on_query, listwise_metric, losses=None, metrics=None, dir='model_outputs',
                    operational_dir='operational'):
    if losses is None:
        losses = ['mse', 'kullback_leibler_divergence']
    if metrics is None:
        metrics = ['mean_squared_error']
    y_train_list = np.array([value for value in y_train_nested.values()])
    y_val_list = np.array([value for value in y_valid_nested.values()])
    y_test_list = np.array([value for value in y_test_nested.values()])

    X_train_padded, train_orig_length = pad_rows(X_train_nested, max_number_document_on_query)
    X_valid_padded, valid_orig_length = pad_rows(X_valid_nested, max_number_document_on_query)
    X_test_padded, test_orig_length = pad_rows(X_test_nested, max_number_document_on_query)

    X_train_arr = convert_dictionary_to_array(X_train_padded)
    X_valid_arr = convert_dictionary_to_array(X_valid_padded)
    X_test_arr = convert_dictionary_to_array(X_test_padded)

    X_train_arr = _limit_sorted_entries(X_train_arr, 10, train_orig_length)
    X_valid_arr = _limit_sorted_entries(X_valid_arr, 10, valid_orig_length)
    X_test_arr = _limit_sorted_entries(X_test_arr, 10, test_orig_length)

    X_train_arr = _append_one_hot_encode(X_train_arr, train_orig_length)
    X_valid_arr = _append_one_hot_encode(X_valid_arr, valid_orig_length)
    X_test_arr = _append_one_hot_encode(X_test_arr, test_orig_length)

    X_train = convert_array_to_tensor(X_train_arr)
    X_valid = convert_array_to_tensor(X_valid_arr)
    X_test = convert_array_to_tensor(X_test_arr)

    for loss in losses:
        print("--------------START LOSS", loss + "---------------------")
        _attention_model(X_train, y_train_list, X_test, y_test_list, X_valid, y_val_list, listwise_metric,
                         loss=loss, metrics=metrics, dir=dir, operational_dir=operational_dir)
        print("--------------END LOSS", loss + "---------------------")


def _attention_model_tuning(hp):
    input_shape = (hp.Choice('num_batches', values=[10]), hp.Choice('num_features', values=[700]))
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = attention_3d_block(i, hp.Int('output_dense_size', min_value=32,
                                     max_value=768,
                                     step=64))
    x = layers.Dense(1, name='dense_1',
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


# input_shape = (X_train.shape[1], X_train.shape[2])
# i = tf.keras.Input(shape=input_shape, name='train_input')
# x = attention_3d_block(i)
# x = Dense(1, activation='sigmoid', name='dense_1')(x)
# model = tf.keras.Model(inputs=[i], outputs=[x])
# model.summary()
# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#               loss=loss,
#               metrics=metrics)
#
# model.fit(X_train, y_train,
#           batch_size=128,
#           epochs=20,
#           validation_data=(X_valid, y_valid))
#
# model.save(dir + '/models/attention_' + loss + '_' + listwise_metric + '.h5')
@timeit
def _attention_model(X_train, y_train, X_test, y_test, X_valid, y_valid, listwise_metric, verbose=True,
                     loss='mse', metrics=None, dir='model_outputs', operational_dir='operational'):
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler_divergence']

    try:
        model = tf.keras.models.load_model(
            operational_dir + '/models/attention_' + loss + '_' + listwise_metric + '.h5')
        model.summary()
    except (OSError, IOError):
        hp = HyperParameters()
        hp.Choice('num_batches', values=[X_train.shape[1]])
        hp.Choice('num_features', values=[X_train.shape[2]])
        hp.Choice('loss', values=[loss])
        tuner = Hyperband(
            _attention_model_tuning,
            hyperparameters=hp,
            objective='val_mean_squared_error',
            max_epochs=20,
            factor=3,
            executions_per_trial=1,
            directory=operational_dir + '/hyperparameter_tuning',
            project_name='attention_' + listwise_metric + '_' + loss + '_project',
            overwrite=False)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3)
        tuner.search(X_train, y_train,
                     epochs=20,
                     validation_data=(X_valid, y_valid),
                     callbacks=[stop_early])
        model = tuner.get_best_models(num_models=1)[0]
        filename = dir + '/hyperparameter_tuning/attention_' + listwise_metric + '_' + loss + '_file.txt'
        _confirm_file_exists(filename)
        with open(filename, "w") as text_file:
            print("Results:", file=text_file)
            with redirected_stdout() as out:
                tuner.results_summary()
                result = out.string
            print(result, file=text_file)
            print("\nBest Model Summary:", file=text_file)
            with redirected_stdout() as out:
                model.summary()
                model_summary = out.string
            print(model_summary, file=text_file)
            print('\n', file=text_file)

        print("Best Model Summary:")
        model.summary()
        print()
        print('Attention_' + listwise_metric + '_' + loss)
        print("Hyperparam Results:")
        tuner.results_summary()
        print()

        model.save(operational_dir + '/models/attention_' + loss + '_' + listwise_metric + '.h5')

    results = model.predict(X_test)
    _evaluate_model(X_test, y_test, model, listwise_metric,
                    filename='attention_' + loss,
                    loss=loss, verbose=verbose, dir=dir,
                    results=results)
    calculate_based_on_cluster(X_test, y_test, results, listwise_metric, 'attention', loss, dir=dir)
    return model


@timeit
def transformer_model(X_train_nested, y_train_nested, X_test_nested, y_test_nested, X_valid_nested, y_valid_nested,
                      max_number_document_on_query, listwise_metric, losses=None, metrics=None, dir='model_outputs',
                      with_feedforward=False, operational_dir='operational'):
    if losses is None:
        losses = ['mse', 'kullback_leibler_divergence']
    if metrics is None:
        metrics = ['mean_squared_error']
    y_train_list = np.array([value for value in y_train_nested.values()])
    y_val_list = np.array([value for value in y_valid_nested.values()])
    y_test_list = np.array([value for value in y_test_nested.values()])

    X_train_padded, train_orig_length = pad_rows(X_train_nested, max_number_document_on_query)
    X_valid_padded, valid_orig_length = pad_rows(X_valid_nested, max_number_document_on_query)
    X_test_padded, test_orig_length = pad_rows(X_test_nested, max_number_document_on_query)

    X_train_arr = convert_dictionary_to_array(X_train_padded)
    X_valid_arr = convert_dictionary_to_array(X_valid_padded)
    X_test_arr = convert_dictionary_to_array(X_test_padded)

    X_train_arr = _limit_sorted_entries(X_train_arr, 10, train_orig_length)
    X_valid_arr = _limit_sorted_entries(X_valid_arr, 10, valid_orig_length)
    X_test_arr = _limit_sorted_entries(X_test_arr, 10, test_orig_length)

    X_train = np.array(_append_one_hot_encode(X_train_arr, train_orig_length))
    X_valid = np.array(_append_one_hot_encode(X_valid_arr, valid_orig_length))
    X_test = np.array(_append_one_hot_encode(X_test_arr, test_orig_length))

    print("INITIAL TRAIN DIM:", X_train.shape)
    print("INITIAL VALID DIM:", X_valid.shape)
    print("INITIAL TEST DIM:", X_test.shape)
    initial_num_feature = X_train.shape[2]
    necessary_addition = 0
    while initial_num_feature % 16 != 0:
        necessary_addition += 1
        initial_num_feature += 1

    z_train = np.zeros((X_train.shape[0], X_train.shape[1], necessary_addition))
    z_valid = np.zeros((X_valid.shape[0], X_valid.shape[1], necessary_addition))
    z_test = np.zeros((X_test.shape[0], X_test.shape[1], necessary_addition))

    X_train = np.append(X_train, z_train, axis=2)
    X_valid = np.append(X_valid, z_valid, axis=2)
    X_test = np.append(X_test, z_test, axis=2)

    print("ALTERED TRAIN DIM:", X_train.shape)
    print("ALTERED VALID DIM:", X_valid.shape)
    print("ALTERED TEST DIM:", X_test.shape)
    for loss in losses:
        print("--------------START LOSS", loss + "---------------------")
        _transformer_model(X_train, y_train_list, X_test, y_test_list, X_valid, y_val_list, listwise_metric,
                           loss=loss, dir=dir, metrics=metrics, with_feedforward=with_feedforward,
                           operational_dir=operational_dir)


def _transformer_model_tuning(hp):
    num_feature = hp.Choice('num_features', values=[700])
    input_shape = (hp.Choice('num_batches', values=[10]), num_feature)
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = transformer.get_encoder_component('transformer', i, hp.Choice('num_heads', values=[8, 16]), num_feature,
                                          feed_forward_activation=hp.Choice('feedforward_act',
                                                                            values=['tanh', 'softmax', 'relu',
                                                                                    'sigmoid']))
    x = layers.Flatten()(x)
    x = layers.Dense(1, name='dense_1',
                     activation=hp.Choice('final_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


def _transformer_with_feedforward_tuner(hp):
    num_feature = hp.Choice('num_features', values=[700])
    input_shape = (hp.Choice('num_batches', values=[10]), num_feature)
    i = tf.keras.Input(shape=input_shape, name='main_input')
    x = transformer.get_encoder_component('transformer', i, hp.Choice('num_heads', values=[8, 16]), num_feature,
                                          feed_forward_activation=hp.Choice('feedforward_act',
                                                                            values=['tanh', 'softmax', 'relu',
                                                                                    'sigmoid']))
    x = layers.Dense(units=hp.Int('first_neuron',
                                  min_value=32,
                                  max_value=768,
                                  step=64),
                     activation=hp.Choice('first_activation', values=['softmax', 'relu', 'sigmoid']), name='dense_1')(x)
    x = layers.Dense(units=hp.Int('second_neuron',
                                  min_value=32,
                                  max_value=768,
                                  step=64),
                     activation=hp.Choice('second_activation', values=['softmax', 'relu', 'sigmoid']), name='dense_2')(
        x)
    x = layers.Dense(units=hp.Int('third_neuron',
                                  min_value=32,
                                  max_value=768,
                                  step=64),
                     activation=hp.Choice('third_activation', values=['softmax', 'relu', 'sigmoid']), name='dense_3')(x)
    x = layers.Dense(1, name='fourth_neuron',
                     activation=hp.Choice('fourth_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1, name='predictions',
                     activation=hp.Choice('output_activation', values=['softmax', 'relu', 'sigmoid']))(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=tf.Variable(
        hp.Choice('learning_rate', values=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]))),
        loss=hp.Choice('loss', values=['mse']),
        metrics=['mean_squared_error', 'kullback_leibler_divergence'])
    return model


@timeit
def _transformer_model(X_train, y_train, X_test, y_test, X_valid, y_valid, listwise_metric, verbose=True,
                       loss='mse', metrics=None, dir='model_outputs', with_feedforward=False,
                       operational_dir='operational'):
    if metrics is None:
        metrics = ['mean_squared_error', 'kullback_leibler_divergence']
    if with_feedforward:
        model_name = 'transformer_with_feedforward_'
    else:
        model_name = 'transformer_'
    try:
        model = tf.keras.models.load_model(
            operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')
        model.summary()
    except (OSError, IOError):
        hp = HyperParameters()
        print("X_TRAIN SHAPE", X_train.shape)
        hp.Choice('num_batches', values=[X_train.shape[1]])
        hp.Choice('num_features', values=[X_train.shape[2]])
        hp.Choice('loss', values=[loss])
        if with_feedforward:
            tuner = Hyperband(
                _transformer_with_feedforward_tuner,
                hyperparameters=hp,
                objective='val_mean_squared_error',
                max_epochs=20,
                factor=3,
                executions_per_trial=1,
                directory=operational_dir + '/hyperparameter_tuning',
                project_name=model_name + listwise_metric + '_' + loss + '_project',
                overwrite=False)
        else:
            tuner = Hyperband(
                _transformer_model_tuning,
                hyperparameters=hp,
                objective='val_mean_squared_error',
                max_epochs=20,
                factor=3,
                executions_per_trial=1,
                directory=operational_dir + '/hyperparameter_tuning',
                project_name=model_name + listwise_metric + '_' + loss + '_project',
                overwrite=False)
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=3)
        tuner.search(X_train, y_train,
                     epochs=20,
                     validation_data=(X_valid, y_valid),
                     callbacks=[stop_early])
        model = tuner.get_best_models(num_models=1)[0]
        filename = dir + '/hyperparameter_tuning/' + model_name + listwise_metric + '_' + loss + '_file.txt'
        _confirm_file_exists(filename)
        with open(filename, "w") as text_file:
            print("Results:", file=text_file)
            with redirected_stdout() as out:
                tuner.results_summary()
                result = out.string
            print(result, file=text_file)
            print("\nBest Model Summary:", file=text_file)
            with redirected_stdout() as out:
                model.summary()
                model_summary = out.string
            print(model_summary, file=text_file)
            print('\n', file=text_file)

        print("Best Model Summary:")
        model.summary()
        print()
        print('Attention_' + listwise_metric + '_' + loss)
        print("Hyperparam Results:")
        tuner.results_summary()
        print()

        model.save(operational_dir + '/models/' + model_name + loss + '_' + listwise_metric + '.h5')

    results = model.predict(X_test)
    _evaluate_model(X_test, y_test, model, listwise_metric,
                    filename=model_name + loss,
                    loss=loss, verbose=verbose, dir=dir,
                    results=results)
    calculate_based_on_cluster(X_test, y_test, results, listwise_metric, model_name[:-1], loss, dir=dir)
    return model


# @timeit
# def tensor_dataset(X, y):
#     X = torch.from_numpy(X).float()
#     y = torch.from_numpy(y).float()
#     print("X SIZE:", X.size())
#     print("y SIZE:", y.size())
#     return TensorDataset(X, y)
#
#
# # class CustomTransformer(torch.nn.Module):
# #     def __init__(self, input_size, num_documents, hidden_size_1, hidden_size_2, hidden_size_3, nhead=8):
# #         super(CustomTransformer, self).__init__()
# #         # print("INPUT SIZE:", input_size, "HIDDEN SIZE:", hidden_size)
# #         self.input_size = input_size
# #         self.hidden_size_1 = hidden_size_1
# #         self.hidden_size_2 = hidden_size_2
# #         self.hidden_size_3 = hidden_size_3
# #
# #         self.num_documents = num_documents
# #         self.relu = torch.nn.ReLU()
# #
# #         self.transformer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
# #         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
# #         self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
# #         self.fc3 = torch.nn.Linear(self.hidden_size_2, self.hidden_size_3)
# #         self.fc4 = torch.nn.Linear(self.hidden_size_3, 1)
# #
# #         self.fc5 = torch.nn.Linear(self.num_documents, 1)
# #         # self.fc6 = torch.nn.Linear(self.batch_size, 1)
# #         # self.fc7 = torch.nn.Linear(self.seq_length, 1)
# #
# #     def forward(self, x):
# #         transformer_layer = self.transformer(x)
# #         print("TRANSFORMER LAYER", transformer_layer.shape)
# #         hidden_1 = self.relu(self.fc1(transformer_layer))
# #         hidden_2 = self.relu(self.fc2(hidden_1))
# #         hidden_3 = self.relu(self.fc3(hidden_2))
# #         perceptron_res = self.relu(self.fc4(hidden_3))
# #         print("PERCEPTRON RES SHAPE", perceptron_res.shape)
# #         reshaped = torch.flatten(perceptron_res, start_dim=1)
# #         output = self.relu(self.fc5(reshaped))
# #
# #         return torch.reshape(output, (-1,))
#
# class CustomTransformer(torch.nn.Module):
#     def __init__(self, input_size, num_documents, hidden_size_1, hidden_size_2, hidden_size_3, nhead=8):
#         super(CustomTransformer, self).__init__()
#         # print("INPUT SIZE:", input_size, "HIDDEN SIZE:", hidden_size)
#         self.input_size = input_size
#         self.hidden_size_1 = hidden_size_1
#         self.hidden_size_2 = hidden_size_2
#         self.hidden_size_3 = hidden_size_3
#
#         self.num_documents = num_documents
#         # self.fc1 = torch.nn.Linear()
#         self.softmax = torch.nn.Softmax()
#
#         self.transformer = torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
#         self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
#         self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
#         self.fc3 = torch.nn.Linear(self.hidden_size_2, self.hidden_size_3)
#         self.fc4 = torch.nn.Linear(self.hidden_size_3, 1)
#
#         self.fc5 = torch.nn.Linear(self.num_documents, 1)
# @timeit
# def _transformer_model(X_train, y_train, X_test, y_test, X_valid, y_valid, listwise_metric, verbose=True, loss='mse',
#                        dir='model_outputs'):
#     def loss_batch(model, loss_func, xb, yb, opt=None):
#         loss = loss_func(model(xb), yb)
#         if opt is not None:
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#
#         return loss.item(), len(xb)
#
#     def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
#         for epoch in range(epochs):
#             model.train()
#             total_loss = 0.0
#             for xb, yb in train_dl:
#                 training_loss_for_batch, _ = loss_batch(model, loss_func, xb, yb, opt)
#                 total_loss += training_loss_for_batch
#             # print("Epoch " + str(epoch) + ":", "TOTAL LOSS:", total_loss, "AVG EPOCH LOSS", total_loss / len(train_dl))
#
#             model.eval()
#             with torch.no_grad():
#                 losses, nums = zip(
#                     *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
#                 )
#             val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
#             with tune.checkpoint_dir(epoch) as checkpoint_dir:
#                 path = os.path.join(checkpoint_dir, "checkpoint")
#                 torch.save((net.state_dict(), opt.state_dict()), path)
#             # print("Epoch:", epoch, "Validation Loss:", val_loss)
#
#     def eval(model, test_dl, listwise_metric, verbose=verbose, loss=loss, filewrite=True, dir=dir):
#         model.eval()
#         filename = 'transformer_output_with_perceptron_' + loss
#         predictions = []
#         numeric_predictions = []
#         y_test = []
#         has_metric = True
#         for xb, yb in test_dl:
#             res = model(xb)
#             numpy_res = res.detach().numpy()
#             if np.any(np.isinf(numpy_res)) or np.any(np.isnan(numpy_res)):
#                 has_metric = False
#             y_test.extend(yb)
#             predictions.extend(res)
#             numeric_predictions.extend(numpy_res)
#
#         mean_squared_error = mse_metric.MeanSquaredError()
#         prediction_tensor = torch.Tensor(predictions)
#         actual_tensor = torch.Tensor(y_test)
#         mean_squared_error_res = mean_squared_error(prediction_tensor, actual_tensor)
#         kl_div = kldiv_metric.KLDivergence()
#         kl_div_res = kl_div(prediction_tensor, actual_tensor)
#         calculate_based_on_cluster(X_test, y_test, numeric_predictions, listwise_metric, 'transformer', loss, dir=dir)
#
#         if verbose:
#             print("RESULTS")
#             print(predictions[:10])
#             # print("MEAN SQUARED ERROR:", test_acc)
#             # print("LOSS:", test_loss)
#         if loss == 'mse':
#             dir += '/mse'
#         elif loss == 'kullback_leibler_divergence':
#             dir += '/kld'
#         if filewrite:
#             with open(dir + '/' + filename + '.txt', "a+") as text_file:
#                 print(dir + '/' + filename + '_' + listwise_metric + '.txt')
#
#                 print("LISTWISE:\t{}".format(listwise_metric), file=text_file)
#                 print("mean_squared_error:\t{}".format(mean_squared_error_res), file=text_file)
#                 print("kullback_leibler_divergence:\t{}".format(kl_div_res), file=text_file)
#                 if has_metric:
#                     r2_res = r2_score(y_test, predictions)
#                     if r2_res == 0:
#                         print("Y_TEST", y_test)
#                         print("PREDICTIONS", predictions)
#                     print("R2 Score:\t{}".format(r2_res), file=text_file)
#                     print("RMSE:\t{}".format(math.sqrt(mean_squared_error(y_test, predictions))), file=text_file)
#                     print("MAE:\t{}".format(math.sqrt(mean_absolute_error(y_test, predictions))), file=text_file)
#                 print("\n", file=text_file)
#
#         return predictions
#
#     def get_data(train_ds, valid_ds, test_ds, bs=1):
#         return (
#             DataLoader(train_ds, batch_size=bs),
#             DataLoader(valid_ds, batch_size=bs // 2),
#             DataLoader(test_ds, batch_size=len(test_ds)),
#         )
#
#     if loss == 'kullback_leibler_divergence':
#         print("----------- LOSS: KLD ---------------------")
#         criterion = nn.KLDivLoss()
#     elif loss == 'mse':
#         print("----------- LOSS: MSE ---------------------")
#         criterion = nn.MSELoss()
#     else:
#         print("----------- WARNING: NO LOSS DEFINED -----------------")
#         criterion = nn.MSELoss()
#
#     sequence_length = X_train.shape[0]
#     num_documents_per_query = X_train.shape[1]
#     num_features = X_train.shape[2]
#
#     # (<batch size>, <sequence length>, <input size>) -> (<sequence length>, <batch size>, <input size>)
#     train_dataset = tensor_dataset(X_train, y_train)
#     valid_dataset = tensor_dataset(X_valid, y_valid)
#     test_dataset = tensor_dataset(X_test, y_test)
#
#     config = {
#         "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
#         "lr": tune.loguniform(1e-4, 1e-1),
#         "batch_size": tune.choice([2, 4, 8, 16])
#     }
#     model = CustomTransformer(num_features, num_documents_per_query,
#                               config['hidden_size_1'],
#                               config['hidden_size_2'],
#                               config['hidden_size_3'])
#     # model = CustomTransformer(num_features, num_documents_per_query, 256, nhead=8)
#     # optimizer = optim.SGD([{'params': model.transformer.parameters(), 'lr': 0.0000001}]
#     #                       , lr=0.000000001, momentum=0.0, weight_decay=0.001)
#     optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
#     train_dl, valid_dl, test_dl = get_data(train_dataset, valid_dataset, test_dataset, bs=100)
#     try:
#         model.load_state_dict(torch.load(dir + '/models/attention_' + loss + '_' + listwise_metric + '.h5'))
#         model.eval()
#     except (OSError, IOError):
#         fit(10, model, criterion, optimizer, train_dl, valid_dl)
#         torch.save(model.state_dict(), dir + '/models/attention_' + loss + '_' + listwise_metric + '.h5')
#     eval(model, test_dl, listwise_metric)
#     return model


@timeit
def process_data(dense_input_dir="ltrc_yahoo/set1/",
                 x_output_dir="ltrc_yahoo/set1/ML/",
                 predict_input_dir="ltrc_yahoo/set1/",
                 y_output_dir="ltrc_yahoo/set1/ML/",
                 dataset="train", add_noise=True,
                 eval_metrics=None,
                 k=10):
    if eval_metrics is None:
        eval_metrics = {"NDCG", "ERR", "Precision", "RR"}
    dense_file = dataset + "_dense.txt"
    predict_file = dataset + ".predict"
    y_file_base = "y_" + dataset
    y_files = [y_output_dir + y_file_base + '_' + eval_metric + "_at_" + str(k) + ".txt" for eval_metric in
               eval_metrics]
    print("Y FILES", y_files)
    y_contains = [os.path.exists(y_file) for y_file in y_files]
    if add_noise:
        print("---------START ADD NOISE - PROCESS DATA----------")
        x_file = "X_" + dataset + "_noise.txt"
        if not os.path.exists(x_output_dir + x_file) or not any(y_contains):
            X_nested, y_nested_by_metric, doc_num_info = return_data(dense_input_dir + dense_file,
                                                                     predict_input_dir + predict_file,
                                                                     eval_metrics=eval_metrics,
                                                                     add_noise=True,
                                                                     sigma=0.1,
                                                                     n=k)
            write_data_to_file(X_nested, y_nested_by_metric, x_output_dir + x_file, y_output_dir + y_file_base, n=k)
        else:
            X_nested, y_nested_by_metric, doc_num_info = read_data_from_file(x_output_dir + x_file,
                                                                             y_output_dir + y_file_base, k=k)
        print("---------END ADD NOISE - PROCESS DATA----------")
    else:
        print("---------START NO NOISE - PROCESS DATA----------")
        x_file = "X_" + dataset + ".txt"
        if not os.path.exists(x_output_dir + x_file) or not any(y_contains):
            X_nested, y_nested_by_metric, doc_num_info = return_data(dense_input_dir + dense_file,
                                                                     predict_input_dir + predict_file,
                                                                     eval_metrics=eval_metrics,
                                                                     add_noise=False,
                                                                     n=k)
            write_data_to_file(X_nested, y_nested_by_metric, x_output_dir + x_file, y_output_dir + y_file_base, n=k)
        else:
            X_nested, y_nested_by_metric, doc_num_info = read_data_from_file(x_output_dir + x_file,
                                                                             y_output_dir + y_file_base, k=k)
        print("---------END NO NOISE - PROCESS DATA----------")
    return X_nested, y_nested_by_metric, doc_num_info


def get_flattened_row_size(X_train_nested, doc_num_info):
    num_features = next(iter(X_train_nested.values())).shape[1]
    average_num_doc = sum(doc_num_info.values()) // len(doc_num_info.keys())
    max_number_document_on_query = int(average_num_doc)
    flattened_row_size = int(max_number_document_on_query * num_features)
    print()
    print("AVG NUM DOC ON QUERY:", sum(doc_num_info.values()) / len(doc_num_info.keys()))
    print("MIN DOC NUM INFO:", min(doc_num_info.values()))
    print("MAX DOC NUM INFO:", max(doc_num_info.values()))
    print()
    print("NUM FEATURES:", num_features)
    print("MAX NUM OF DOCUMENTS ON QUERY:", max_number_document_on_query)
    print("FLATTENED ROW SIZE:", flattened_row_size)
    print()

    return flattened_row_size, max_number_document_on_query


def process_models(X_train_nested, y_train_nested_by_metric,
                   X_valid_nested, y_valid_nested_by_metric,
                   X_test_nested, y_test_nested_by_metric,
                   doc_num_info, dir='model_outputs', operational_dir='operations',
                   eval_metrics=None, model_mode="Transformer",
                   run_all=True):
    flattened_row_size, max_number_document_on_query = get_flattened_row_size(X_train_nested, doc_num_info)
    _confirm_file_exists(dir)
    _confirm_file_exists(dir + '/mse')
    _confirm_file_exists(dir + '/kld')
    tf.compat.v1.enable_eager_execution()
    if eval_metrics is None:
        eval_metrics = {"RR", "NDCG", "Precision", "ERR"}
    loss = None
    # loss = ['kullback_leibler_divergence', 'mse']
    metrics = ['mean_squared_error', 'kullback_leibler_divergence']
    for listwise_eval_metric in eval_metrics:
        y_train_nested = y_train_nested_by_metric[listwise_eval_metric]
        y_test_nested = y_test_nested_by_metric[listwise_eval_metric]
        y_valid_nested = y_valid_nested_by_metric[listwise_eval_metric]
        if model_mode.lower() == "feedforward" or run_all:
            print("--------------START FEEDFORWARD---------------------")
            feed_forward_model(X_train_nested, y_train_nested,
                               X_test_nested, y_test_nested,
                               X_valid_nested, y_valid_nested,
                               flattened_row_size,
                               listwise_eval_metric,
                               losses=loss,
                               metrics=metrics,
                               dir=dir,
                               operational_dir=operational_dir)
            print("--------------END FEEDFORWARD---------------------")
        if model_mode.lower() == "lstm" or run_all:
            print("--------------START LSTM---------------------")
            lstm_model(X_train_nested, y_train_nested,
                       X_test_nested, y_test_nested,
                       X_valid_nested, y_valid_nested,
                       max_number_document_on_query,
                       listwise_eval_metric,
                       losses=loss,
                       metrics=metrics,
                       dir=dir,
                       operational_dir=operational_dir)
            print("--------------END LSTM---------------------")
        if model_mode.lower() == "lstm_with_attention" or run_all:
            print("--------------START LSTM WITH ATTENTION---------------------")
            lstm_model(X_train_nested, y_train_nested,
                       X_test_nested, y_test_nested,
                       X_valid_nested, y_valid_nested,
                       max_number_document_on_query,
                       listwise_eval_metric,
                       losses=loss,
                       metrics=metrics,
                       dir=dir, with_attention=True,
                       operational_dir=operational_dir)
            print("--------------END LSTM WITH ATTENTION---------------------")
        if model_mode.lower() == "gru" or run_all:
            print("--------------START GRU---------------------")
            gru_model(X_train_nested, y_train_nested,
                      X_test_nested, y_test_nested,
                      X_valid_nested, y_valid_nested,
                      max_number_document_on_query,
                      listwise_eval_metric,
                      losses=loss,
                      metrics=metrics,
                      dir=dir,
                      operational_dir=operational_dir)
            print("--------------END GRU---------------------")
        if model_mode.lower() == "gru_with_attention" or run_all:
            print("--------------START GRU---------------------")
            gru_model(X_train_nested, y_train_nested,
                      X_test_nested, y_test_nested,
                      X_valid_nested, y_valid_nested,
                      max_number_document_on_query,
                      listwise_eval_metric,
                      losses=loss,
                      metrics=metrics,
                      dir=dir, with_attention=True,
                      operational_dir=operational_dir)
            print("--------------END GRU---------------------")
        if (model_mode.lower() == "attention" or model_mode.lower() == "attn") or run_all:
            print("--------------START ATTN---------------------")
            attention_model(X_train_nested, y_train_nested,
                            X_test_nested, y_test_nested,
                            X_valid_nested, y_valid_nested,
                            max_number_document_on_query,
                            listwise_eval_metric,
                            losses=loss,
                            metrics=metrics,
                            dir=dir,
                            operational_dir=operational_dir)
            print("--------------END ATTN---------------------")
        if model_mode.lower() == "transformer" or run_all:
            print("--------------START Transformer---------------------")
            transformer_model(X_train_nested, y_train_nested,
                              X_test_nested, y_test_nested,
                              X_valid_nested, y_valid_nested,
                              max_number_document_on_query,
                              listwise_eval_metric,
                              losses=loss,
                              dir=dir,
                              operational_dir=operational_dir)
            print("--------------END Transformer---------------------")
        if model_mode.lower() == "transformer_with_feedforward" or run_all:
            print("--------------START Transformer WITH FEEDFORWARD---------------------")
            transformer_model(X_train_nested, y_train_nested,
                              X_test_nested, y_test_nested,
                              X_valid_nested, y_valid_nested,
                              max_number_document_on_query,
                              listwise_eval_metric,
                              losses=loss,
                              dir=dir, with_feedforward=True,
                              operational_dir=operational_dir)
            print("--------------END Transformer WITH FEEDFORWARD---------------------")
        if (model_mode.lower() == "light gbm"
            or model_mode.lower() == "lightgbm"
            or model_mode.lower() == "light_gbm") or run_all:
            print("--------------START Light GBM---------------------")
            light_gbm_model(X_train_nested, y_train_nested,
                            X_test_nested, y_test_nested,
                            X_valid_nested, y_valid_nested,
                            flattened_row_size,
                            listwise_eval_metric,
                            loss='mse',
                            metrics=metrics,
                            dir=dir,
                            operational_dir=operational_dir)
            print("--------------END Light GBM---------------------")


def calculate_based_on_cluster(X_test, y_test, y_pred, listwise_metric, model_name, loss, n_clusters=5,
                               dir='model_outputs'):
    X_test = np.array(X_test)
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    print("Y SHAPES", y_test.shape, "Y PRED", y_pred.shape)
    y_pred_for_cluster = y_pred.reshape(-1, 1)
    results = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(y_pred_for_cluster)
    X_by_cluster, y_test_by_cluster, y_pred_by_cluster = {}, {}, {}
    for cluster in range(n_clusters):
        X_by_cluster[cluster] = []
        y_test_by_cluster[cluster] = []
        y_pred_by_cluster[cluster] = []

    print("SHAPE FOR X TEST", X_test.shape, results.shape)
    for index, cluster in enumerate(results):
        X_by_cluster[cluster].append(X_test[index])
        y_test_by_cluster[cluster].append(y_test[index])
        y_pred_by_cluster[cluster].append([y_pred[index]])

    file_name = model_name + '_' + loss
    for cluster, info in X_by_cluster.items():
        # For model that is overfit really badly,
        # it's possible for 5 clusters with same value but other 4 don't get assigned
        if len(y_test_by_cluster[cluster]) != 0:
            _evaluate_model(X_by_cluster[cluster], y_test_by_cluster[cluster],
                            None, listwise_metric, scores=-1, cluster_num=str(cluster),
                            filename=file_name, loss=loss, results=y_pred_by_cluster[cluster],
                            dir=dir)


def main():
    parser = OptionParser()
    parser.add_option("-a", "--train_features",
                      help="Features file for training")
    parser.add_option("-b", "--train_predictions",
                      help="Predictions file for training")
    parser.add_option("-c", "--test_features",
                      help="Features file for testing")
    parser.add_option("-d", "--test_predictions",
                      help="Predictions file for testing")
    parser.add_option("-e", "--valid_features",
                      help="Features file for validation")
    parser.add_option("-f", "--valid_predictions",
                      help="Predictions file for validation")
    parser.add_option("-k", "--y_evaluate_at",
                      help="number cutoff for evaluation (i.e. NDCG at k)")
    parser.add_option("-o", "--output_dir",
                      help="Dir to save the outputs")
    parser.add_option("-m", "--model_to_run",
                      help="type of ML model to use (LightGBM, Feedforward, LSTM, GRU, Attention, Transformer)")
    parser.add_option("-r", "--ranking_type",
                      help="type of underlying LTR model to use (SVM, LambdaMart, RankNet)")
    (options, args) = parser.parse_args()

    underlying_model = "SVM"
    if options.ranking_type is not None:
        underlying_model = options.ranking_type

    run_all_option = True
    model_to_run_option = "Feedforward"
    if options.model_to_run is not None:
        run_all_option = False
        model_to_run_option = options.model_to_run

    k = 10
    if options.y_evaluate_at is not None:
        k = int(options.y_evaluate_at)

    print("UNDERLYING MACHINE LEARNING MODEL:", underlying_model)
    print("EVALUATION METRIC AT:", k)
    if not run_all_option:
        print("RUNNING ML MODEL", model_to_run_option)
    else:
        print("RUNNING ALL ML MODELS")

    X_train_nested, y_train_nested_by_metric = None, None
    X_test_nested, y_test_nested_by_metric = None, None
    X_valid_nested, y_valid_nested_by_metric = None, None
    doc_num_info = {}
    if _validate_flags(options.train_features, options.train_predictions) \
            and _validate_flags(options.test_features, options.test_predictions) \
            and _validate_flags(options.valid_features, options.valid_predictions):
        x_output_dir = options.output_dir
        if not os.path.exists(x_output_dir + "X_train.txt") or not os.path.exists(x_output_dir + "y_train.txt"):
            X_train_nested, y_train_nested_by_metric, doc_num_info = return_data(options.train_features,
                                                                                 options.train_predictions)
            write_data_to_file(X_train_nested, y_train_nested_by_metric, x_output_dir + "X_train.txt",
                               x_output_dir + "y_train.txt")

        if not os.path.exists(x_output_dir + "X_test.txt") or not os.path.exists(x_output_dir + "y_test.txt"):
            X_test_nested, y_test_nested_by_metric, doc_num_info = return_data(options.test_features,
                                                                               options.test_predictions)
            write_data_to_file(X_test_nested, y_test_nested_by_metric, x_output_dir + "X_test.txt",
                               x_output_dir + "y_test.txt")

        if not os.path.exists(x_output_dir + "X_valid.txt") or not os.path.exists(x_output_dir + "y_valid.txt"):
            X_valid_nested, y_valid_nested_by_metric, doc_num_info = return_data(options.valid_features,
                                                                                 options.valid_predictions)
            write_data_to_file(X_valid_nested, y_valid_nested_by_metric, x_output_dir + "X_valid.txt",
                               x_output_dir + "y_valid.txt")
        process_models(X_train_nested, y_train_nested_by_metric,
                       X_valid_nested, y_valid_nested_by_metric,
                       X_test_nested, y_test_nested_by_metric,
                       doc_num_info, run_all=False, model_mode='feedforward')
    else:
        print("WARNING: NO USER INPUT")
        print("RUNNING DEFAULT SETTING")
        x_output_dir = "ltrc_yahoo/set1/ML/"
        dense_input_dir = "ltrc_yahoo/set1/"
        y_output_dir = "ltrc_yahoo/set1/ML/" + underlying_model + "/"
        predict_input_dir = "ltrc_yahoo/set1/" + underlying_model + "/"
        # print("---------START WITH NOISE-----------------")
        # X_train_nested, y_train_nested_by_metric, doc_num_info = process_data(output_dir, input_dir,
        #                                                                       dataset="train",
        #                                                                       add_noise=True)
        # X_test_nested, y_test_nested_by_metric, doc_num_info = process_data(output_dir, input_dir,
        #                                                                     dataset="test",
        #                                                                     add_noise=True)
        # X_valid_nested, y_valid_nested_by_metric, doc_num_info = process_data(output_dir, input_dir,
        #                                                                       dataset="valid",
        #                                                                       add_noise=True)
        # process_models(X_train_nested, y_train_nested_by_metric,
        #                X_valid_nested, y_valid_nested_by_metric,
        #                X_test_nested, y_test_nested_by_metric,
        #                doc_num_info, dir='model_outputs_with_noise',
        #                run_all = False, model_mode = 'feedforward',
        #                # eval_metrics={"Precision"}
        #                )
        # print("---------END WITH NOISE-----------------")
        # print()

        print("---------START WITHOUT NOISE-----------------")
        X_train_nested, y_train_nested_by_metric, doc_num_info = process_data(x_output_dir=x_output_dir,
                                                                              dense_input_dir=dense_input_dir,
                                                                              predict_input_dir=predict_input_dir,
                                                                              y_output_dir=y_output_dir,
                                                                              dataset="train",
                                                                              add_noise=False,
                                                                              k=k)
        X_test_nested, y_test_nested_by_metric, doc_num_info = process_data(x_output_dir=x_output_dir,
                                                                            dense_input_dir=dense_input_dir,
                                                                            predict_input_dir=predict_input_dir,
                                                                            y_output_dir=y_output_dir,
                                                                            dataset="test",
                                                                            add_noise=False,
                                                                            k=k)
        X_valid_nested, y_valid_nested_by_metric, doc_num_info = process_data(x_output_dir=x_output_dir,
                                                                              dense_input_dir=dense_input_dir,
                                                                              predict_input_dir=predict_input_dir,
                                                                              y_output_dir=y_output_dir,
                                                                              dataset="valid",
                                                                              add_noise=False,
                                                                              k=k)
        process_models(X_train_nested, y_train_nested_by_metric,
                       X_valid_nested, y_valid_nested_by_metric,
                       X_test_nested, y_test_nested_by_metric,
                       doc_num_info, dir='model_outputs/' + underlying_model + '/' + (str(k)),
                       operational_dir='operational/' + underlying_model + '/' + (str(k)),
                       run_all=run_all_option, model_mode=model_to_run_option
                       # eval_metrics={"Precision"}
                       )
        print("---------END WITHOUT NOISE-----------------")


if __name__ == '__main__':
    main()
