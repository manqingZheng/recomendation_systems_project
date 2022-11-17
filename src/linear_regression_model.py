import numpy as np
from tqdm import tqdm, trange
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from extract_data import extract_data, prepare_train_valid_test
from bag_of_words_feature import BagOfWordsFeature
from tf_idf import TfIdfFeature


def extract_label(data, rating_index=4):
    Y = []
    for i in trange(len(data)):
        Y.append(float(data[i][rating_index]))
    Y = np.array(Y)
    print('extracted label shape', Y.shape)
    return Y


def extract_semantic_feature(feature_extractor, data):
    X = []
    for i in trange(len(data)):
        X.append(feature_extractor.feature(data[i]))
    X = np.array(X)
    print('extracted feature shape', X.shape)
    return X


def extract_length_feature(data):
    X = []
    for i in trange(len(data)):
        X.append([len(data[i][2]), len(data[i][3])])
    X = np.array(X)
    print('extracted feature shape', X.shape)
    return X


class LinearModel:
    def __init__(self, data_path, feature_type):
        """
        linear regression model for rating prediction
        feature type should be on of the following:
        `review_length`, `bag_of_words`, or `tf_idf`
        """
        self.data_path = data_path
        self.feature_type = feature_type
        self.model = LinearRegression()

    @staticmethod
    def prepare_bow_data(train_data, val_data, test_data):
        print('extracting feature ...')
        feature_extractor = BagOfWordsFeature(text_index=2,
                                              bag_length=4,
                                              use_stemming=True,
                                              remove_stop_words=True,
                                              feature_length=500)
        feature_extractor.preprocessing(train_data)
        train_X = extract_semantic_feature(feature_extractor, train_data)
        val_X = extract_semantic_feature(feature_extractor, val_data)
        test_X = extract_semantic_feature(feature_extractor, test_data)
        return train_X, val_X, test_X

    @staticmethod
    def prepare_tf_idf_data(train_data, val_data, test_data):
        print('extracting feature ...')
        feature_extractor = TfIdfFeature()
        feature_extractor.preprocessing(train_data)
        train_X = extract_semantic_feature(feature_extractor, train_data)
        val_X = extract_semantic_feature(feature_extractor, val_data)
        test_X = extract_semantic_feature(feature_extractor, test_data)
        return train_X, val_X, test_X

    @staticmethod
    def prepare_length_data(train_data, val_data, test_data):
        print('extracting feature ...')
        train_X = extract_length_feature(train_data)
        val_X = extract_length_feature(val_data)
        test_X = extract_length_feature(test_data)
        return train_X, val_X, test_X

    def prepare_model_data(self):
        train_data, val_data, test_data, \
        user_count, item_count = prepare_train_valid_test(self.data_path)
        if self.feature_type == 'review_length':
            train_X, val_X, test_X = self.prepare_length_data(train_data, val_data, test_data)
        elif self.feature_type == 'bag_of_words':
            train_X, val_X, test_X = self.prepare_bow_data(train_data, val_data, test_data)
        elif self.feature_type == 'tf_idf':
            train_X, val_X, test_X = self.prepare_tf_idf_data(train_data, val_data, test_data)
        else:
            raise NotImplementedError('Invalid Feature Type!')

        train_Y = extract_label(train_data)
        val_Y = extract_label(val_data)
        test_Y = extract_label(test_data)
        return train_X, train_Y, val_X, val_Y, test_X, test_Y

    def train(self, train_X, train_Y):
        self.model.fit(train_X, train_Y)

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def evaluate(gt_Y, predict_Y):
        return mean_squared_error(gt_Y, predict_Y)

    def run(self):
        train_X, train_Y, val_X, val_Y, test_X, test_Y = self.prepare_model_data()
        self.train(train_X, train_Y)
        val_predict_Y = self.predict(val_X)
        mse = self.evaluate(val_Y, val_predict_Y)
        print('mse on validation set', mse)
        test_predict_Y = self.predict(test_X)
        mse = self.evaluate(test_Y, test_predict_Y)
        print('mse on test set', mse)


if __name__ == '__main__':
    data_path = 'reviews_Toys_and_Games_5.json'
    for feature_type in ['review_length', 'bag_of_words', 'tf_idf']:
        print('{} linear regression with feature type {} {}'.format('*' * 10, feature_type, '*' * 10))
        model = LinearModel(data_path, feature_type=feature_type)
        model.run()
        print('*' * 30)
