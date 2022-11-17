## For user/item prediction
from collections import defaultdict
from extract_data import prepare_train_valid_test
import numpy as np

class collaborate_filtering_with_user_item_pairs:
    def __init__(self):
        self.ready = False

    def Jaccard(s1, s2):
        numer = len(s1.intersection(s2))
        denom = len(s1.union(s2))
        if denom == 0:
            return 0
        return numer / denom

    def predict_with_other_items(self, target_user,
            target_item,
            distance=Jaccard):
        assert(self.ready == True)
        target_users = self.users_per_item[target_item]
        similarities = []
        ratings = []
        for item in self.items_per_user[target_user]-{target_item}:
            similarities.append(distance(target_users, self.users_per_item[item]))
            ratings.append(self.ratings[target_user, item])
        return self.rateMean + sum(sim * rate for sim, rate in zip(similarities, ratings)) / (1e-6 + sum(similarities))

    def predict_with_other_users(self, target_user,
            target_item,
            distance=Jaccard):
        assert(self.ready == True)
        target_items = self.items_per_user[target_user]
        similarities = []
        ratings = []
        for user in self.users_per_item[target_item]-{target_user}:
            similarities.append(distance(target_items, self.items_per_user[user]))
            ratings.append(self.ratings[user, target_item])
        return self.rateMean + sum(sim * (rate - self.rateMean) for sim, rate in zip(similarities, ratings)) / (1e-6 + sum(similarities))

    def train(self, trainset):
        self.items_per_user = defaultdict(set)
        self.users_per_item = defaultdict(set)
        self.ratings = {}
        self.users = set()
        self.items = set()
        for datum in trainset:
            user, item, rate = datum[0], datum[1], datum[4]
            self.users.add(user)
            self.items.add(item)
            self.items_per_user[user].add(item)
            self.users_per_item[item].add(user)
            self.ratings[(user, item)] = rate
        self.rateMean = sum(self.ratings.values())/len(self.ratings.values())
        self.ready = True

    def MSE(self, X, y):
        X = np.array(X)
        y = np.array(y)
        return np.mean(np.square(X - y))

    def test(self, dataset, predict):
        assert(self.ready == True)
        y = []
        X = []
        for datum in dataset:
            user, item, rate = datum[0], datum[1], datum[4]
            y.append(rate)
            X.append(predict(user, item))
        return self.MSE(X, y)


if __name__ == "__main__":
    model = collaborate_filtering_with_user_item_pairs()
    training_data, validation_data, testing_data, user_count, item_count = prepare_train_valid_test()
    model.train(training_data)
    print('validation mse with items: ', model.test(validation_data, model.predict_with_other_items))
    print('test mse with items: ', model.test(testing_data, model.predict_with_other_items))
    print('validation mse with users: ', model.test(validation_data, model.predict_with_other_users))
    print('test mse with users: ', model.test(testing_data, model.predict_with_other_users))