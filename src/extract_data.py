import json
import random

def extract_data(filepath="../data/reviews_Toys_and_Games_5.json"):
    """
    从数据集里面读取数据并取出 user(reviewerID), item(asin), review(reviewText), summary(summary), rating(overall) 信息
    """
    data = list()
    with open(filepath, "r") as fin:
        for line in fin:
            datum = json.loads(line.strip())
            data.append((datum["reviewerID"], datum["asin"], datum["reviewText"], datum["summary"], datum["overall"]))
    
    # len = 167597
    # print(len(data))
    return data

def verify_random_seed(origin_data):
    """
    选择一个随机种子，把 data 分成 80% training，20% validation + testing
    需要保证在 validation / testing 中出现的 user 和 item，在 training 中都出现过
    防止 cold start 的问题
    """
    
    while True:
        try:
            seed = random.randint(1, 100000000)
            random.seed(seed)

            data = origin_data[:]
            random.shuffle(data)

            training_size = int(len(data) * 0.8)

            training_data = data[:training_size]
            vt_data = data[training_size:]

            training_user, training_item = set(), set()
            for user, item, *_ in training_data:
                training_user.add(user)
                training_item.add(item)
            
            for user, item, *_ in vt_data:
                if user not in training_user:
                    raise ValueError(f"user {user} is not in the training data")
                if item not in training_item:
                    raise ValueError(f"item {item} is not in the training data")
        except ValueError as e:
            print(f"for seed {seed}, {e}")
        else:
            # 63863183 是一个合法的 seed
            return seed

def prepare_train_valid_test(filepath="../data/reviews_Toys_and_Games_5.json", seed=63863183):
    """
    把 data 分成 80% training, 10% validation, 10% testing
    需要数据请直接调用该函数
    会把 (user, item, review, summary) 中的 user 和 item 转换为自然数的 index
    返回值为：
        - 长度为 134077 的 training data
        - 长度为 16759 的 validation data
        - 长度为 16761 的 testing data
        - user 的数量 19412
        - item 的数量 11924
    """
    
    data = extract_data(filepath)
    random.seed(seed)
    random.shuffle(data)
    
    training_size = int(len(data) * 0.8)
    validation_size = int(len(data) * 0.1)

    training_data = data[:training_size]
    validation_data = data[training_size:training_size + validation_size]
    testing_data = data[training_size + validation_size:]

    user_to_idx, item_to_idx = dict(), dict()

    for user, item, *_ in training_data:
        if user not in user_to_idx:
            user_to_idx[user] = len(user_to_idx)
        if item not in item_to_idx:
            item_to_idx[item] = len(item_to_idx)
    
    training_data = [(user_to_idx[user], item_to_idx[item], *_) for user, item, *_ in training_data]
    validation_data = [(user_to_idx[user], item_to_idx[item], *_) for user, item, *_ in validation_data]
    testing_data = [(user_to_idx[user], item_to_idx[item], *_) for user, item, *_ in testing_data]

    return training_data, validation_data, testing_data, len(user_to_idx), len(item_to_idx)


if __name__ == "__main__":
    training_data, validation_data, testing_data, user_count, item_count = prepare_train_valid_test()
    assert(len(training_data) == 134077)
    assert(len(validation_data) == 16759)
    assert(len(testing_data) == 16761)
    assert(user_count == 19412)
    assert(item_count == 11924)
