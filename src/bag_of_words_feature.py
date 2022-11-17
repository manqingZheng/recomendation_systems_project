from nltk.stem.porter import *
from extract_data import extract_data
import string
import collections
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 
from nltk.util import ngrams
from tqdm import tqdm


class BagOfWordsFeature():
    def __init__(self, 
                    text_index=2, 
                    bag_length=4, 
                    use_stemming=True,
                    remove_stop_words=True, 
                    feature_length=1000):
        self.text_index = text_index
        self.bag_length = bag_length
        self.remove_stop_words = remove_stop_words
        self.feature_length = feature_length - 1
        self.use_stemming = use_stemming
        self.stop_words_set = set(stopwords.words("english"))
        self.ready = False

    def preprocessing(self, dataset):
        word_count = collections.Counter()
        self.punctuation = set(string.punctuation)
        stemmer = PorterStemmer()
        for d in tqdm(dataset):
            r = ''.join([c for c in d[self.text_index].lower() if not c in self.punctuation])
            ws = r.split()
            if self.remove_stop_words:
                ws = list(filter(lambda x: x not in self.stop_words_set, ws))
            if self.use_stemming:
                ws = [stemmer.stem(word) for word in ws]
            bags = [list(ngrams(ws, i)) for i in range(1, self.bag_length + 1)]
            dummyWords = [' '.join(x) for bag in bags for x in bag]
            dummyCount = collections.Counter(dummyWords)
            word_count.update(dummyCount)

        counts = [(word_count[w], w) for w in word_count]
        counts.sort()
        counts.reverse()
        words = [x[1] for x in counts[:self.feature_length]]

        self.wordId = dict(zip(words, range(len(words))))
        self.wordSet = set(words)
        self.ready = True

    def feature(self, datum):
        assert(self.ready == True)

        feat = [0] * len(self.wordSet)
        r = ''.join([c for c in datum[2].lower() if not c in self.punctuation])
        ws = r.split()
        bags = [list(ngrams(ws, i)) for i in range(1, self.bag_length + 1)]
        dummyWords = [' '.join(x) for bag in bags for x in bag]
        dummyCount = collections.Counter(dummyWords)
        for word in dummyWords:
            if word in self.wordSet:
                feat[self.wordId[word]] += dummyCount[word]
        feat.append(1) #offset
        return feat

if __name__ == "__main__":
    data = extract_data()
    smldata = data[:100]
    bagOfWordsFeature = BagOfWordsFeature(text_index=2, 
                                        bag_length=4, 
                                        use_stemming=True,
                                        remove_stop_words=True, 
                                        feature_length=500)
    bagOfWordsFeature.preprocessing(smldata)
    print(len(bagOfWordsFeature.feature(smldata[0])))