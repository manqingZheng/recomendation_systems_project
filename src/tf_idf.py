from extract_data import extract_data
import string
from collections import defaultdict
import nltk
from nltk.stem.porter import *
import math
from tqdm import tqdm


class TfIdfFeature():
    def __init__(self, text_index = 2, feature_length = 500):
        self.stemmer = PorterStemmer()
        self.df = defaultdict(float)
        self.text_index = text_index
        self.words = []
        self.N = 0
        self.punctuation = set(string.punctuation)
        self.feature_length = feature_length

    def preprocessing(self, dataset):
        self.df = defaultdict(int)
        self.N = len(dataset)
        
        for d in tqdm(dataset):
            r = ''.join([c for c in d[self.text_index].lower() if not c in self.punctuation])
            for w in set(r.split()):
                w = self.stemmer.stem(w)
                self.df[w] += 1

        counts = [(self.df[w], w) for w in self.df]
        counts.sort()
        counts.reverse()
        self.words = [x[1] for x in counts[:self.feature_length]]
        self.wordId = dict(zip(self.words, range(len(self.words))))
        self.wordSet = set(self.words)

    def feature(self, datum):
        feat = [0] * len(self.words)
        r = ''.join([c for c in datum[self.text_index].lower() if not c in self.punctuation])
        ws = r.split()
        for word in ws:
            w = self.stemmer.stem(word)
            if w in self.wordSet:
                feat[self.wordId[w]] += math.log2(self.N / self.df[w])
        return feat

if __name__ == "__main__":
    data = extract_data(filepath='../data/Toys_and_Games_5.json')
    smldata = data[:100]
    tfIdfFeature = TfIdfFeature()
    tfIdfFeature.preprocessing(smldata)
    print(len(tfIdfFeature.feature(smldata[0])))
    print(tfIdfFeature.feature(smldata[0]))
    print(tfIdfFeature.feature(smldata[1]))