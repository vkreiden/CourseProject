import os
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import gensim.models

class Corpus(object):
    """
    A collection of norm and fail docs - thus prelabeled.
    """

    def __init__(self, norm_files_path, fail_files_path):
        """
        Initialize empty document list.
        """
        self.norm_files_path = norm_files_path
        self.fail_files_path = fail_files_path
        self.norm_files = []
        self.fail_files = []
        self.vocabulary = set()
        self.model = {}
        self.files_pos = []
        self.files_labels = []

    def process_files(self, files_path):
        is_norm = (self.norm_files_path == files_path)

        dir_entries = os.scandir(files_path)
        for entry in dir_entries:
            f = []
            if entry.is_dir():
                continue

            with open(entry) as fp:
                line = fp.readline()
                while line:
                    s = re.sub(r'[^a-zA-Z0-9]', ' ', line).split()
                    for i in s:
                        self.vocabulary.add(i)

                    f.append(s)
                    line = fp.readline()

            if is_norm:
                self.norm_files.append(f)
            else:
                self.fail_files.append(f)

    def train_word2vec(self):
        self.process_files(self.norm_files_path)
        self.process_files(self.fail_files_path)

        sentences = [line for f in (
            self.norm_files + self.fail_files) for line in f]
        
        self.model = gensim.models.Word2Vec(sentences=sentences, min_count=1)

        return

    def build_training_set(self):
        for f in self.norm_files + self.fail_files:
            lines = []
            for line in f:
                l_pos = np.average(
                    np.array([self.model[w] for w in line]), axis=0)
                lines.append(l_pos)

            self.files_pos.append(
                np.average(np.array([l for l in lines]), axis=0))

        ones = np.ones((len(self.norm_files),), dtype=int)
        zeroes = np.zeros((len(self.norm_files),), dtype=int)
        self.files_labels = np.concatenate([ones, zeroes])

        return

    def assess_classifier(self):

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.files_pos), self.files_labels, test_size=0.2, random_state=0)

        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)

        print("Number of mislabeled points out of a total %d points : %d" %
              (X_test.shape[0], (y_test != y_pred).sum()))

        return


def main():
    norm_files_path = '../data/norm'
    fail_files_path = '../data/fail'

    corpus = Corpus(norm_files_path, fail_files_path)

    corpus.train_word2vec()
    corpus.build_training_set()
    corpus.assess_classifier()


if __name__ == '__main__':
    main()
