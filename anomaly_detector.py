import os
import time
import configparser
import importlib
import re
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import gensim.models


class Corpus(object):
    """
    A collection of norm and fail docs - thus prelabeled.
    """

    def __init__(self, norm_files_path, fail_files_path,
                 remove_prefix_delim, remove_prefix_delim_occur):
        """
        Initialize empty document list.
        """

        self.norm_files_path = norm_files_path
        self.fail_files_path = fail_files_path
        self.remove_prefix_delim = remove_prefix_delim
        self.remove_prefix_delim_occur = remove_prefix_delim_occur
        self.norm_files = []
        self.fail_files = []
        self.vocabulary = set()
        self.model = {}
        self.files_pos = []
        self.files_labels = []

        self.cls_name = []
        self.cls_runtime = []
        self.cls_accuracy = []

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
                    parts = line.split(
                        self.remove_prefix_delim, self.remove_prefix_delim_occur + 1)
                    i = len(line) - len(parts[-1]) - \
                        len(self.remove_prefix_delim)

                    s = re.sub(r'[^a-zA-Z0-9]', ' ', line[i +
                                                          len(self.remove_prefix_delim):]).split()
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

    def assess_classifier(self, classifier):

        X = np.array(self.files_pos)
        Y = self.files_labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1)

        module_name, class_name = classifier.rsplit(".", 1)
        Classifier = getattr(importlib.import_module(
            module_name), class_name)

        cl = Classifier()

        start = time.time()
        y_pred = cl.fit(X_train, y_train).predict(X_test)
        end = time.time()





        print("Classifier %s: number of mislabeled points out of a total %d points : %d" %
              (class_name, X_test.shape[0], (y_test != y_pred).sum()))

        self.cls_name.append(class_name)
        self.cls_accuracy.append(accuracy_score(y_test, y_pred))
        self.cls_runtime.append(end - start)

        return

    def compare_classifiers(self, classifiers):

        for classifier in classifiers:
            self.assess_classifier(classifier)

        x = np.array(self.cls_accuracy)
        y = np.array(self.cls_runtime)

        plt.scatter(x, y, alpha=0.5)

        for i, txt in enumerate(self.cls_name):
            plt.annotate(txt, (x[i], y[i]))

        plt.show()

        return


def main():
    CONFIG_FILE = './config.ini'

    LOCATIONS = 'locations'
    NORM_FILES_PATH = 'norm_files_path'
    FAIL_FILES_PATH = 'fail_files_path'

    PARSING = 'parsing'
    REMOVE_PREFIX_DELIM = 'remove_prefix_delim'
    REMOVE_PREFIX_DELIM_OCCUR = 'remove_prefix_delim_occur'

    CLASSIFICATION = 'classification'
    CLASSIFIERS = 'classifiers'

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    norm_files_path = config[LOCATIONS][NORM_FILES_PATH]
    fail_files_path = config[LOCATIONS][FAIL_FILES_PATH]
    classifiers = config[CLASSIFICATION][CLASSIFIERS]
    remove_prefix_delim = config[PARSING][REMOVE_PREFIX_DELIM]
    remove_prefix_delim_occur = int(config[PARSING][REMOVE_PREFIX_DELIM_OCCUR])

    corpus = Corpus(norm_files_path, fail_files_path,
                    remove_prefix_delim, remove_prefix_delim_occur)

    corpus.train_word2vec()
    corpus.build_training_set()
    corpus.compare_classifiers(classifiers.split(","))


if __name__ == '__main__':
    main()
