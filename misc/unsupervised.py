from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from utils.logger import logger_factory
from config.config import config as config
from pathlib import Path


def unsupervised(logger, filename, predfilename):
    logger.info(f"Corpus datafile: {filename}")
    data = pd.read_csv(filename)
    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(data["Title"])
    true_k = 1000
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(x)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster % d:" % i),
        s = ""
        for ind in order_centroids[i, :10]:
            s = s + " " + terms[ind]
        print(s)
    print("*********Prediction***************")
    pred_data = pd.read_csv(predfilename)
    for i in range(10):
        print(pred_data["Title"][i])
        x = vectorizer.transform([pred_data["Title"][i]])
        predicted = model.predict(x)
        print(predicted)
        s = ""
        for ind in order_centroids[predicted[0], :100]:
            s = s + " " + terms[ind]
        print(s)
        print()


def main():
    logger = logger_factory(log_name=config['model']['arch'], log_dir='.')
    unsupervised(logger, Path('..')/config['data']['train_file_path'],
                 Path('..')/config['data']['validation_file_path'])


if __name__ == '__main__':
    main()
