from math import log2
import anonypy
import pandas as pd
from threading import Thread 

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args,
                                                **self._kwargs)
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class MultithreadedDataSpliting():
    def __init__(self, K: int, distance_fn):
        self.K = K
        self.distance_fn = distance_fn

    def pprint(self, t):
        for i in t:
            print(i)

    # read dataset from file
    def read_dataset(self, filename):
        dataset = []
        with open(filename) as f:
            for line in f:
                line = line.strip()
                if line:
                    dataset.append(line.split(","))
        return dataset

    # divide the cluster into two clusters based on the centroids
    def splitCurrentCluster(self, cluster):
        centroid = [None, None]
        # find the two farthest most records in the cluster and use them as centroids
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                if centroid == [None, None] or self.distance_fn(cluster[i], cluster[j]) > self.distance_fn(
                    centroid[0], centroid[1]
                ):
                    centroid = [cluster[i], cluster[j]]

        # divide the cluster into two clusters based on the centroids
        # TODO: Deal with outliers
        new_clusters = [[], []]
        for i in range(len(cluster)):
            if self.distance_fn(cluster[i], centroid[0]) < self.distance_fn(cluster[i], centroid[1]):
                new_clusters[0].append(cluster[i])
            else:
                new_clusters[1].append(cluster[i])

        return new_clusters


    # use divide and conquer approach for splitting the dataset into parallel number of clusters
    def splitCluster(self, dataset):
        # print(dataset)
        if (len(dataset) < 2 * self.K):
            return [dataset]
        
        clusters = self.splitCurrentCluster(dataset)

        threads = [
            ThreadWithReturnValue(target=self.splitCluster, args=(clusters[0],)),
            ThreadWithReturnValue(target=self.splitCluster, args=(clusters[1],))
        ]

        threads[0].start()
        threads[1].start()

        result = []
        result.extend(threads[0].join())
        result.extend(threads[1].join())

        return result

def main():
    K = 2  # k-anonymity

    # # filename = ""
    # # dataset = read_dataset(filename)

    # sample dataset
    columns = ["age", "zipcode", "salary"]
    feature_columns = ["age", "zipcode"]
    sensitive_column = "salary"

    data = [
        [28, 62816, 24489],
        [52, 11788, 80813],
        [37, 28921, 57770],
        [27, 12660, 88575],
        [59, 27185, 59408],
        [54, 97909, 56725],
        [32, 45637, 57556],
        [64, 31019, 33096],
        [45, 12799, 63743],
        [40, 36755, 50332],
    ]

    # create a custom distance function for figuring out the distance
    def distance_fn(rec1, rec2):
        return (abs(rec1[0] - rec2[0]) + abs(rec1[1] - rec2[1]) * 0.001)

    anonimizer = MultithreadedDataSpliting(K, distance_fn)
    clusters = anonimizer.splitCluster(data)
    anonymized_dataframe = pd.DataFrame(data=[], columns=columns)
    # for each cluster, use mondrian algorithm on all dimensions (QI)
    for cluster in clusters:
        df = pd.DataFrame(data=cluster, columns=columns)
        p = anonypy.Preserver(df, feature_columns, sensitive_column)
        rows = p.anonymize_k_anonymity(k=K)

        # print anonymized cluster
        # dfn = pd.DataFrame(rows)
        # print(dfn)

        anonymized_dataframe = pd.concat([anonymized_dataframe, pd.DataFrame(rows, columns=columns)])
    anonymized_dataframe.reset_index(drop=True, inplace = True)
    print(anonymized_dataframe)

main()