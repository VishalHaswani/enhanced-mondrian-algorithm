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

def pprint(t):
    for i in t:
        print(i)

# read dataset from file
def read_dataset(filename):
    dataset = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(line.split(","))
    return dataset

# find the distance between two records based on the parameter
def distance(rec1, rec2, parameter):
    if parameter == "age":
        return abs(int(rec1[0]) - int(rec2[0]))
    elif parameter == "zipcode":
        return abs(int(rec1[1]) - int(rec2[1]))


# divide the cluster into two clusters based on the centroids
def splitCurrentCluster(cluster, parameter):
    centroid = [None, None]
    # find the two farthest most records in the cluster and use them as centroids
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            if centroid == [None, None] or distance(cluster[i], cluster[j], parameter) > distance(
                centroid[0], centroid[1], parameter
            ):
                centroid = [cluster[i], cluster[j]]

    # divide the cluster into two clusters based on the centroids
    cluster1 = []
    cluster2 = []
    for i in range(len(cluster)):
        if distance(cluster[i], centroid[0], parameter) < distance(
            cluster[i], centroid[1], parameter
        ):
            cluster1.append(cluster[i])
        else:
            cluster2.append(cluster[i])

    return [cluster1, cluster2]


# use divide and conquer approach for splitting the dataset into parallel number of clusters
def splitCluster(dataset, parameter, k):
    if (len(dataset) < 2 * k):
        return [dataset]
    
    clusters = splitCurrentCluster(dataset, parameter)

    threads = [
        ThreadWithReturnValue(target=splitCluster, args=(clusters[0], parameter, k,)),
        ThreadWithReturnValue(target=splitCluster, args=(clusters[1], parameter, k,))
    ]

    threads[0].start()
    threads[1].start()

    result = []
    result.extend(threads[0].join())
    result.extend(threads[1].join())

    return result

def main():
    k = 2  # k-anonymity
    processors = 4

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
    # df = pd.DataFrame(data=data, columns=columns)

    # parallel = len(data) / processors

    # divide into parallel number of clusters based on 1Dimension only!!
    # clusters = divideConquer(data, "age", parallel)
    clusters = splitCluster(data, "age", k)
    anonymized_dataframe = pd.DataFrame(data=[], columns=columns)
    # for each cluster, use mondrian algorithm on all dimensions (QI)
    for cluster in clusters:
        df = pd.DataFrame(data=cluster, columns=columns)
        p = anonypy.Preserver(df, feature_columns, sensitive_column)
        rows = p.anonymize_k_anonymity(k=k)

        # print anonymized cluster
        # dfn = pd.DataFrame(rows)
        # print(dfn)

        anonymized_dataframe = pd.concat([anonymized_dataframe, pd.DataFrame(rows, columns=columns)])
    anonymized_dataframe.reset_index(inplace = True)
    print(anonymized_dataframe)

main()