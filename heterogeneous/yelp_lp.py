"""
Perform link prediction for yelp
"""
import os
import pandas as pd
import random
from math import ceil
from collections import defaultdict
import numpy as np

from heterogeneous.yelp_parser import *
from heterogeneous import yelp_embedding

from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as sklmet


def split_links(filename, ratio=0.5, train_filename=None):
    """
    Split edge list into train and test
    :param filename: name of the edge list
    :param ratio: fraction of rows to be sampled into training set
    :param train_filename:
    :return:
    """
    df = pd.read_csv(filename, sep=" ", header=None)

    train = df.sample(frac=ratio)
    if train_filename:
        train.to_csv(train_filename, sep=" ", header=False, index=False)

    test = df.loc[~df.index.isin(train.index)]
    return train, test


def read_node_embeddings(embedding_file):
    """
    Read the file with embeddings per node and return a dictionary.
    :param embedding_file:
    :return:
    """
    node_embeds = sc.textFile(embedding_file)\
        .map(lambda s: s.split("\t"))\
        .map(lambda s: [int(s[0]), [float(x) for x in s[1].split(",")]] )\
        .collect()

    return dict(node_embeds)


def construct_lp_instances(train_links, test_links, embedding_file,
                           neg_links_file, train_file, test_file):
    """
    Costruct train dataset and test dataset
    :param train_links: pandas dataframe of tuples
    :param test_links: pandas dataframe of tuples
    :param embedding_file: file with node embeddings
    :param neg_links_file: file to write sampled negative links
    :param train_file:
    :param test_file:
    :return:
    """

    nodes = list(set(train_links[1].values).union(set(train_links[0].values)))
    random.shuffle(nodes)
    train_nodes = sc.broadcast(nodes)
    factor = 12

    frac = max(0.001, 3*float(train_links.shape[0]) / (len(nodes) ** 2))
    node_frac = max(2, 2 * ceil(frac / len(nodes)))

    set1 = set((min(x), max(x)) for x in train_links.to_records(index=False))
    set2 = set((min(x), max(x)) for x in test_links.to_records(index=False))
    pos_links = set1.union(set2)

    neighbors = defaultdict(set)
    for (n1, n2) in pos_links:
        neighbors[n1].add(n2)
        neighbors[n2].add(n1)

    broad_neigh = sc.broadcast(neighbors)

    def sample_neg(index):
        node1 = train_nodes.value[index]
        neighbors = broad_neigh.value[node1]
        avail_list = list(set(train_nodes.value) - {node1} - neighbors)
        temp_frac = factor * max(len(neighbors), node_frac)
        random.shuffle(avail_list)
        # temp = [(min(node1, node2), max(node1, node2)) for node2 in avail_list]
        # if len(temp) > temp_frac:
        #     return random.sample(temp, temp_frac)
        return [(min(node1, node2), max(node1, node2)) for node2 in avail_list[:temp_frac]]
        # return [(min(node1, node2), max(node1, node2)) for node2 in avail_list[:node_frac]]

    # sample unconnected pairs
    parts = sc.parallelize(range(len(nodes))) \
        .repartition(200) \
        .flatMap(lambda idx: sample_neg(idx))

    neg_links = set(parts.take(factor*train_links.shape[0]))  # we need distinct pairs

    with open(neg_links_file, "w+") as f:
        csvf = csv.writer(f)
        csvf.writerows(neg_links)

    # remove broadcast vars
    broad_neigh.destroy()
    train_nodes.destroy()

    # get embeddings for the nodes
    node_embeds = sc.broadcast(read_node_embeddings(embedding_file))

    operators = [average, hadamard, weighted_l1, weighted_l2]

    def link_feature(node1, node2):
        """operator: hadamard, average, weighted_l1, weighted_l2"""
        if node1 in node_embeds.value and node2 in node_embeds.value:
            node1vec = node_embeds.value[node1]
            node2vec = node_embeds.value[node2]
            return [[operator(x, y) for (x,y) in zip(node1vec, node2vec)] for operator in operators]
        return [None for _ in operators]

    # train model
    train_neg_links = set(random.sample(neg_links, train_links.shape[0]))
    train_lp = set((min(x), max(x), 1) for x in train_links.to_records(index=False))\
        .union(x + (0,) for x in train_neg_links)  # (node1, node2, link_exists?)

    train_lp_all = sc.parallelize(train_lp)\
        .repartition(200)\
        .map(lambda element: [element[0], element[1]] + link_feature(element[0], element[1]) + [element[2]])\
        .collect()

    with open(train_file, "w+") as f:
        csvf = csv.writer(f)
        csvf.writerow(["n1", "n2", "average", "hadamard", "weighted_l1", "weighted_l2", "link_exists"])
        csvf.writerows(train_lp_all)

    # test model
    test_neg_links = set(random.sample(neg_links - train_neg_links, test_links.shape[0]))
    test_lp = set((min(x), max(x), 1) for x in test_links.to_records(index=False)) \
        .union(x + (0,) for x in test_neg_links)  # (node1, node2, link_exists?)

    test_lp_all = sc.parallelize(test_lp) \
        .repartition(200) \
        .map(lambda element: [element[0], element[1]] + link_feature(element[0], element[1]) + [element[2]]) \
        .collect()

    with open(test_file, "w+") as f:
        csvf = csv.writer(f)
        csvf.writerow(["n1", "n2", "average", "hadamard", "weighted_l1", "weighted_l2", "link_exists"])
        csvf.writerows(test_lp_all)

    return train_lp_all, test_lp_all


def hadamard(val1, val2):
    """
    :param val1: i-th component of the first feature vector
    :param val2: i-th component of the second feature vector
    :return:
    """
    return val1 * val2


def average(val1, val2):
    """
    :param val1: i-th component of the first feature vector
    :param val2: i-th component of the second feature vector
    :return:
    """
    return float(val1 + val2)/2


def weighted_l1(val1, val2):
    """
    :param val1: i-th component of the first feature vector
    :param val2: i-th component of the second feature vector
    :return:
    """
    return abs(val1 - val2)


def weighted_l2(val1, val2):
    """
    :param val1: i-th component of the first feature vector
    :param val2: i-th component of the second feature vector
    :return:
    """
    return (val1 - val2) ** 2


def evaluate_lp(train_file, test_file, performance_file=None):
    """

    :param train_file:
    :param test_file:
    :param performance_file:
    :return:
    """
    print("Performing evaluation for link prediction")
    operators = ["average", "hadamard", "weighted_l1", "weighted_l2"]
    performance = []

    df = pd.read_csv(train_file)
    df = df.dropna()  # remove na
    test_df = pd.read_csv(test_file)
    test_df = test_df.dropna()  # remove na

    y_train = list(df["link_exists"])
    y_test = list(test_df["link_exists"])

    for operator in operators:
        print("working with operator {}".format(operator))
        df[operator] = df[operator].apply(
            lambda x: [float(y) for y in str(x).replace("[", "").replace("]", "").replace("'", "").split(", ")])
        X_train = np.array([np.array(xi) for xi in list(df[operator]) if xi])

        clf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1)
        clf = clf.fit(X_train, y_train)

        test_df[operator] = test_df[operator].apply(
            lambda x: [float(y) for y in str(x).replace("[", "").replace("]", "").replace("'", "").split(", ")])
        X_test = np.array([np.array(xi) for xi in list(test_df[operator]) if xi])

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        scores = [x[0] for x in y_proba]

        # metrics: accuracy, f-score, precision, auc, aupr
        precision = sklmet.accuracy_score(y_test, y_pred)
        auc = sklmet.roc_auc_score(y_test, scores)
        aupr = sklmet.average_precision_score(y_test, scores)
        accuracy = sklmet.accuracy_score(y_test, y_pred)
        fmeasure = sklmet.f1_score(y_test, y_pred)

        performance.append([operator, precision, auc, aupr, accuracy, fmeasure])

    print("Writing performance file...")
    if performance_file:
        with open(performance_file, "w+") as f:
            csvf = csv.writer(f)
            csvf.writerow(["operator", "precision", "auc", "aupr", "accuracy", "fmeasure"])
            csvf.writerows(performance)

    return performance

if __name__ == "__main__":
    print("Splitting links for review-business")
    dir_path = "/home/natalia/Projects/network-analysis/node2vec/graph/yelp"
    filename = dir_path + "/yelp-sushi-review-business.edgelist/part-00000-88c5743b-1abd-4210-8b63-236de24e9077.csv"
    out_file = dir_path + "/yelp-sushi-review-business-train.edgelist"
    train_links, test_links = split_links(filename, ratio=0.5, train_filename=out_file)

    edgelists = [out_file,
                 dir_path + "/yelp-sushi-review-keyword.edgelist/part-00000-c4cb97ac-4312-4912-86e8-06d4b12649fc.csv",
                 dir_path + "/yelp-sushi-user-review.edgelist/part-00000-834d86d8-2885-495b-af05-270c2356e441.csv"]
    file_path = os.path.join(dir_path, "yelp-sushi.edgelist")
    yelp_embedding.create_edgelist(edgelists, file_path)

    embedding_file = "/home/natalia/Projects/network-analysis/node2vec/emb/yelp/yelp-sushi-deeper.emb.emb"
    # node_embeds = read_node_embeddings(embedding_file)
    # spark implementation for node2vec is faster!
    # yelp_embedding.node2vec_embed(input_file=file_path,
    #                               out_file="/home/natalia/Projects/network-analysis/node2vec/emb/yelp-sushi-review-business.emd")


    train_file = "/home/natalia/Projects/network-analysis/node2vec/heterogeneous/yelp_lp/hadamard_train_file.txt"
    test_file = "/home/natalia/Projects/network-analysis/node2vec/heterogeneous/yelp_lp/hadamard_test_file.txt"

    train, test = construct_lp_instances(train_links, test_links, embedding_file, "negative_links.txt", train_file, test_file)

    perf = evaluate_lp(train_file, test_file, performance_file="performance_yelp_sushi_more.txt")