import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from meta_feature_extraction import sort_files


def get_traces(df):
    traces, ids = [], []
    for group in df.groupby("case:concept:name"):
        events = list(group[1]["concept:name"])
        traces.append(" ".join(x for x in events))
        ids.append(group[0])
    return traces, ids


def get_traces_list(df):
    largest_trace = -1
    traces = []
    for group in df.groupby("case:concept:name"):
        traces.append(list(group[1]["concept:name"]))
        if len(group[1]) > largest_trace:
            largest_trace = len(group[1])

    return traces, largest_trace


def one_hot(traces, file):
    if "trace" in file:
        corpus = CountVectorizer(analyzer="char").fit_transform(traces)
    else:
        corpus = CountVectorizer().fit_transform(traces)
    return Binarizer().fit_transform(corpus.toarray())


def n_grams(traces):
    bi_traces, tri_traces = [], []
    for trace in traces:
        acts = trace.split(" ")
        bi_trace = ""
        for i in range(len(acts) - 1):
            bi_trace += acts[i] + "_" + acts[i + 1] + " "

        tri_trace = ""
        for i in range(len(acts) - 2):
            tri_trace += acts[i] + "_" + acts[i + 1] + "_" + acts[i + 2] + " "

        bi_traces.append(bi_trace)
        tri_traces.append(tri_trace)

    bi_gram = CountVectorizer().fit_transform(bi_traces).toarray()
    tri_gram = CountVectorizer().fit_transform(tri_traces).toarray()

    return bi_gram, tri_gram


def position_profile(df):
    traces, largest_trace = get_traces_list(df)

    # creating position matrix
    activities = sorted(list(set(df["concept:name"])))
    position_matrix = pd.DataFrame(
        0, index=activities, columns=list(range(1, largest_trace + 1))
    )

    # populating position matrix
    for trace in traces:
        for i in range(len(trace)):
            position_matrix.loc[trace[i], i + 1] += 1

    # compute trace encodings
    n_positions = position_matrix.shape[1]
    position_encoding = []
    for trace in traces:
        trace_encoding = []
        for i in range(len(trace)):
            trace_encoding.append(position_matrix.loc[trace[i], i + 1])

        if len(trace_encoding) < n_positions:
            trace_encoding.extend([0] * (n_positions - len(trace_encoding)))
        position_encoding.append(trace_encoding)

    return np.array(position_encoding)


def variant_score(labels, traces):
    cluster_ids = dict()
    for label, trace in zip(labels, traces):
        if label not in cluster_ids:
            cluster_ids[label] = [trace]
        else:
            cluster_ids[label].append(trace)

    weighted_mean = 0
    for k, v in cluster_ids.items():
        n_instances = len(v)
        if k != -1:
            variation = (len(set(v)) - 1) / n_instances
            weighted_mean += variation * n_instances
        else:
            weighted_mean += 1 * n_instances

    return weighted_mean / len(traces)


def compute_silhouette(encoding, labels):
    n_clusters = len(np.unique(labels))

    if n_clusters == 1:
        return -1
    elif n_clusters == encoding.shape[0]:
        return -1

    return silhouette_score(encoding, labels)


def cluster_dbscan(encoding, traces, enc_type, log):
    out = []
    for eps in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]:
        start_time = time.time()

        dbscan = DBSCAN(eps=eps, n_jobs=-1)
        labels = dbscan.fit_predict(encoding)

        end_time = time.time() - start_time

        sil = compute_silhouette(encoding, labels)
        var_score = variant_score(labels, traces)
        out.append([log, enc_type, f"dbscan_eps{eps}", sil, var_score, end_time])
    return out


@ignore_warnings(category=ConvergenceWarning)
def cluster_kmeans(encoding, traces, enc_type, log):
    out = []
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        start_time = time.time()

        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(encoding)

        end_time = time.time() - start_time

        sil = compute_silhouette(encoding, labels)
        var_score = variant_score(labels, traces)
        out.append([log, enc_type, f"kmeans_k{k}", sil, var_score, end_time])
    return out


def cluster_agglomerative(encoding, traces, enc_type, log):
    out = []
    for k in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        start_time = time.time()

        aggclus = AgglomerativeClustering(n_clusters=k)
        labels = aggclus.fit_predict(encoding)

        end_time = time.time() - start_time

        sil = compute_silhouette(encoding, labels)
        var_score = variant_score(labels, traces)
        out.append([log, enc_type, f"agglomerative_k{k}", sil, var_score, end_time])
    return out


def cluster(encoding, traces, enc_type, log):
    cluster_results = cluster_dbscan(encoding, traces, enc_type, log)
    cluster_results.extend(cluster_kmeans(encoding, traces, enc_type, log))
    cluster_results.extend(cluster_agglomerative(encoding, traces, enc_type, log))
    return cluster_results


path = "event_logs"
log_clustering = []
for file in tqdm(sort_files(os.listdir(path))):
    f_name = file.split(".csv")[0]
    df = pd.read_csv(f"{path}/{file}")
    traces, ids = get_traces(df)

    # one-hot
    onehot = one_hot(traces, f_name)
    log_clustering.extend(cluster(onehot, traces, "onehot", f_name))

    # ngrams
    bi_gram, tri_gram = n_grams(traces)
    log_clustering.extend(cluster(bi_gram, traces, "bi_gram", f_name))
    log_clustering.extend(cluster(tri_gram, traces, "tri_gram", f_name))

    # position profile
    profile_encoding = position_profile(df)
    log_clustering.extend(cluster(profile_encoding, traces, "position_profile", f_name))

pd.DataFrame(
    log_clustering,
    columns=["log", "encoding", "clustering", "silhouette", "variant_score", "time"],
).to_csv(f"clustering_metrics.csv", index=False)
