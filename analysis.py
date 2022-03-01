import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted
from statistics import mode
import holoviews as hv
from holoviews import opts, dim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from skmultilearn.problem_transform import (
    BinaryRelevance,
    ClassifierChain,
    LabelPowerset,
)
import shap
import warnings

warnings.filterwarnings("ignore")


def silhouette_per_target(df, outpath="analysis"):
    """
    Silhouette performance by target
    """

    plt.figure(figsize=(25, 6))
    ax = sns.barplot(x="clustering", y="silhouette", hue="encoding", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(f"{outpath}/silhouette_encoding_bar.pdf")
    plt.close()

    plt.figure(figsize=(30, 10))
    ax = sns.lineplot(x="log", y="silhouette", hue="encoding", data=df)
    ax.tick_params(axis="x", labelrotation=90)
    plt.tight_layout()
    plt.savefig(f"{outpath}/silhouette_encoding_log.pdf")
    plt.close()

    plt.figure(figsize=(30, 10))
    ax = sns.lineplot(x="log", y="silhouette", hue="clustering", data=df)
    ax.tick_params(axis="x", labelrotation=90)
    plt.tight_layout()
    plt.savefig(f"{outpath}/silhouette_clustering_log.pdf")
    plt.close()


def variant_score_per_target(df, outpath="analysis"):
    """
    Variant score performance by target
    """

    plt.figure(figsize=(25, 6))
    ax = sns.barplot(x="clustering", y="variant_score", hue="encoding", data=df)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    plt.savefig(f"{outpath}/variant_score_encoding_bar.pdf")
    plt.close()

    plt.figure(figsize=(30, 10))
    ax = sns.lineplot(x="log", y="variant_score", hue="encoding", data=df)
    ax.tick_params(axis="x", labelrotation=90)
    plt.tight_layout()
    plt.savefig(f"{outpath}/variant_score_encoding_log.pdf")
    plt.close()

    plt.figure(figsize=(30, 10))
    ax = sns.lineplot(x="log", y="variant_score", hue="clustering", data=df)
    ax.tick_params(axis="x", labelrotation=90)
    plt.tight_layout()
    plt.savefig(f"{outpath}/variant_score_clustering_log.pdf")
    plt.close()


def heatmaps(df, metrics, outpath="analysis"):
    """
    Heatmaps of average positions for clustering and encoding methods
    """

    # clustering
    final_rank = df.groupby("clustering")[metrics].mean()
    final_rank.columns = ["Silhouette", "Variant Score", "Time"]
    final_rank.index.name = "Clustering Algorithm"
    final_rank = final_rank.reindex(natsorted(final_rank.index))

    plt.figure(figsize=(11, 3))
    sns.heatmap(final_rank.transpose(), annot=True, fmt=".0f", cmap="YlGn")
    plt.yticks(rotation=0)
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(f"{outpath}/clustering_ranking.pdf", dpi=300)
    plt.close()

    # encoding
    final_rank = df.groupby("encoding")[metrics].mean()
    final_rank.columns = ["Silhouette", "Variant Score", "Time"]
    final_rank.index.name = "Encoding Algorithm"
    final_rank = final_rank.reindex(natsorted(final_rank.index))

    plt.figure(figsize=(3, 3))
    sns.heatmap(final_rank.transpose(), annot=True, fmt=".0f", cmap="BuPu")
    plt.yticks(rotation=0)
    plt.ylabel("Metric")
    plt.tight_layout()
    plt.savefig(f"{outpath}/encoding_ranking.pdf", dpi=300)
    plt.close()

    # encoding + clustering
    final_rank = df.groupby("enc_clus")[metrics].mean()
    final_rank.columns = ["Silhouette", "Variant Score", "Time"]
    final_rank.index.name = "Encoding + Clustering Algorithm"

    plt.figure(figsize=(8, 25))
    sns.heatmap(final_rank, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{outpath}/enc_clus_ranking.pdf", dpi=300)
    plt.close()


def create_meta_database(df_targets, df_features, metrics):
    # Rotulating meta-target using the mean rankings
    df_targets["target_final_rank"] = df_targets.filter(metrics).mean(axis=1)

    # Finding the best techniques (ranking = min)
    best = pd.DataFrame(df_targets.groupby("log")["target_final_rank"].min())
    best.reset_index(level=0, inplace=True)

    # Filtering meta-instances where target_final_rank = min
    meta_target = df_targets.set_index(["log", "target_final_rank"]).join(
        best.set_index("log", "target_final_rank")
    )
    meta_target.reset_index(level=0, inplace=True)
    meta_target = meta_target[meta_target.index == meta_target.target_final_rank]
    meta_target.reset_index(drop=True, inplace=True)

    # Removing repetitions
    meta_target.drop_duplicates(subset="log", keep=False, inplace=True)

    meta_database = (
        meta_target.filter(["log", "clustering", "encoding", "enc_clus"])
        .set_index(["log"])
        .join(df_features.set_index("log"))
    )
    meta_database.reset_index(level=0, inplace=True)
    print("Creating meta-database")
    print(
        "#instances:",
        len(meta_database),
        "| #classes (clustering):",
        len(meta_database["clustering"].unique()),
        "| #classes (enc+clus):",
        len(meta_database["enc_clus"].unique()),
    )
    print()
    # print("Clustering as class")
    # print(meta_database["clustering"].value_counts())
    # print()
    # print("Encoding + clustering as class")
    # print(meta_database["enc_clus"].value_counts())

    return meta_database


def clean_meta_database(df, target):
    """
    Excluding classes with less than 5 instances
    """

    df_clean = df.loc[
        df[target].isin(df[target].value_counts().index[df[target].value_counts() > 4])
    ]
    print("Removing classes with low occurrence")
    print("Old shape:", df.shape, "| New shape:", df_clean.shape)
    print()

    return df_clean


def chord_diagrams(df, outpath="analysis"):
    """
    Chord Diagram for Cardinality Evaluation
    """

    names = list(df["clustering"].value_counts().keys())
    names.extend(df["encoding"].value_counts().keys())

    sources_dict = {}
    for x, y in zip(df["encoding"].value_counts().keys(), range(12, 16)):
        sources_dict[x] = y

    target_dict = {}
    for x, y in zip(df["clustering"].value_counts().keys(), range(12)):
        target_dict[x] = y

    sources, targets = [], []
    for enc_clus in df["enc_clus"].value_counts().keys():
        for enc in sources_dict:
            if enc in enc_clus:
                sources.append(sources_dict[enc])
                break
        for clus in target_dict:
            if clus in enc_clus:
                targets.append(target_dict[clus])

    hv.extension("matplotlib")
    input = pd.DataFrame(
        {
            "name": names,
            "group": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 16, 16, 16],
        }
    )

    links_ = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "value": df["enc_clus"].value_counts().tolist(),
        }
    )

    input.columns = ["name", "group"]
    input.reset_index(inplace=True)
    nodes_ = hv.Dataset(input, "index")

    hv.output(size=200, dpi=300)
    chord = hv.Chord((links_, nodes_))
    chord.opts(
        opts.Chord(
            cmap="tab20",
            edge_cmap="tab20",
            edge_color=dim("target").str(),
            labels="name",
            node_color=dim("group"),
        )
    )
    hv.output(chord, fig="png")
    hv.save(obj=chord, filename=f"{outpath}/chord_clustering.png")

    hv.output(size=200, dpi=300)
    chord = hv.Chord((links_, nodes_))
    chord.opts(
        opts.Chord(
            cmap="Accent",
            edge_cmap="Accent",
            edge_color=dim("source").str(),
            labels="name",
            node_color=dim("group"),
        )
    )
    hv.output(chord, fig="png")
    hv.save(obj=chord, filename=f"{outpath}/chord_encoding.png")


def multi_class_rf(X, y):
    """
    Multi-class solution (clustering)
    """

    result_df = pd.DataFrame()
    for step in range(30):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=step
        )
        scaler = Normalizer().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier(random_state=step)
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)
        result_df = pd.concat(
            [
                result_df,
                pd.DataFrame(
                    [
                        [
                            rf.score(X_test, y_test),
                            f1_score(y_test, y_pred, average="weighted"),
                        ]
                    ]
                ),
            ]
        )

    result_df.columns = ["Acc", "F1"]
    return result_df


def multi_class_pipe(df_targets, df_features):
    """
    Multi-class classification experiment
    """
    print("Classification experiments", end="\n\n")
    print("Multi-class RF with silhouette, variant score and time as metrics")
    metrics = ["silhouette_rank", "variant_rank", "time_rank"]
    df_meta_database = create_meta_database(df_targets, df_features, metrics)
    df_meta_database = clean_meta_database(df_meta_database, "clustering")

    label = list(df_meta_database["clustering"])
    X = df_meta_database.drop(["log", "clustering", "encoding", "enc_clus"], axis=1)

    print("Meta-database shape:", X.shape)
    results = multi_class_rf(X, label)
    print("Mean performance")
    print(results.mean(), end="\n\n")

    print("Multi-class RF with silhouette and variant score (excluding time)")
    metrics = ["silhouette_rank", "variant_rank"]
    df_meta_database = create_meta_database(df_targets, df_features, metrics)
    df_meta_database = clean_meta_database(df_meta_database, "clustering")

    label = list(df_meta_database["clustering"])
    X = df_meta_database.drop(["log", "clustering", "encoding", "enc_clus"], axis=1)

    print("Meta-database shape:", X.shape)
    results = multi_class_rf(X, label)
    print("Mean performance")
    print(results.mean(), end="\n\n")


def multi_output_clf(transform_type, seed):
    if transform_type == "br":
        return BinaryRelevance(RandomForestClassifier(random_state=seed))
    elif transform_type == "cc":
        return ClassifierChain(RandomForestClassifier(random_state=seed), order=[0, 1])
    elif transform_type == "lp":
        return LabelPowerset(RandomForestClassifier(random_state=seed))


def multi_output_pipe(X, y, transform_type):
    out = []
    for step in range(30):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, random_state=step
        )
        scaler = Normalizer().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        clf = multi_output_clf(transform_type, step)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        preds = pd.DataFrame(predictions.toarray(), columns=["clustering", "encoding"])
        out.append(
            [
                accuracy_score(y_test["clustering"], preds["clustering"]),
                accuracy_score(y_test["encoding"], preds["encoding"]),
                f1_score(y_test["clustering"], preds["clustering"], average="weighted"),
                f1_score(y_test["encoding"], preds["encoding"], average="weighted"),
            ]
        )

    results = pd.DataFrame(out, columns=["Acc_clus", "Acc_enc", "F1_clus", "F1_enc"])
    print("Mean performance")
    print(results.mean(), end="\n\n")


def tuning_hps(X_train, y_train, seed):
    """
    Tuning hyperparameters
    """
    # Creating pipeline with placeholder estimator
    pipe = Pipeline([("clf", DummyClassifier())])

    # Learning algorithms and their hyperparameters
    parameters = [
        {
            "clf": [RandomForestClassifier(random_state=seed)],
            "clf__n_estimators": [10, 25, 50, 75, 100, 150, 200],
            "clf__criterion": ["gini", "entropy"],
            "clf__min_samples_split": [2, 3, 4, 5, 10],
            "clf__min_samples_leaf": [1, 3, 5, 10],
            "clf__max_features": ["sqrt", "log2"],
        }
    ]
    scaler = Normalizer().fit(X_train)

    # Applying grid search
    clf = GridSearchCV(
        pipe,
        parameters,
        scoring=["accuracy", "f1_weighted"],
        refit="f1_weighted",
        n_jobs=-1,
    )
    clf.fit(scaler.transform(X_train), y_train)

    print("Best configuration")
    print(clf.best_params_)
    print("Best F1")
    print(clf.best_score_, end="\n\n")


def _predict_loop(X_train, X_test, y_train, y_test):
    """
    Predicting on a given target (y_)
    """
    # Normalizing data
    scaler = Normalizer().fit(X_train)

    out = []
    for i in range(30):
        # Creating classifier with tuned hyperparameters
        clf = RandomForestClassifier(
            criterion="gini",
            max_features="log2",
            min_samples_leaf=1,
            min_samples_split=3,
            n_estimators=50,
            random_state=i,
            n_jobs=-1,
        )
        clf.fit(scaler.transform(X_train), y_train)
        y_pred = clf.predict(scaler.transform(X_test))

        out.append(
            [
                accuracy_score(y_test, y_pred),
                f1_score(y_test, y_pred, average="weighted"),
            ]
        )

    return pd.DataFrame(out, columns=["Accuracy", "F1"])


def predict_tuned(X_train, X_test, y_train, y_test):
    """
    Using the selected hyperparameters in the (unseen) test data
    """
    results_clustering = _predict_loop(
        X_train, X_test, y_train["clustering"], y_test["clustering"]
    )

    print("Clustering")
    print(
        f"Accuracy (std): {np.round(results_clustering['Accuracy'].mean(), 3)} ({np.round(results_clustering['Accuracy'].std(), 3)})"
    )
    print(
        f"F1 (std): {np.round(results_clustering['F1'].mean(), 3)} ({np.round(results_clustering['F1'].std(), 3)})",
        end="\n\n",
    )

    results_encoding = _predict_loop(
        X_train, X_test, y_train["encoding"], y_test["encoding"]
    )

    print("Encoding")
    print(
        f"Accuracy (std): {np.round(results_encoding['Accuracy'].mean(), 3)} ({np.round(results_encoding['Accuracy'].std(), 3)})"
    )
    print(
        f"F1 (std): {np.round(results_encoding['F1'].mean(), 3)} ({np.round(results_encoding['F1'].std(), 3)})",
        end="\n\n",
    )

    results = results_encoding.append(results_clustering)
    results["Step"] = "Clustering"
    results.iloc[0:30, 2] = "Encoding"
    results = results.melt(id_vars=["Step"])

    results.columns = ["Algorithms", "Performance", "Value"]
    results.insert(0, "Method", "Meta-Model")

    return results


def fill_df(values, method, algorithm, performance):
    df = pd.DataFrame(values, columns=["Value"])
    df.insert(0, "Method", method)
    df.insert(1, "Algorithms", algorithm)
    df.insert(2, "Performance", performance)
    return df


def baseline_comparison(df, y_test):
    """
    Majority voting and random approaches as baseline comparisons
    """
    # Random and majority approaches
    labels_clus = df["clustering"].astype("category").cat.codes
    labels_enc = df["encoding"].astype("category").cat.codes
    random_clus_f1, random_enc_f1 = [], []
    maj_clus_f1, maj_enc_f1 = [], []
    for step in range(30):
        random_y = np.random.randint(len(set(labels_clus)), size=(1, len(y_test)))
        random_clus_f1.append(
            f1_score(y_test["clustering"], random_y[0], average="weighted")
        )

        majority_y = [mode(y_test["clustering"])] * len(y_test)
        maj_clus_f1.append(
            f1_score(y_test["clustering"], majority_y, average="weighted")
        )

        random_y = np.random.randint(len(set(labels_enc)), size=(1, len(y_test)))
        random_enc_f1.append(
            f1_score(y_test["encoding"], random_y[0], average="weighted")
        )

        majority_y = [mode(y_test["encoding"])] * len(y_test)
        maj_enc_f1.append(f1_score(y_test["encoding"], majority_y, average="weighted"))

    df_rand_clus_f1 = fill_df(random_clus_f1, "Random", "Clustering", "F1")
    df_rand_enc_f1 = fill_df(random_enc_f1, "Random", "Encoding", "F1")
    df_maj_clus_f1 = fill_df(maj_clus_f1, "Majority", "Clustering", "F1")
    df_maj_enc_f1 = fill_df(maj_enc_f1, "Majority", "Encoding", "F1")

    print("Random approach")
    print(
        f"Clustering F1 (std): {np.round(df_rand_clus_f1['Value'].mean(), 3)} ({np.round(df_rand_clus_f1['Value'].std(), 3)})"
    )
    print(
        f"Encoding F1 (std): {np.round(df_rand_enc_f1['Value'].mean(), 3)} ({np.round(df_rand_enc_f1['Value'].std(), 3)})",
        end="\n\n",
    )

    print("Majority voting")
    print(
        f"Clustering F1 (std): {np.round(df_maj_clus_f1['Value'].mean(), 3)} ({np.round(df_maj_clus_f1['Value'].std(), 3)})"
    )
    print(
        f"Encoding F1 (std): {np.round(df_maj_enc_f1['Value'].mean(), 3)} ({np.round(df_maj_enc_f1['Value'].std(), 3)})",
        end="\n\n",
    )

    return df_rand_clus_f1, df_rand_enc_f1, df_maj_clus_f1, df_maj_enc_f1


def f1_per_target(df, outpath="analysis"):
    plt.figure(figsize=(6, 2.5))
    ax = sns.boxplot(
        x="Value",
        y="Method",
        hue="Algorithms",
        data=df[df["Performance"] == "F1"].sort_values(["Method"]),
        palette="Accent",
    )
    ax.axhline(0.5, ls="--")
    ax.axhline(1.5, ls="--")
    ax.set_xlabel("F1")
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(f"{outpath}/f1_per_target.pdf", dpi=300)
    plt.close()


def global_interpretability(df, seed, outpath="analysis"):
    """
    Generates plots for encoding and clustering interpretability
    """

    X = df.drop(["log", "clustering", "encoding", "enc_clus"], axis=1)
    y = df[["clustering", "encoding"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=seed
    )

    clf = RandomForestClassifier(
        criterion="gini",
        max_features="log2",
        min_samples_leaf=1,
        min_samples_split=3,
        n_estimators=50,
        random_state=random_seed,
        n_jobs=-1,
    )

    # clustering
    clf.fit(X_train, y_train["clustering"])
    y_pred = clf.predict(X_test)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    plt.figure(figsize=(10, 7))
    ax = shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        class_names=y["clustering"].unique(),
        color=plt.get_cmap("tab20"),
        plot_size=0.20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{outpath}/global_interpretability_clustering.pdf")
    plt.close()

    # encoding
    clf.fit(X_train, y_train["encoding"])
    y_pred = clf.predict(X_test)

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_train)
    plt.figure(figsize=(10, 7))
    ax = shap.summary_plot(
        shap_values,
        X_train,
        plot_type="bar",
        class_names=["bi-gram", "onehot", "position profile"],
        plot_size=0.20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"{outpath}/global_interpretability_encoding.pdf")
    plt.close()


# Create output directory
out_path = "analysis"
os.makedirs(out_path, exist_ok=True)

# Importing meta-features
df_logs = pd.read_csv("log_meta_features.csv")

# Importing clustering metrics
df_models = pd.read_csv("clustering_metrics.csv")
df_models.insert(3, "enc_clus", df_models["encoding"] + "_" + df_models["clustering"])

print("Plotting metric analysis per target")
silhouette_per_target(df_models)
variant_score_per_target(df_models)

# Creating ranks for each metric (silhouette, variant, time)
df_models["silhouette_rank"] = df_models.groupby("log")["silhouette"].rank(
    method="min", ascending=False, na_option="bottom"
)
df_models["variant_rank"] = df_models.groupby("log")["variant_score"].rank(
    method="min", ascending=True, na_option="bottom"
)
df_models["time_rank"] = df_models.groupby("log")["time"].rank(
    method="min", ascending=True, na_option="bottom"
)

metrics = ["silhouette_rank", "variant_rank", "time_rank"]

# Plotting positional heatmaps
heatmaps(df_models, metrics)

# Meta-database creation
df_meta_database = create_meta_database(df_models, df_logs, metrics)

# Plotting chord diagrams
chord_diagrams(df_meta_database)

# Multi-class experiment
multi_class_pipe(df_models, df_logs)

print("Multi-output solution", end="\n\n")
df_meta_database = clean_meta_database(df_meta_database, "enc_clus")

# Preparing meta-database
X = df_meta_database.drop(["log", "clustering", "encoding", "enc_clus"], axis=1)
y = pd.DataFrame(columns=["clustering", "encoding"])
y["clustering"] = df_meta_database["clustering"].astype("category").cat.codes
y["encoding"] = df_meta_database["encoding"].astype("category").cat.codes

print("Binary relevance")
multi_output_pipe(X, y, "br")
print("Classifier chain")
multi_output_pipe(X, y, "cc")
print("Label powerset")
multi_output_pipe(X, y, "lp")

print("Tuning hyperparameters")
random_seed = 30
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=random_seed
)
tuning_hps(X_train, y_train["clustering"], random_seed)

print("Meta-learning prediction")
results = predict_tuned(X_train, X_test, y_train, y_test)

# Baseline predictions
df_rand_clus_f1, df_rand_enc_f1, df_maj_clus_f1, df_maj_enc_f1 = baseline_comparison(
    df_meta_database, y_test
)
df_results = pd.concat(
    [results, df_rand_clus_f1, df_rand_enc_f1, df_maj_clus_f1, df_maj_enc_f1],
    ignore_index=True,
)

# Average and std performances for all methods
# print(df_results.groupby(["Method", "Performance", "Algorithms"]).mean())
# print(df_results.groupby(["Method", "Performance", "Algorithms"]).std())

f1_per_target(df_results)

# Explaining meta-features
print("Generating global interpretability plots", end="\n\n")
global_interpretability(df_meta_database, random_seed)
