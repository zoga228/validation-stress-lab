"""Quantify the deployment question that random versus grouped CV actually answers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from make_data import grouped


def run():
    rows = []
    for seed in (11, 22, 33, 44, 55):
        df = grouped(seed)
        x, y = df.drop(columns="target"), df.target
        random_train, random_test = train_test_split(np.arange(len(df)), test_size=.25, stratify=y, random_state=seed)
        group_train, group_test = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(x, y, df.group_id))
        for protocol, tr, te in (("same groups: random split", random_train, random_test), ("new groups: group split", group_train, group_test)):
            preprocess = ColumnTransformer([("group", OneHotEncoder(handle_unknown="ignore"), ["group_id"]), ("numeric", StandardScaler(), ["sensor_noise"])])
            model = make_pipeline(preprocess, LogisticRegression(max_iter=1000))
            model.fit(x.iloc[tr], y.iloc[tr])
            overlap = len(set(df.group_id.iloc[tr]) & set(df.group_id.iloc[te]))
            if protocol.startswith("new"):
                assert overlap == 0
            rows.append({"seed": seed, "protocol": protocol, "auc": roc_auc_score(y.iloc[te], model.predict_proba(x.iloc[te])[:, 1]), "overlapping_groups": overlap})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    print(result.groupby("protocol").auc.agg(["mean", "std"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (protocol, frame) in enumerate(result.groupby("protocol")):
        ax.scatter(np.full(len(frame), i), frame.auc, s=60, alpha=.7)
        ax.errorbar(i, frame.auc.mean(), yerr=frame.auc.std(), color="#153e75", fmt="D", capsize=8)
    ax.set_xticks([0, 1], sorted(result.protocol.unique()))
    ax.set(ylim=(.35, 1), ylabel="ROC AUC", title="Identity leakage depends on the deployment population")
    ax.axhline(.5, ls="--", color="gray")
    fig.tight_layout()
    out = Path("outputs"); out.mkdir(exist_ok=True)
    result.to_csv(out / "group-validation.csv", index=False)
    fig.savefig(out / "group-validation.png", dpi=160)
    plt.show()
    return result


if __name__ == "__main__":
    run()
