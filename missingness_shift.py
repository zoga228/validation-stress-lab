"""Train-only preprocessing and stress testing an unstable shortcut feature."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from make_data import missingness


def run():
    rows = []
    for seed in (11, 22, 33):
        train = missingness(seed)
        tests = {"IID": missingness(seed+100, 3000), "shifted": missingness(seed+200, 3000, True)}
        for features in ("all", "stable only"):
            columns = [f"x{i}" for i in range(6)] + (["spurious"] if features == "all" else [])
            models = {
                "linear + missing flags": make_pipeline(SimpleImputer(add_indicator=True), StandardScaler(), LogisticRegression(max_iter=1000)),
                "histogram boosting": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=1, random_state=seed),
            }
            for name, model in models.items():
                model.fit(train[columns], train.target)
                for domain, test in tests.items():
                    p = model.predict_proba(test[columns])[:, 1]
                    rows.append({"seed": seed, "model": name, "features": features, "domain": domain, "auc": roc_auc_score(test.target, p), "log_loss": log_loss(test.target, p)})
    result = pd.DataFrame(rows)
    summary = result.groupby(["model", "features", "domain"])[["auc", "log_loss"]].mean()
    print(summary.to_string())
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = result.groupby(["model", "features", "domain"]).auc.mean().unstack()
    pivot.plot.bar(ax=ax, color=["#2563eb", "#f59e0b"])
    ax.set(ylabel="ROC AUC", ylim=(0, 1), title="High IID performance can depend on a reversible shortcut")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out=Path("outputs");out.mkdir(exist_ok=True)
    result.to_csv(out / "missingness-shift.csv", index=False)
    fig.savefig(out / "missingness-shift.png", dpi=160)
    plt.show()
    return result


if __name__ == "__main__":
    run()
