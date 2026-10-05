"""Probability calibration with untouched evaluation and paired bootstrap intervals."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from make_data import missingness


def run():
    df = missingness(42, 12000)
    train, rest = train_test_split(df, test_size=.5, stratify=df.target, random_state=42)
    calibration, test = train_test_split(rest, test_size=.5, stratify=rest.target, random_state=43)
    features = [f"x{i}" for i in range(6)]
    model = make_pipeline(SimpleImputer(add_indicator=True), RandomForestClassifier(n_estimators=250, min_samples_leaf=5, max_features="sqrt", random_state=42, n_jobs=2))
    model.fit(train[features], train.target)
    def logit(p):
        p = np.clip(p, 1e-6, 1-1e-6)
        return np.log(p / (1-p)).reshape(-1, 1)
    raw_cal = model.predict_proba(calibration[features])[:, 1]
    calibrator = LogisticRegression(C=1e6).fit(logit(raw_cal), calibration.target)
    raw = model.predict_proba(test[features])[:, 1]
    calibrated = calibrator.predict_proba(logit(raw))[:, 1]
    y = test.target.to_numpy()
    metrics = pd.DataFrame([{"model": name, "auc": roc_auc_score(y, p), "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p)} for name, p in (("raw forest", raw), ("sigmoid calibrated", calibrated))])
    rng = np.random.default_rng(42)
    diffs=[]
    for _ in range(1000):
        ix=rng.integers(len(y),size=len(y))
        diffs.append(np.mean((calibrated[ix]-y[ix])**2-(raw[ix]-y[ix])**2))
    interval=np.quantile(diffs,[.025,.975])
    print(metrics.to_string(index=False))
    print("Paired test bootstrap, delta Brier (calibrated - raw), 95% CI:",interval)
    print("Rows train/calibration/test:",len(train),len(calibration),len(test))
    fig,ax=plt.subplots(figsize=(6,5))
    for name,p in (("raw forest",raw),("sigmoid calibrated",calibrated)):
        true,pred=calibration_curve(y,p,n_bins=10,strategy="quantile")
        ax.plot(pred,true,"o-",label=name)
    ax.plot([0,1],[0,1],"--",color="gray")
    ax.set(xlabel="Mean predicted probability",ylabel="Observed fraction positive",title="Calibration evaluated on untouched data")
    ax.legend();fig.tight_layout()
    out=Path("outputs");out.mkdir(exist_ok=True)
    metrics.to_csv(out / "calibration.csv",index=False)
    pd.DataFrame({"low":[interval[0]],"high":[interval[1]]}).to_csv(out / "calibration-bootstrap.csv",index=False)
    fig.savefig(out / "calibration.png",dpi=160)
    plt.show()
    return metrics


if __name__ == "__main__":
    run()
