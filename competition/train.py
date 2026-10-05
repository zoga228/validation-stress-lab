"""Airline satisfaction: fixed-fold LightGBM OOF evaluation and bounded experiments."""
import argparse
import json
from pathlib import Path
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

ROOT=Path(__file__).resolve().parent
CONFIGS=[
    {"name":"baseline-leaves31","leaves":31,"features":False,"regularization":1},
    {"name":"services-leaves31","leaves":31,"features":True,"regularization":1},
    {"name":"services-leaves63","leaves":63,"features":True,"regularization":3},
    {"name":"services-leaves15","leaves":15,"features":True,"regularization":1},
    {"name":"services-leaves63-reg10","leaves":63,"features":True,"regularization":10},
    {"name":"crossfit-target-encoding","leaves":63,"features":False,"regularization":5,"target_encoding":True},
]


def features(frame, enriched):
    x=frame.drop(columns=[c for c in ("id","satisfaction") if c in frame]).copy()
    if enriched:
        service=[c for c in x if any(word in c.lower() for word in ("service","comfort","booking","boarding","entertainment","handling","cleanliness","food","room")) and pd.api.types.is_numeric_dtype(x[c])]
        x["service_mean"]=x[service].mean(axis=1)
        x["service_min"]=x[service].min(axis=1)
        x["service_std"]=x[service].std(axis=1)
        x["service_low_count"]=(x[service]<=2).sum(axis=1)
        x["service_zero_count"]=(x[service]==0).sum(axis=1)
        x["delay_change"]=x["Arrival Delay in Minutes"]-x["Departure Delay in Minutes"]
        x["log_distance"]=np.log1p(x["Flight Distance"].clip(lower=0))
        x["log_departure_delay"]=np.log1p(x["Departure Delay in Minutes"].clip(lower=0))
    return x


def train(config, data_root=None, output_root=None, folds=3):
    data_root=Path(data_root or ROOT/"data")
    output_root=Path(output_root or ROOT/"runs")
    directory=output_root/config["name"]
    directory.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(data_root/"train.csv")
    test_raw=pd.read_csv(data_root/"test.csv")
    sample=pd.read_csv(data_root/"sample_submission.csv")
    y=raw.satisfaction.astype(int).to_numpy()
    x,xt=features(raw,config["features"]),features(test_raw,config["features"])
    # Vocabulary is derived from training features only; unseen test categories are missing.
    for col in x.select_dtypes(include=["object","string"]).columns:
        categories=x[col].fillna("__missing__").astype(str).unique()
        dtype=pd.CategoricalDtype(categories=categories)
        x[col]=x[col].fillna("__missing__").astype(str).astype(dtype)
        xt[col]=xt[col].fillna("__missing__").astype(str).astype(dtype)
    oof=np.zeros(len(x));prediction=np.zeros(len(xt));rows=[];importances=[]
    begin=time.perf_counter()
    for fold,(tr,va) in enumerate(StratifiedKFold(n_splits=folds,shuffle=True,random_state=42).split(x,y)):
        xtr,xva,xtest=x.iloc[tr].copy(),x.iloc[va].copy(),xt.copy()
        if config.get("target_encoding",False):
            cols=["Flight Distance","Age","Class","Type of Travel","Inflight wifi service","Online boarding"]
            encoder=TargetEncoder(target_type="binary",smooth="auto",cv=5,shuffle=True,random_state=42+fold)
            encoded_train=encoder.fit_transform(xtr[cols].astype(str),y[tr])
            encoded_valid=encoder.transform(xva[cols].astype(str))
            encoded_test=encoder.transform(xtest[cols].astype(str))
            for i,col in enumerate(cols):
                xtr["te_"+col]=encoded_train[:,i]
                xva["te_"+col]=encoded_valid[:,i]
                xtest["te_"+col]=encoded_test[:,i]
        model=lgb.LGBMClassifier(n_estimators=2200,learning_rate=.035,num_leaves=config["leaves"],min_child_samples=80,colsample_bytree=.9,subsample=.9,subsample_freq=1,reg_lambda=config["regularization"],random_state=42+fold,n_jobs=4,verbosity=-1,deterministic=True,force_col_wise=True)
        model.fit(xtr,y[tr],eval_set=[(xva,y[va])],eval_metric="auc",callbacks=[lgb.early_stopping(100,verbose=False)])
        oof[va]=model.predict_proba(xva)[:,1]
        prediction+=model.predict_proba(xtest)[:,1]/folds
        rows.append({"fold":fold,"auc":roc_auc_score(y[va],oof[va]),"best_iteration":model.best_iteration_})
        model.booster_.save_model(str(directory/f"fold{fold}.txt"))
        importances.append(model.booster_.feature_importance(importance_type="gain"))
        print(config["name"],rows[-1],flush=True)
    sample["satisfaction"]=prediction
    assert sample.id.equals(test_raw.id)
    assert sample.satisfaction.between(0,1).all() and np.isfinite(prediction).all()
    sample.to_csv(directory/"submission.csv",index=False)
    np.savez_compressed(directory/"predictions.npz",oof=oof,test=prediction)
    table=pd.DataFrame({"feature":xtr.columns,"mean_gain":np.mean(importances,axis=0)}).sort_values("mean_gain",ascending=False)
    table.to_csv(directory/"importance.csv",index=False)
    metrics={"config":config,"folds":rows,"oof_auc":roc_auc_score(y,oof),"oof_log_loss":log_loss(y,oof),"seconds":time.perf_counter()-begin,"train_rows":len(x),"test_rows":len(xt),"excluded_id":True,"lightgbm_version":lgb.__version__,"split_seed":42}
    directory.joinpath("metrics.json").write_text(json.dumps(metrics,indent=2))
    print("COMPLETE",metrics,flush=True)
    return metrics


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,default=0)
    args=parser.parse_args();train(CONFIGS[args.index])
