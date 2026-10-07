"""Study-level acquisition-descriptor baselines; no reports or identifiers as features."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID='StudyInstanceUID'
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']


def features(series,ids):
    # Fixed vocabulary is part of the acquisition schema, never a fit on test data.
    s=series.copy();out=s.groupby(ID).size().rename('series_count').to_frame()
    for flag in ['Fluid_Sensitive','Fat_Suppression']:
        out[flag+'_sum']=s.groupby(ID)[flag].sum()
        out[flag+'_fraction']=s.groupby(ID)[flag].mean()
    for plane in ['Sagittal','Coronal','Axial']:
        sub=s[s.Anatomical_Plane==plane];out[plane+'_count']=sub.groupby(ID).size()
        for flag in ['Fluid_Sensitive','Fat_Suppression']:
            out[plane+'_'+flag]=sub.groupby(ID)[flag].sum()
    s['both']=(s.Fluid_Sensitive.eq(1)&s.Fat_Suppression.eq(1)).astype(int)
    out['both_count']=s.groupby(ID)['both'].sum()
    return out.reindex(ids).fillna(0).astype(float)


def estimator(variant):
    if variant=='logistic':return make_pipeline(StandardScaler(),LogisticRegression(C=0.1,max_iter=2000,class_weight='balanced',random_state=42))
    if variant=='forest':return RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_leaf=5,max_features=0.7,class_weight='balanced_subsample',random_state=42,n_jobs=4)
    raise ValueError(variant)


def run(data,output,variant):
    data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(data/'train.csv');test=pd.read_csv(data/'test.csv')
    x=features(pd.read_csv(data/'train_series.csv'),train[ID])
    xt=features(pd.read_csv(data/'test_series.csv'),test[ID])
    prediction=pd.DataFrame({ID:test[ID]});metrics={}
    for label in LABELS:
        valid=train[label].notna();y=train.loc[valid,label].astype(int).to_numpy();xx=x.loc[valid.to_numpy()].to_numpy()
        n_splits=min(3,int(np.bincount(y,minlength=2).min()))
        if n_splits<2:raise RuntimeError('Insufficient labeled outcomes for validation: '+label)
        oof=np.empty(len(y))
        for ti,vi in StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42).split(xx,y):
            model=estimator(variant);model.fit(xx[ti],y[ti]);oof[vi]=model.predict_proba(xx[vi])[:,1]
        auc=float(roc_auc_score(y,oof));metrics[label]={'n_labeled':len(y),'positive':int(y.sum()),'oof_auc':auc,'folds':n_splits}
        model=estimator(variant);model.fit(xx,y);prediction[label]=model.predict_proba(xt.to_numpy())[:,1]
        print(label,metrics[label],flush=True)
    assert prediction[ID].is_unique and len(prediction)==len(test)
    assert np.isfinite(prediction[LABELS].to_numpy()).all() and prediction[LABELS].ge(0).all().all() and prediction[LABELS].le(1).all().all()
    result={'variant':variant,'macro_oof_auc':float(np.mean([m['oof_auc'] for m in metrics.values()])),'targets':metrics,'features':list(x),'limitation':'Only 58 labeled studies. Acquisition descriptors can reflect site/protocol confounding. No report features, image interpretation, clinical validation, or site-disjoint generalization claim. Per-target stratified folds use seed 42; fixed hyperparameters, no leaderboard tuning.'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2));prediction.to_csv(output/'submission.csv',index=False)
    print('Macro OOF AUC',result['macro_oof_auc'],flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--variant',choices=['logistic','forest'],required=True);a=p.parse_args();run(a.data,a.output,a.variant)
