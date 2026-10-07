"""Frozen ImageNet embeddings from actual MRI slices with study-level CV.

Independent pooling/validation implementation. Public torchvision ResNet18
weights are not trained by this account. No reports or IDs are model inputs.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
import pydicom
from torchvision.models import resnet18,ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ID='StudyInstanceUID'
LABELS=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
PLANES=['Sagittal','Coronal','Axial']


def slice_indices(length):
    if length<1:return []
    return sorted(set([min(length-1,int(length*.35)),min(length-1,int(length*.65))]))


def normalise_pixels(pixels,photometric='MONOCHROME2'):
    a=np.asarray(pixels,dtype=np.float32)
    if a.ndim==3:a=a[len(a)//2]
    if a.ndim!=2 or not np.isfinite(a).any():raise ValueError('Invalid grayscale slice')
    low,high=np.percentile(a[np.isfinite(a)],[1,99]);a=np.nan_to_num(a,nan=float(low))
    a=np.clip((a-low)/max(float(high-low),1e-6),0,1)
    if photometric=='MONOCHROME1':a=1-a
    return np.rint(a*255).astype(np.uint8)


def ordered_slices(folder):
    files=list(folder.glob('*.dcm'));ranked=[]
    for path in files:
        try:
            header=pydicom.dcmread(path,stop_before_pixels=True,specific_tags=['InstanceNumber'],force=True)
            ranked.append((int(getattr(header,'InstanceNumber',0)),path.name,path))
        except Exception:ranked.append((0,path.name,path))
    return [x[2] for x in sorted(ranked)]


def embedding_features(data,subset,series,ids,model,transform,device):
    groups={str(key):rows for key,rows in series.groupby(ID,sort=False)};vectors=[]
    diagnostics={'studies':len(ids),'decoded_slices':0,'failed_slices':0,'missing_series':0,'studies_without_images':0,'examples':[]}
    started=time.monotonic()
    for number,study_id in enumerate(ids):
        rows=groups.get(str(study_id));images=[];slots=[]
        if rows is not None:
            for plane_index,plane in enumerate(PLANES):
                choices=rows[rows.Anatomical_Plane==plane].sort_values(['Fluid_Sensitive','Fat_Suppression','SeriesInstanceUID'],ascending=[False,False,True])
                if choices.empty:continue
                # One fixed preferred sequence per plane, two central slices.
                row=choices.iloc[0];folder=data/(subset+'_series')/str(study_id)/str(row.SeriesInstanceUID)
                paths=ordered_slices(folder)
                if not paths:diagnostics['missing_series']+=1;continue
                for index in slice_indices(len(paths)):
                    try:
                        ds=pydicom.dcmread(paths[index]);pixels=ds.pixel_array.astype(np.float32)
                        pixels=pixels*float(getattr(ds,'RescaleSlope',1))+float(getattr(ds,'RescaleIntercept',0))
                        array=normalise_pixels(pixels,str(getattr(ds,'PhotometricInterpretation','MONOCHROME2')))
                        images.append(transform(Image.fromarray(array).convert('RGB')));slots.append(plane_index)
                        diagnostics['decoded_slices']+=1
                    except Exception as exc:
                        diagnostics['failed_slices']+=1
                        if len(diagnostics['examples'])<5:diagnostics['examples'].append(type(exc).__name__+': '+str(exc)[:200])
        pooled=np.zeros((3,512),dtype=np.float32);counts=np.zeros(3,dtype=np.float32)
        if images:
            with torch.inference_mode():emb=model(torch.stack(images).to(device)).float().cpu().numpy()
            for slot,vector in zip(slots,emb):pooled[slot]+=vector;counts[slot]+=1
            pooled/=np.maximum(counts[:,None],1)
        else:diagnostics['studies_without_images']+=1
        vectors.append(np.concatenate([pooled.flatten(),(counts>0).astype(np.float32),counts]))
        if (number+1)%25==0:print(subset,number+1,'studies; decoded',diagnostics['decoded_slices'],'seconds',round(time.monotonic()-started,1),flush=True)
    diagnostics['seconds']=time.monotonic()-started
    if diagnostics['decoded_slices']==0:raise RuntimeError('No MRI pixels decoded; refusing an image-model claim')
    if diagnostics['studies_without_images']/max(len(ids),1)>.05:raise RuntimeError('Over 5% of studies lack usable image features: '+str(diagnostics))
    return np.asarray(vectors),diagnostics


def acquisition_features(series,ids):
    s=series.copy();out=s.groupby(ID).size().rename('series_count').to_frame()
    for flag in ['Fluid_Sensitive','Fat_Suppression']:
        out[flag+'_sum']=s.groupby(ID)[flag].sum();out[flag+'_fraction']=s.groupby(ID)[flag].mean()
    for plane in PLANES:
        sub=s[s.Anatomical_Plane==plane];out[plane+'_count']=sub.groupby(ID).size()
        for flag in ['Fluid_Sensitive','Fat_Suppression']:out[plane+'_'+flag]=sub.groupby(ID)[flag].sum()
    out['both_count']=s.assign(both=(s.Fluid_Sensitive.eq(1)&s.Fat_Suppression.eq(1)).astype(int)).groupby(ID)['both'].sum()
    return out.reindex(ids).fillna(0).astype(float).to_numpy()


def classifier():
    # Fixed regularization; the scaler is fitted inside each training fold.
    return make_pipeline(StandardScaler(),LogisticRegression(C=.01,max_iter=2000,class_weight='balanced',random_state=42))


def fit_predict(x,xt,train,test,output,variant):
    predictions=pd.DataFrame({ID:test[ID]});scores={}
    for label in LABELS:
        valid=train[label].notna().to_numpy();y=train.loc[valid,label].astype(int).to_numpy();xx=x[valid]
        folds=min(3,int(np.bincount(y,minlength=2).min()))
        if folds<2:raise RuntimeError('Insufficient outcomes: '+label)
        oof=np.empty(len(y))
        for ti,vi in StratifiedKFold(n_splits=folds,shuffle=True,random_state=42).split(xx,y):
            estimator=classifier();estimator.fit(xx[ti],y[ti]);oof[vi]=estimator.predict_proba(xx[vi])[:,1]
        estimator=classifier();estimator.fit(xx,y);predictions[label]=estimator.predict_proba(xt)[:,1]
        scores[label]={'n_labeled':len(y),'positive':int(y.sum()),'oof_auc':float(roc_auc_score(y,oof)),'folds':folds}
        print(variant,label,scores[label],flush=True)
    assert list(predictions[ID])==list(test[ID]) and predictions[ID].is_unique
    assert np.isfinite(predictions[LABELS].to_numpy()).all() and predictions[LABELS].ge(0).all().all() and predictions[LABELS].le(1).all().all()
    metrics={'variant':variant,'macro_oof_auc':float(np.mean([x['oof_auc'] for x in scores.values()])),'targets':scores,'feature_dimension':x.shape[1]}
    predictions.to_csv(output/('submission-'+variant+'.csv'),index=False)
    return metrics


def run(data,output,weights_path):
    data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=True);torch.manual_seed(42);torch.set_num_threads(4)
    train=pd.read_csv(data/'train.csv');train=train[train[LABELS].notna().any(axis=1)].reset_index(drop=True);test=pd.read_csv(data/'test.csv')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model=resnet18(weights=None);model.load_state_dict(torch.load(weights_path,map_location='cpu',weights_only=True),strict=True);model.fc=torch.nn.Identity();model=model.to(device).eval()
    ts=pd.read_csv(data/'train_series.csv');vs=pd.read_csv(data/'test_series.csv');transform=ResNet18_Weights.IMAGENET1K_V1.transforms()
    x,train_diagnostics=embedding_features(data,'train',ts,train[ID],model,transform,device)
    xt,test_diagnostics=embedding_features(data,'test',vs,test[ID],model,transform,device)
    results={}
    results['image']=fit_predict(x,xt,train,test,output,'image')
    results['image-protocol']=fit_predict(np.concatenate([x,acquisition_features(ts,train[ID])],axis=1),np.concatenate([xt,acquisition_features(vs,test[ID])],axis=1),train,test,output,'image-protocol')
    np.savez_compressed(output/'study-features.npz',train=x,test=xt)
    report={'backbone':'Official torchvision ResNet18 ImageNet-1K; frozen, not trained by this account','device':device,'selection':'Two prespecified feature sets; C=.01 and per-target stratified 3-fold seed42 fixed before predictions. No leaderboard tuning.','limitation':'58 labeled studies; large feature dimension, possible site/protocol confounding, no site-disjoint or clinical validation. ImageNet transfer is not a specialist MRI model.','train_diagnostics':train_diagnostics,'test_diagnostics':test_diagnostics,'variants':results}
    (output/'image-metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
    return report
