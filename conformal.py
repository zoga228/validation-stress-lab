"""Finite-sample split conformal: marginal coverage versus covariate-shift failure."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from make_data import regression


def conformal_quantile(scores, alpha):
    k = math.ceil((len(scores)+1)*(1-alpha))
    if k > len(scores):
        return float("inf")
    return float(np.sort(scores)[k-1])


def run():
    train,cal=regression(42),regression(43,2000)
    model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=15,l2_regularization=2,random_state=42).fit(train[["x"]],train.target)
    residual=np.abs(cal.target.to_numpy()-model.predict(cal[["x"]]))
    rows=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,(domain,test) in zip(axes,(("IID",regression(44,3000)),("covariate shift",regression(45,3000,True)))):
        pred=model.predict(test[["x"]])
        for alpha in (.05,.1,.2):
            q=conformal_quantile(residual,alpha)
            covered=np.abs(test.target.to_numpy()-pred)<=q
            rows.append({"domain":domain,"target_coverage":1-alpha,"coverage":covered.mean(),"mean_width":2*q,"n_test":len(test)})
        q=conformal_quantile(residual,.1)
        order=np.argsort(test.x.to_numpy())
        take=order[::10]
        ax.scatter(test.x.iloc[take],test.target.iloc[take],s=6,alpha=.3)
        ax.plot(test.x.iloc[order],pred[order],color="#2563eb")
        ax.fill_between(test.x.iloc[order],pred[order]-q,pred[order]+q,alpha=.15,color="#2563eb")
        ax.set(title=domain,xlabel="x")
        # Conditional coverage is diagnostic; split conformal does not promise it.
        for label,mask in (("|x| <= 1",np.abs(test.x.to_numpy())<=1),("|x| > 1",np.abs(test.x.to_numpy())>1)):
            print(domain,label,"90% interval coverage",np.mean((np.abs(test.target.to_numpy()-pred)<=q)[mask]),"n=",mask.sum())
    result=pd.DataFrame(rows);print(result.to_string(index=False))
    axes[0].set_ylabel("Target and prediction interval");fig.suptitle("Exchangeability matters for conformal coverage");fig.tight_layout()
    out=Path("outputs");out.mkdir(exist_ok=True)
    result.to_csv(out / "conformal.csv",index=False)
    fig.savefig(out / "conformal.png",dpi=160);plt.show()
    return result


if __name__ == "__main__":
    run()
