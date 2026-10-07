"""Reproduce an acquisition-only comparator at a declared regularization."""
import argparse
from pathlib import Path
import metadata_baseline as baseline

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--regularization',type=float,choices=[.01,.1],default=.01)
    args=parser.parse_args()
    baseline.estimator=lambda variant:baseline.make_pipeline(
        baseline.StandardScaler(),baseline.LogisticRegression(
            C=args.regularization,max_iter=2000,class_weight='balanced',random_state=42))
    baseline.run(args.data,args.output,'logistic')
