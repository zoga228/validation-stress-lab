"""Run the sparse-label MRI transfer study on an official local dataset."""
import argparse
from pathlib import Path
from image_embeddings import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    args=parser.parse_args()
    run(args.data,args.output,args.weights)
