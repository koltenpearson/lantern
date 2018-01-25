

import argparse
from . import engine
from . import structures


def train_experiment(args) :
    model_info = structures.ModelInfo(args.model_dir)
    engine.train_model(model_info, args.data_dir, args.epochs, on_gpu=(not args.cpu), threads=args.threads, run_id=args.id)

parser = argparse.ArgumentParser("utilties for machine learning with pytorch")
sub_add = parser.add_subparsers()

train_parser = sub_add.add_parser('train', help="train a model") 
train_parser.add_argument('model_dir', help='model root directory')
train_parser.add_argument('data_dir', help='root directory of dataset')
train_parser.add_argument('--cpu', help='use cpu instead of gpu', action='store_true')
train_parser.add_argument('--epochs', help='target epoch to train to', default='30', type=int)
train_parser.add_argument('--threads', help='number of threads to use', default=2, type=int)
train_parser.add_argument('--id', help='continue the run with this id', default=-1, type=int)
train_parser.set_defaults(func=train_experiment)

args = parser.parse_args()
args.func(args)



