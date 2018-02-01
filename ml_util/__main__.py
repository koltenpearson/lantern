
#hack to allow cherrypy to autoreload when called with the -m 
if __name__ == '__main__' :
    import ml_util
    __name__ = 'ml_util'

import argparse
from . import engine
from . import structures
from .vis import server as vis

def train_experiment(args) :

    model_info = structures.ModelInfo(args.model_dir)
    engine.train_model(model_info, args.data_dir, args.epochs, on_gpu=(not args.cpu), threads=args.threads, run_id=args.id)


def vis_server(args) :
    print(__name__)
    vis.run_server(args.root_dir, args.port)

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


vis_parser = sub_add.add_parser('vis', help='run visualization server')
vis_parser.add_argument('root_dir', help='directory to run from', nargs='?', default='.')
vis_parser.add_argument('-p', '--port', help='port to run on', type=int, default=3030)
vis_parser.set_defaults(func=vis_server) 

args = parser.parse_args()
args.func(args)



