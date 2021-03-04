import data
import argparse
from model import EDSR
parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="data")
parser.add_argument("--imgsize",default=66,type=int)
parser.add_argument("--scale",default=3,type=int)
parser.add_argument("--layers",default=32,type=int)
parser.add_argument("--featuresize",default=128,type=int)
parser.add_argument("--batchsize",default=5,type=int)
parser.add_argument("--savedir",default='save1_models_128_20_re_20000')
parser.add_argument("--iterations",default=1,type=int)
args = parser.parse_args()
iters=data.load_dataset(args.dataset,args.iterations,args.batchsize)
print("iters:",iters)
down_size = args.imgsize//args.scale
network = EDSR(down_size,args.layers,args.featuresize,args.scale)
network.set_data_fn(data.get_batch,(args.batchsize,args.imgsize,down_size),data.get_test_set,(args.imgsize,down_size))
network.train(iters,args.savedir)
