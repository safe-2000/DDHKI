import argparse
import torch
import numpy as np
from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='music', help='which dataset to use (music, book, movie)')
parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
parser.add_argument('--batch_size', type=int, default=8192, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=8192, help='batch size')
parser.add_argument('--n_layer', type=int, default=3, help='depth of layer')
parser.add_argument('--lr', type=float, default=0.0035, help='learning rate')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of the l2 regularization term')


parser.add_argument('--dim', type=int, default=128, help='dimension of entity and relation embeddings')
parser.add_argument('--user_triple_set_size', type=int, default=32, help='the number of triples in triple set of user')
parser.add_argument('--item_triple_set_size', type=int, default=16, help='the number of triples in triple set of item')
parser.add_argument('--agg', type=str, default='concat', help='the type of aggregator (sum, pool, concat)')


parser.add_argument('--random_seed', type=int, default=42, help='random_seed')
parser.add_argument('--use_cuda', type=bool, default=True, help='whether using gpu or cpu')
parser.add_argument('--show_topk', type=bool, default=False, help='whether showing topk or not')
parser.add_argument('--random_flag', type=bool, default=False, help='whether using random seed or not')

parser.add_argument('--use_kg_cross', type=bool, default=True, help='whether use_kg_cross')
parser.add_argument('--temperature', type=float, default=15, help='temperature')
parser.add_argument('--use_cl_loss', type=bool, default=True, help='whether use_contrastive_loss')
parser.add_argument('--cl_loss_weight', type=float, default=0.02, help='contrastive_loss')
parser.add_argument('--use_knowledge_gate', type=bool, default=True, help='whether use_knowledge_gate')


# Diffusion 模型
parser.add_argument('--device', type=str, default='cuda', help='Device to run the diffusion model on. Choose between "cuda" and "cpu".')
parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
parser.add_argument('--steps', type=int, default=5, help='diffusion steps')
parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating') #0.1
parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating') #0.0001
parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')#0.02
parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
parser.add_argument('--sampling_steps', type=int, default=0, help='steps of the forward process during inference')
parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
parser.add_argument('--dims', type=list, default= [1000], help='the dims for the DNN')
parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
parser.add_argument('--emb_size', type=int, default=20, help='timestep embedding size')
parser.add_argument('--structure', type=str, default="UNet", help="transformer/ MLP /ResNet/ UNet" )
parser.add_argument('--num_heads', type=int, default=4, help="the number of attentation head")
parser.add_argument('--hiddenSize', type=int, default=128)  # best is 128


args = parser.parse_args()

print(" --------------  打印所有的参数 ----------------")
print(args)
print(" ---------------------------------------------")

def set_random_seed(torch_seed):
    np.random.seed(torch_seed)                  
    torch.manual_seed(torch_seed)       
    torch.cuda.manual_seed(torch_seed)      
    torch.cuda.manual_seed_all(torch_seed)  

if not args.random_flag:
    set_random_seed(args.random_seed)
    
data_info = load_data(args)
train(args, data_info)
    