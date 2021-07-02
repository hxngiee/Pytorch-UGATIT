import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

import torch.backends.cudnn as cudnn
from train import *
from utils import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

cudnn.benchmark = True
cudnn.fastest = True

## setup parse
parser = argparse.ArgumentParser(description='Train the U-GAT-IT network',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num_workers', type=int, default=4, dest='num_workers', help='')
parser.add_argument('--gpu_id', type=int, default=0, dest='gpu_id')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, dest='gpu_devices', help="")
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('distributed', action='store_true', help='')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

parser.add_argument('--scope', default='ugatit', dest='scope')
parser.add_argument('--norm', type=str, default='inorm', choices=['bnorm', 'inorm', 'adaILN'], dest='norm')
parser.add_argument('--name_data', type=str, default='selfie2anime', dest='name_data')

parser.add_argument('--dir_checkpoint', default='./checkpoints', dest='dir_checkpoint')
parser.add_argument('--dir_log', default='./log', dest='dir_log')
parser.add_argument('--dir_data', default='./datasets', dest='dir_data')
parser.add_argument('--dir_result', default='./results', dest='dir_result')

parser.add_argument('--num_epoch', type=int,  default=300, dest='num_epoch')
parser.add_argument('--batch_size', type=int, default=64, dest='batch_size')

parser.add_argument('--lr_G', type=float, default=2e-4, dest='lr_G')
parser.add_argument('--lr_D', type=float, default=2e-4, dest='lr_D')

parser.add_argument('--num_freq_disp', type=int,  default=50, dest='num_freq_disp')
parser.add_argument('--num_freq_save', type=int,  default=10, dest='num_freq_save')

parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step', 'plateau', 'cosine'], dest='lr_policy')
parser.add_argument('--n_epochs', type=int, default=100, dest='n_epochs')
parser.add_argument('--n_epochs_decay', type=int, default=100, dest='n_epochs_decay')
parser.add_argument('--lr_decay_iters', type=int, default=50, dest='lr_decay_iters')

# weight init
parser.add_argument('--wgt_adv', type=float, default=1, dest='wgt_adv', help='Adv Weight')
parser.add_argument('--wgt_c', type=float, default=1000, dest='wgt_c', help='Cam Weight')
parser.add_argument('--wgt_cyc', type=float, default=10, dest='wgt_cyc', help='Cycle Weight')
parser.add_argument('--wgt_i', type=float, default=10, dest='wgt_i', help='Identity weight')

parser.add_argument('--optim', default='adam', choices=['sgd', 'adam', 'rmsprop'], dest='optim')
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--ny_in', type=int, default=256, dest='ny_in')
parser.add_argument('--nx_in', type=int, default=256, dest='nx_in')
parser.add_argument('--nch_in', type=int, default=3, dest='nch_in')

parser.add_argument('--ny_load', type=int, default=286, dest='ny_load')
parser.add_argument('--nx_load', type=int, default=286, dest='nx_load')
parser.add_argument('--nch_load', type=int, default=3, dest='nch_load')

parser.add_argument('--ny_out', type=int, default=256, dest='ny_out')
parser.add_argument('--nx_out', type=int, default=256, dest='nx_out')
parser.add_argument('--nch_out', type=int, default=3, dest='nch_out')

parser.add_argument('--nch_ker', type=int, default=64, dest='nch_ker')

parser.add_argument('--data_type', default='float32', dest='data_type')
parser.add_argument('--direction', default='A2B', dest='direction')

parser.add_argument('--nblk', type=int, default=4, dest='nblk')
parser.add_argument('--nblk_D', type=int, default=6, dest='nblk')

PARSER = Parser(parser)


if __name__ == '__main__':
    ARGS = PARSER.get_arguments()
    PARSER.write_args()
    PARSER.print_args()

    TRAINER = Train(ARGS)

    if ARGS.mode == 'train':
        if not ARGS.gpu_devices:
            TRAINER.train(ARGS.gpu_id,1,ARGS)
        else:
            ngpus_per_node = len(ARGS.gpu_devices)
            ARGS.world_size = ngpus_per_node * ARGS.world_size
            mp.spawn(TRAINER.train, nprocs=ngpus_per_node, args=(ngpus_per_node, ARGS))
    elif ARGS.mode == 'test':
        TRAINER.test()