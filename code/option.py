import argparse
import template

parser = argparse.ArgumentParser(description='CSRN')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=2,
                    help='number of threads for data loading')
parser.add_argument('--cpu', default=False,
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default='../../../dataset/train',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=100,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')
parser.add_argument('--scale', default=[4],
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=32,
                    help='input patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', type=bool, default=False,
                    help='enable memory-efficient forward')

# Model specifications
parser.add_argument('--model', default='CSR_FDN',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='./experiment/CSR_FDN/csr_fdnx4.pt',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=2000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', default=False,
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=100,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e4',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', default=False,
                    help='save output results')

# options for test
parser.add_argument('--testpath', type=str, default='../dataset/benchmark',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Set5',
                    help='dataset name for testing')

parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')
parser.add_argument('--no_augment', default=False,
                    help='do not use data augmentation')

args = parser.parse_args()
#template.set_template(args)
#args.scale = list(map(lambda x: int(x), args.scale.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

