
import argparse
import time
import random
import numpy as np

import torch
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import rlctr.trainer as trainer
import rlctr.utils.tensor_utils as utils

def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser)
    args = parser.parse_args()

    return args


def register_default_args(parser):
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive', 'random', 'bayes', 'finetuning'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--save_epoch', type=int, default=2)
    parser.add_argument('--max_save_num', type=int, default=5)

    # controller
    parser.add_argument('--time_budget', type=float, default=5.0,help='time budget(h) for training controller.')
    parser.add_argument('--layers_of_child_model', type=int, default=3)
    parser.add_argument('--shared_initial_step', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    parser.add_argument('--shared_rnn_max_length', type=int, default=35)
    parser.add_argument('--load_path', type=str, default='')
    
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--search_space', type=str, default='CAD')
    parser.add_argument('--format', type=str, default='two')
    # parser.add_argument('--format', type=str, default='two')
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--gnn_hidden', type=int, default=64)

    parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--controller_max_step', type=int, default=10,
                        help='step for controller parameters')
    parser.add_argument('--controller_optim', type=str, default='adam')
    parser.add_argument('--controller_lr', type=float, default=3.5e-4,
                        help="will be ignored if --controller_lr_cosine=True")
    parser.add_argument('--controller_grad_clip', type=float, default=0)
    parser.add_argument('--tanh_c', type=float, default=2.5)
    parser.add_argument('--softmax_temperature', type=float, default=5.0)
    parser.add_argument('--derive_num_sample', type=int, default=10)
    parser.add_argument('--hyper_eval_inters', type=int, default=30)
    parser.add_argument('--derive_finally', type=bool, default=True)
    parser.add_argument('--derive_from_history', type=bool, default=True)
    parser.add_argument('--controller_hid', type=int, default=100)
    # child model
    parser.add_argument("--early_stop_epoch", type=int, default=50,
                        help="epoch that valid loss bot decrease.")
    parser.add_argument("--only_one_act_funtion", type=bool, default=False,
                        help="epoch that valid loss bot decrease.")
    parser.add_argument("--shared_params", action='store_true', default=False,
                        help="shared_params between child model.")
    parser.add_argument("--cos_lr", action='store_true', default=False,
                        help="use cos lr in training stage")
    parser.add_argument("--ln", action='store_true', default=False,
                        help="layer norm")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu number")
    parser.add_argument("--without_jk", type=bool, default=False,
                        help="without_jk: remove jk in snag.")
    parser.add_argument("--dataset", type=str, default="catA", required=False,
                        help="The input dataset.")
    parser.add_argument("--epochs", type=int, default=800,
                        help="number of training epochs")
    parser.add_argument("--train_epochs", type=int, default=800,
                        help="number of controller training epoch.")
    parser.add_argument("--gnn_epochs", type=int, default=50,
                        help="number of training gnn epochs")
    parser.add_argument("--retrain_epochs", type=int, default=800,
                        help="number of training epochs")
    parser.add_argument("--multi_label", type=bool, default=False,
                        help="multi_label or single_label task")
    parser.add_argument("--residual", action="store_false",
                        help="use residual connection")
    parser.add_argument("--in_drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument("--param_file", type=str, default="cora_test.pkl",
                        help="learning rate")
    parser.add_argument("--optim_file", type=str, default="opt_cora_test.pkl",
                        help="optimizer save path")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--max_param', type=float, default=5E6)
    parser.add_argument('--supervised', type=bool, default=False)
    parser.add_argument('--submanager_log_file', type=str, default=f"sub_manager_logger_file_{time.time()}.txt")
    
    parser.add_argument("--show_gnn_training_info", action='store_true', default=True,
                        help="show gnn training details")
    parser.add_argument('--gnn_hidden_dim', type=int, default=64)

    parser.add_argument('--task', type=str, default='rating_pred')
    parser.add_argument('--out_path', type=str, default='output/')
    
    parser.add_argument('--out_file', type=str, default='default.csv')
    
    parser.add_argument('--finetune_num', type=int, default=10)
    parser.add_argument("--best_test_from_val", action='store_true', default=True,
                        help="best searched test performance according to val results")
    parser.add_argument("--load_ckpt", action='store_true', default=False)
    parser.add_argument('--ckpt_path', type=str)

    parser.add_argument('--catalog_name', type=str, default='catA', help='catalog name')
    parser.add_argument('--data_location', type=str, default='./data/', help='data path')
    parser.add_argument('--embedding_size', type=str, default='one_hot', help='embedding type')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='eval batch size')
    parser.add_argument('--eval_batch_size', type=int, default=1024, help='eval batch size')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout for model')
    parser.add_argument('--hidden_size', type=int, default=2048, help='default hidden_size in supernet')
    parser.add_argument('--embedding_keyword', type=str, default='embedding', help='data path')
    parser.add_argument('--num_layers', type=int, default=3, help='num of layers of GNN method.')

def main(args):  # pylint:disable=redefined-outer-name

    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False

    torch.cuda.set_device(args.gpu)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)

    trnr = trainer.Trainer(args)

    if args.mode =='train':
        print(args)
        trnr.train()
    elif args.mode in ['random', 'bayes']:
        print(args)
        trnr.random_bayes_search(mode=args.mode, max_evals=args.train_epochs * args.controller_max_step)
    elif args.mode == 'derive':
        trnr.derive()

    else:
        raise Exception(f"[!] Mode not found: {args.mode}")

    trnr.save_training_dict()


if __name__ == "__main__":
    args = build_args()
    main(args)



