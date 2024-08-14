import os
import sys
import time
import torch
import logging
import dgl
import torch.utils
import psutil
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.search_space.network_f2 import SubNetwork
from models.component_prediction.gat_predictor import GatPredictor
from models.component_prediction.gcn_predictor import GcnPredictor
from util.logging import init_logger
from util.file_handler import deserialize
from models.component_prediction.prediction_dataset import PredictionDataset, FakeDataset
from models.component_prediction.hyperparameter_configuration import hyperparameters
torch.set_printoptions(precision=4)
import pickle
import numpy as np
from util.utils import count_parameters_in_MB
from util.parser import get_args
from nas_trainer.subnet_trainer import SubPredictionTrainer, SubFakePredictionTrainer

from nas_trainer.prediction_evaluator import PredictionEvaluator, PredictionCsEvaluator

def get_dataset(data_filename, dataset_path, transform_to_emb_size, mode='train'):
    if os.path.exists(f'{dataset_path}_{mode}'):
        with open(f'{dataset_path}_{mode}', 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = PredictionDataset(data_filename, mode, transform_to_emb_size)
        with open(f'{dataset_path}_{mode}', 'wb') as f:
            pickle.dump(dataset, f)
    return dataset

def get_test_dataset(data_filename, dataset_path, transform_to_emb_size, mode='train'):
    if os.path.exists(f'{dataset_path}_{mode}_test'):
        with open(f'{dataset_path}_{mode}_test', 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = PredictionDataset(data_filename, mode, transform_to_emb_size)
        dataset.set_len()
        dataset.graphs = dataset.graphs[:100]
        dataset.labels = dataset.labels[:100]
        with open(f'{dataset_path}_{mode}_test', 'wb') as f:
            pickle.dump(dataset, f)
    dataset.set_len()
    return dataset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dgl.seed(seed)
    np.random.seed(seed)


def main(args):
    global device
    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    # args.save = 'logs/search-{}'.format(args.save)
    # if not os.path.exists(args.save):
    #     os.mkdir(args.save)
    # log_filename = os.path.join(args.save, 'log.txt')
    # init_logger('', log_filename, logging.INFO, False)
    # print('*************log_filename=%s************' % log_filename)
    # device = torch.device('cuda:1')
    # device = torch.device('cpu')
    print(f'here is the args{args}')
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    set_seed(args.seed)
    logging.info("args = %s", args.__dict__)
    vocabulary_size = len(deserialize(f'{args.data_location}{args.catalog_name}_vocabulary.dat'))
    embedding_width = vocabulary_size if args.embedding_size == 'one_hot' else int(args.embedding_size)
    transform_to_emb_size = vocabulary_size if args.embedding_size == 'one_hot' else None
    data_filename = f'{args.data_location}{args.catalog_name}_component_{args.embedding_size}_dgl.hdf5'

    model_path = os.path.join(args.save_path, args.model_path)
    model_output_file = os.path.join(model_path, f'{args.catalog_name}_{args.embedding_size}_{args.exp_id}_sub.dat')
    model_plot_file = os.path.join(model_path, f'{args.catalog_name}_{args.embedding_size}_{args.exp_id}_sub.png')

    args.in_dim = embedding_width + 7
    args.out_dim = vocabulary_size + 7
    split_dataset = [get_test_dataset(data_filename, f'data/prepare_{args.catalog_name}', transform_to_emb_size, mode=mode) for
                     mode in ['train', 'val', 'test']]

    for i in range(len(['train', 'val', 'test'])):
        split_dataset[i].graphs = [dgl.add_self_loop(graph) for graph in split_dataset[i].graphs]




    fake_dataset = FakeDataset()

    print(fake_dataset.tgraphs)

    save_path = os.path.join(args.save_path, args.model_path, args.save_searched_file)



    with open(save_path, 'rb') as f:
        op_dict = pickle.load(f)
    op_dict = {'FF': ['concat', 'concat', 'sum', 'mean'], 'NA': ['gatv2', 'gat', 'gin'], 'RE': ['global_max'], 'SC': ['identity', 'zero', 'identity', 'identity', 'identity', 'identity',  'identity', 'identity', 'zero', 'identity']}
    #  'SC': ['identity', 'zero', 'identity', 'zero', 'zero', 'identity', 'identity', 'zero', 'zero', 'zero']}
    # op_dict = {'LA': 'l_lstm', 'NA': ['gat', 'gat'], 'RE': ['global_att', 'global_att', 'global_att']}
    # op_dict = {'LA': 'l_sum', 'NA': ['mlp', 'gin'], 'RE': ['set_transformer', 'set_transformer', 'set_transformer']}
    # op_dict = {'LA': 'l_lstm', 'NA': ['gatv2'], 'RE': ['none', 'global_mean']}
    print(f'search architecture {op_dict}')
    model = SubNetwork(args, op_dict)

    model = model.to(device)

    logging.info("param size = %fMB", count_parameters_in_MB(model))
    print(f"param size = {count_parameters_in_MB(model)}fMB")
    trainer = SubFakePredictionTrainer(model, split_dataset[0], split_dataset[1], split_dataset[2], fake_dataset, model_output_file, args)
    if not args.skip_training :
        trainer.train(device)



    evaluator = PredictionCsEvaluator(fake_dataset, trainer.model, model_output_file, args, device)
    evaluator.evaluate(0, k_values_to_test=(1, 2, 3, 5, 10, 15, 20), investigate_attention=False)


    logging.shutdown()



# def run_by_seed(args):
#     res = []
#     for i in range(args.num_sampled_archs):
#         print('searched {}-th for {}...'.format(i+1, args.data))
#         args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
#         seed = np.random.randint(0, 10000)
#         # seed = 0
#         args.seed = seed
#         genotype = main(args)
#         res.append('seed={}, genotype={}, saved_dir={}'.format(seed, genotype, args.save))
#     filename = './exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
#     fw = open(filename, 'w+')
#     fw.write('\n'.join(res))
#     fw.close()
#     print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':
    # if len(sys.argv) == 1:
    #     print('Please specify the required config info!!!')
    #     sys.exit(0)
    args = get_args()
    main(args)


