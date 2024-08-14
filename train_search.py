import os
import sys
import time
import torch
import logging
import dgl
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.search_space.network_f2 import Network
from util.logging import init_logger
from util.file_handler import deserialize
from models.component_prediction.prediction_dataset import PredictionDataset
from models.component_prediction.hyperparameter_configuration import hyperparameters
torch.set_printoptions(precision=4)
import pickle
import numpy as np
from util.utils import count_parameters_in_MB
from util.parser import get_args
from nas_trainer.supernet_trainer import PredictionTrainer
def get_dataset(data_filename, dataset_path, transform_to_emb_size, mode='train'):
    if os.path.exists(f'{dataset_path}_{mode}'):
        with open(f'{dataset_path}_{mode}', 'rb') as f:
            print(f'{dataset_path}_{mode}')
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
    # device = torch.device('cpu')
    # args.save = 'logs/search-{}'.format(args.save)
    # if not os.path.exists(args.save):
    #     os.mkdir(args.save)
    # log_filename = os.path.join(args.save, 'log.txt')
    # init_logger('', log_filename, logging.INFO, False)
    # print('*************log_filename=%s************' % log_filename)
    print(args)
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

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_path = os.path.join(args.save_path, args.model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_output_file = os.path.join(model_path, f'{args.catalog_name}_{args.embedding_size}_{args.exp_id}.dat')
    model_plot_file = os.path.join(model_path, f'{args.catalog_name}_{args.embedding_size}_{args.exp_id}.png')
    args.in_dim = embedding_width
    args.out_dim = vocabulary_size
    split_dataset = [get_dataset(data_filename, f'data/prepare_{args.catalog_name}', transform_to_emb_size, mode=mode) for
                     mode in ['train', 'val', 'test']]
    print('load_data')
    # for i in range(len(['train', 'val', 'test'])):
    #     split_dataset[i].graphs = [dgl.add_self_loop(graph) for graph in split_dataset[i].graphs]
    # print('add self loop')
    model = Network(args)

    model = model.to(device)
    logging.info("param size = %fMB", count_parameters_in_MB(model))

    trainer = PredictionTrainer(model, split_dataset[0], split_dataset[1], split_dataset[2], model_output_file,
                                args)
    trainer.train(device)
    logging.shutdown()



# def run_by_seed(args):
#     for i in range(args.num_sampled_archs):
#         print('searched {}-th for {}...'.format(i+1, args.data))
#         args.save = '{}-{}'.format(args.data, time.strftime("%Y%m%d-%H%M%S"))
#         seed = np.random.randint(0, 10000)
#         # seed = 0
#         args.seed = seed
#         genotype = main(args)

#     filename = './exp_res/%s-searched_res-%s-eps%s-reg%s.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'), args.epsilon, args.weight_decay)
#     fw = open(filename, 'w+')
#     fw.write('\n'.join(res))
#     fw.close()
#     print('searched res for {} saved in {}'.format(args.data, filename))


if __name__ == '__main__':

    args = get_args()
    main(args)


