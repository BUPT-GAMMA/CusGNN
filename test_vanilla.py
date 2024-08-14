import os
import torch
import logging
import dgl
import torch.utils
from models.component_prediction.gat_predictor import GatPredictor
from models.component_prediction.gcn_predictor import GcnPredictor
from nas_trainer.prediction_evaluator import PredictionEvaluator
from util.file_handler import deserialize
from models.word_to_vec.embedding_handler import EmbeddingHandler
from models.component_prediction.prediction_dataset import PredictionDataset
from models.component_prediction.hyperparameter_configuration import hyperparameters
from models.component_prediction.prediction_trainer import PredictionTrainer
from models.component_prediction.vanilla_evaluator import PredictionEvaluator
from models.word_to_vec.embedding import Embedding
torch.set_printoptions(precision=4)
import pickle
import numpy as np
from util.utils import count_parameters_in_MB
from util.parser import get_args
import torch.backends.cudnn as cudnn

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
    # torch.cuda.manual_seed(seed)
    dgl.seed(seed)
    np.random.seed(seed)


def main(args):
    global device
    if args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    # device = torch.device('cuda:1')
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    set_seed(args.seed)
    logging.info("args = %s", args.__dict__)
    vocabulary = deserialize(f'{args.data_location}{args.catalog_name}_vocabulary.dat')
    vocabulary_size = len(vocabulary)
    embedding_width = vocabulary_size if args.embedding_size == 'one_hot' else int(args.embedding_size)
    transform_to_emb_size = vocabulary_size if args.embedding_size == 'one_hot' else None
    embedding_name = f'embedding_{args.embedding_size}' if str(args.embedding_size).isdigit() \
        else f'{args.embedding_size}_embedding'
    embedding_file = f'{args.data_location}{args.catalog_name}_component_{embedding_name}.dat'
    embeddings = EmbeddingHandler().load_embedding(embedding_file)
    id2target = embeddings.get_index_to_word()
    embedded_vectors = embeddings.get_embedded_vectors()
    temp = [embedded_vectors[id2target[i]].cpu().tolist()  for i in range(vocabulary_size)]
    del embeddings
    del embedded_vectors
    del id2target
    torch.cuda.empty_cache()
    embedded_vectors = torch.tensor(temp)
    data_filename = f'{args.data_location}{args.catalog_name}_component_one_hot_dgl.hdf5'
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    model_path = os.path.join(args.save_path, args.model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    #model_path = os.path.join(args.save_path, args.model_path)
    
    model_output_file = os.path.join(model_path, f'{args.catalog_name}_{args.embedding_size}.dat')
    model_plot_file = os.path.join(model_path, f'{args.catalog_name}_{args.embedding_size}.png')

    args.in_dim = embedding_width
    args.out_dim = vocabulary_size
    split_dataset = [get_dataset(data_filename, f'data/prepare_{args.catalog_name}', transform_to_emb_size, mode=mode) for
                     mode in ['train', 'val', 'test']]

    save_path = os.path.join(args.save_path, args.model_path, args.save_searched_file)


    model_type = args.model_type
    if embedding_width in [20, 100]:
        hyper = hyperparameters['catA'][embedding_width][model_type]
    else:
        hyper = hyperparameters['catA']['one_hot'][model_type]
    if model_type == 'gat':
        model = GatPredictor(node_feature_dim=embedding_width, hidden_dim=hyper['hidden_size'], num_heads=hyper['num_heads'],
                       num_classes=vocabulary_size, num_hidden_layers=hyper['n_hidden_layers'],
                       feat_drop=hyper['dropout_rate'], attn_drop=0.0, residual=False, embeddings=embedded_vectors)
    else:
        model = GcnPredictor(node_feature_dim=embedding_width, hidden_dim=hyper['hidden_size'],
                             num_classes=vocabulary_size, num_hidden_layers=hyper['n_hidden_layers'],
                             dropout_rate=hyper['dropout_rate'],embeddings=embedded_vectors)
    model = model.to(device)

    logging.info("param size = %fMB", count_parameters_in_MB(model))
    print(f"param size = {count_parameters_in_MB(model)}fMB")
    # p = psutil.Process()
    # cpu_lst = p.cpu_affinity()
    # print("cpu list", cpu_lst)
    # p.cpu_affinity(cpu_lst[-8:])
    trainer = PredictionTrainer(model, split_dataset[0], split_dataset[1], split_dataset[2], model_output_file,
                                args)
    if not args.skip_training:
        trainer.train(device)
    evaluator = PredictionEvaluator(split_dataset[2], trainer.model, model_output_file, args, device)
    evaluation = evaluator.evaluate(k_values_to_test=(1, 2, 3, 5, 10, 15, 20), investigate_attention=False)

    print('====================')
    print('here is the test retrieval in test')
    print(evaluation)
    print('====================')
    logging.shutdown()


if __name__ == '__main__':
    args = get_args()
    main(args)



