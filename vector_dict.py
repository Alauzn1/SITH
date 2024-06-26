import torch.nn as nn
from models.model.transformer import Transformer_Model
from argparse import ArgumentParser
from datasets import *
from datasets.data import Multi30k
import pytorch_lightning as pl
import numpy as np
from utils import reduce_dimension
from utils_t import greedy_decode, gettgt
import torch
import pickle
from tqdm import tqdm
from argparse import Namespace
import yaml
from collections import OrderedDict

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_reduce_word_vec(wait_reduce, reduce_type, low_dim=3):
   
    print("Start integrating coordinates")
    all_vec_unique = wait_reduce
    all_vec_2_dimension = reduce_dimension.reduce_dimension(all_vec_unique, low_dim, reduce_type)
    print("Coordinate integration completed")
    return all_vec_unique, all_vec_2_dimension


def collect_vectors(ckpt_file, hparams_file, vectors_file):
    """
    Obtain dictionaries from high to low dimensions in the dataset
    :return:
    """
    print('Load model')

    model = Transformer_Model.load_from_checkpoint(ckpt_file)
    model.eval()
    print('Successfully loaded model')

    def namespace_constructor(loader, node):
        return Namespace(**loader.construct_mapping(node))

    yaml.add_constructor("tag:yaml.org,2002:python/object:argparse.Namespace", namespace_constructor)

    with open(hparams_file, 'r') as file:
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)

    print('Load Dataset')
    pl.seed_everything(0)
    # yaml_config = yaml.load(open(hparams_file, 'r'), Loader=yaml.FullLoader)
    ds = ds_dict[yaml_config['args'].dataset]()
    ds.prepare_data()
    ds.setup()

    hidden_high = None

    file_path_src = "/home/project/SITH/data/test_2016_flickr.en"
    file_path_tgt = "/home/project/SITH/data/test_2016_flickr.fr"

    with open(file_path_src, "r") as file1, open(file_path_tgt, 'r') as file2:
        for line1, line2 in zip(file1, file2):
            print(line1)
            src = str(line1.strip())
            tgt = str(line2.strip())
            embedding_pos_tgt = ds.get_tgt_embed(model, tgt, gettgt)
            embedding_pos, encoder_sixall, decoder_sixall = ds.transex(model, src, greedy_decode)
            #print(len(encoder_sixall))
            #print(len(decoder_sixall))
            encoder_1 = encoder_sixall[0]
            encoder_2 = encoder_sixall[1]
            encoder_3 = encoder_sixall[2]
            encoder_4 = encoder_sixall[3]
            encoder_5 = encoder_sixall[4]
            encoder_6 = encoder_sixall[5]
        
            decoder_1 = decoder_sixall[0]
            decoder_2 = decoder_sixall[1]
            decoder_3 = decoder_sixall[2]
            decoder_4 = decoder_sixall[3]
            decoder_5 = decoder_sixall[4]
            decoder_6 = decoder_sixall[5]
            
            # Remove the batch from the first dimension
            embedding_pos_seq = embedding_pos.squeeze(0).detach().numpy()
            encoder_1_seq = encoder_1.squeeze(0).detach().numpy()
            encoder_2_seq = encoder_2.squeeze(0).detach().numpy()
            encoder_3_seq = encoder_3.squeeze(0).detach().numpy()
            encoder_4_seq = encoder_4.squeeze(0).detach().numpy()
            encoder_5_seq = encoder_5.squeeze(0).detach().numpy()
            encoder_6_seq = encoder_6.squeeze(0).detach().numpy()

            decoder_1_seq = decoder_1.squeeze(0).detach().numpy()
            decoder_2_seq = decoder_2.squeeze(0).detach().numpy()
            decoder_3_seq = decoder_3.squeeze(0).detach().numpy()
            decoder_4_seq = decoder_4.squeeze(0).detach().numpy()
            decoder_5_seq = decoder_5.squeeze(0).detach().numpy()
            decoder_6_seq = decoder_6.squeeze(0).detach().numpy()
            embedding_pos_tgt_seq = embedding_pos_tgt.squeeze(0).detach().numpy()

            _ = np.concatenate((embedding_pos_seq, encoder_1_seq, encoder_2_seq, encoder_3_seq, encoder_4_seq, encoder_5_seq, encoder_6_seq
                                , decoder_1_seq, decoder_2_seq, decoder_3_seq, decoder_4_seq, decoder_5_seq, decoder_6_seq, embedding_pos_tgt_seq))

            if hidden_high is None:
                hidden_high = _
            else:
                hidden_high = np.vstack((hidden_high, _))
        print('Start saving-------------------')
        np.savez(vectors_file, hidden_high=hidden_high)


def generate_vector_dict(ckpt_file, hparams_file, vectors_file, reduce_file, reduce_type):

    collect_vectors(ckpt_file, hparams_file, vectors_file)
    all_vectors = np.load('{}.npz'.format(vectors_file))

    all_vectors_np = np.vstack((all_vectors['hidden_high']))

    embedding_high_uni, embedding_low = get_reduce_word_vec(all_vectors_np, reduce_type)
    print(embedding_high_uni.shape)
    print(embedding_low.shape)

    np.savez_compressed(reduce_file, high=embedding_high_uni, low=embedding_low)


