import logging
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = ['MMDataLoader']

logger = logging.getLogger('IFL')

class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATASET_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
        }
        DATASET_MAP[args.dataset]()

    def __init_mosi(self):
        # use deault feature file specified in config file
        with open(self.args.featurePath + f'{self.mode}.pkl', 'rb') as f:
            data = pickle.load(f)
        
        self.text = np.array(data['text_bert']).astype(np.float32)
        self.args.feature_dims[0] = 768
        self.audio = np.array(data['audio']).astype(np.float32)
        self.args.feature_dims[1] = self.audio.shape[2]
        self.vision = np.array(data['vision']).astype(np.float32)
        self.args.feature_dims[2] = self.vision.shape[2]
        self.raw_text = data['raw_text']
        self.ids = data['id']

        self.labels = {
            # 'M': data[self.mode][self.args['train_mode']+'_labels'].astype(np.float32)
            'M': np.array(data['regression_labels']).astype(np.float32)
        }

        logger.info(f"{self.mode} samples: {self.labels['M'].shape}")

        self.audio_lengths = data['audio_lengths']
        self.vision_lengths = data['vision_lengths']
        self.audio[self.audio == -np.inf] = 0
    
    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __len__(self):
        return len(self.labels['M'])

    def get_seq_len(self):
        if 'use_bert' in self.args and self.args.use_bert:
            return (self.text.shape[2], self.audio.shape[1], self.vision.shape[1])
        else:
            return (self.text.shape[1], self.audio.shape[1], self.vision.shape[1])

    def get_feature_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def __getitem__(self, index):
        sample = {
            'raw_text': self.raw_text[index],
            'text': torch.Tensor(self.text[index]), 
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()}
        } 
        if not self.args.need_data_aligned:
            sample['audio_lengths'] = self.audio_lengths[index]
            sample['vision_lengths'] = self.vision_lengths[index]

        return sample

def MMDataLoader(args, num_workers, mode):

    if mode == 'train':
        datasets = {'train': MMDataset(args, mode='train'),
                    'valid': MMDataset(args, mode='valid')}
    else:
        datasets = {mode: MMDataset(args, mode=mode)}
    # if 'seq_lens' in args:
    #     args.seq_lens = datasets['train'].get_seq_len() 

    dataLoader = {
        ds: DataLoader(datasets[ds],
                       batch_size=args.batch_size,
                       num_workers=num_workers,
                       pin_memory=True,
                       shuffle=True)
        for ds in datasets.keys()
    }
    
    return dataLoader
