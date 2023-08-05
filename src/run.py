import errno
import gc
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path
import argparse

# import wandb
# wandb.init(project="IFL4MSA", entity="lingfengking")

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from config import get_config_regression, get_config_tune
from data_loader import MMDataLoader
from models.IFL_ES import IFL_ES
from trains import Train
from utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2" # This is crucial for reproducibility

SUPPORTED_MODELS = [
    'LF_DNN', 'EF_LSTM', 'TFN', 'LMF', 'MFN', 'Graph_MFN', 'MFM',
    'MulT', 'MISA', 'BERT_MAG', 'MLF_DNN', 'MTFN', 'MLMF', 'Self_MM', 'MMIM'
]
SUPPORTED_DATASETS = ['MOSI', 'MOSEI', 'SIMS']

def _set_logger(log_dir, model_name, dataset_name, verbose_level):

    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger('IFL') 
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def _run(args, num_workers=4):
    # load data and models
    dataloader = MMDataLoader(args, num_workers, 'train')
    model = IFL_ES(args).to(args.device)

    logger.info(f'The model has {count_parameters(model)} trainable parameters')
    # TODO: use multiple gpus
    # if using_cuda and len(args.gpu_ids) > 1:
    #     model = torch.nn.DataParallel(model,
    #                                   device_ids=args.gpu_ids,
    #                                   output_device=args.gpu_ids[0])
    trainer = Train(args)
    # do train
    # epoch_results = trainer.do_train(model, dataloader)
    epoch_results = trainer.do_train(model, dataloader)

    # load trained model & do test
    assert Path(args.model_save_path).exists()
    model.load_state_dict(torch.load(args.model_save_path))
    model.to(args.device)

    dataloader = MMDataLoader(args, num_workers, 'test_iid')
    results = trainer.do_test(model, dataloader['test_iid'], mode="TEST")

    dataloader = MMDataLoader(args, num_workers, 'test_nn_text')
    results_text = trainer.do_test(model, dataloader['test_nn_text'], mode="TEST")

    dataloader = MMDataLoader(args, num_workers, 'test_nn_audio')
    results_audio = trainer.do_test(model, dataloader['test_nn_audio'], mode="TEST")

    dataloader = MMDataLoader(args, num_workers, 'test_nn_vision')
    results_video = trainer.do_test(model, dataloader['test_nn_vision'], mode="TEST")

    dataloader = MMDataLoader(args, num_workers, 'test_nn_tav')
    results_tav = trainer.do_test(model, dataloader['test_nn_tav'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return results, results_text, results_audio, results_video, results_tav

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', type=str, default='lf_dnn', help='Name of model',
                        choices=['ifl_es', 'ifl_es1', 'ifl_es2', 'ifl_ls'])
    parser.add_argument('-d', '--dataset', type=str, default='mosei',
                        choices=['sims', 'mosi', 'mosei'], help='Name of dataset')
    parser.add_argument('-s', '--seeds', action='append', type=int, default=[],
                        help='Random seeds. Specify multiple times for multiple seeds. Default: [1111, 1112, 1113, 1114, 1115]')
    parser.add_argument('-n', '--num_workers', type=int, default=16,
                        help='Number of workers used to load data. Default: 4')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model_save_dir', type=str, default='',
                        help='Path to save trained models. Default: "~/MMSA/saved_models"')
    parser.add_argument('--res_save_dir', type=str, default='',
                        help='Path to save csv results. Default: "~/MMSA/results"')
    parser.add_argument('--log_dir', type=str, default='',
                        help='Path to save log files. Default: "~/MMSA/logs"')
    parser.add_argument('-g', '--gpu_ids', action='append', default=[],
                        help='Specify which gpus to use. If an empty list is supplied, will automatically assign to the most memory-free gpu. \
                              Currently only support single gpu. Default: []')
    parser.add_argument('--KeyEval', type=str, default='')

    # self-mm
    parser.add_argument('--H', type=float, default='3.0')
    parser.add_argument('--excludeZero', type=bool, default=True)
    # commonParams
    parser.add_argument('--transformers', type=str, default='bert')
    parser.add_argument('--pretrained', type=str, default='/home/st/disk0/nijuntong/bert-base-uncased')
    parser.add_argument('--need_data_aligned', type=bool, default=False)
    parser.add_argument('--need_model_aligned', type=bool, default=False)
    parser.add_argument('--need_normalized', type=bool, default=False)
    parser.add_argument('--use_bert', type=bool, default=True)
    parser.add_argument('--use_finetune', type=bool, default=True)
    parser.add_argument('--early_stop', type=int, default=8)
    parser.add_argument('--update_epochs', type=int, default=2)

    # hyperparameters
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-lrb', '--learning_rate_bert', type=float, default=5e-5)
    parser.add_argument('-lra', '--learning_rate_audio', type=float, default=0.005)
    parser.add_argument('-lrv', '--learning_rate_video', type=float, default=0.0001)
    parser.add_argument('-lro', '--learning_rate_other', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-wdb', '--weight_decay_bert', type=float, default=0.001)
    parser.add_argument('-wda', '--weight_decay_audio', type=float, default=0)
    parser.add_argument('-wdv', '--weight_decay_video', type=float, default=0)
    parser.add_argument('-wdo', '--weight_decay_other', type=float, default=0.01)
    parser.add_argument('-alhs', '--a-lstm_hidden_size', type=int, default=16)
    parser.add_argument('-vlhs', '--v-lstm_hidden_size', type=int, default=32)
    parser.add_argument('-all', '--a_lstm_layers', type=int, default=1)
    parser.add_argument('-vll', '--v_lstm_layers', type=int, default=1)
    parser.add_argument('-to', '--text_out', type=int, default=768)
    parser.add_argument('-ao', '--audio_out', type=int, default=16)
    parser.add_argument('-vo', '--video_out', type=int, default=32)    
    parser.add_argument('-ald', '--a_lstm_dropout', type=float, default=0)
    parser.add_argument('-vld', '--v_lstm_dropout', type=float, default=0)
    # parser.add_argument('-tbd', '--t_bert_dropout', type=float, default=0.1)
    parser.add_argument('-pfd', '--post_fusion_dim', type=int, default=128)
    # parser.add_argument('-ptd', '--post_text_dim', type=int, default=32)
    # parser.add_argument('-pad', '--post_audio_dim', type=int, default=16)
    # parser.add_argument('-pvd', '--post_video_dim', type=int, default=32)
    parser.add_argument('-sd', '--swap_dim', type=int, default=32)
    parser.add_argument('-pfdt', '--post_fusion_dropout', type=float, default=0.1)
    parser.add_argument('-ptdt', '--post_text_dropout', type=float, default=0)
    parser.add_argument('-padt', '--post_audio_dropout', type=float, default=0)
    parser.add_argument('-pvdt', '--post_video_dropout', type=float, default=0)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('-ld', '--lambda_dis', type=int, default=15)
    parser.add_argument('-ls', '--lambda_swap', type=float, default=1)
    parser.add_argument('-se', '--swap_epochs', type=int, default=5)
    parser.add_argument('--gce', type=str, default='')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--trans_layers', type=int, default=2)
    parser.add_argument('--trans_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_results, model_results_text, model_results_audio, model_results_video, model_results_tav = [],[],[],[],[]
    
    # Initialization
    model_name = args.model.lower()
    dataset_name = args.dataset.lower()
    if dataset_name == 'mosei':
        seeds = args.seeds if args.seeds != [] else [1003, 1006, 1008]
    else:
        seeds = args.seeds if args.seeds != [] else [1003, 1006, 1008]

    # data settings
    if dataset_name == 'mosei':
        args.featurePath = '/home/st/disk0/ni/MMSA/MOSEI/data/'
        args.train_samples = 16326
        args.seq_lens = [50, 500, 375]
        args.feature_dims = [768, 74, 35]
        args.num_classes = 3
        if os.uname()[1] == 'aa3bc75198f8':
            args.pretrained = '/home/bert-base-uncased'
            args.featurePath = '/root/data/data/mosei/'
    elif dataset_name == 'mosi':
        args.featurePath = '/home/st/disk0/ni/MMSA/MOSI/data/'
        args.train_samples = 1284
        args.seq_lens = [50, 500, 375]
        args.feature_dims = [768, 5, 20]
        args.num_classes = 3
        if os.uname()[1] == 'aa3bc75198f8':
            args.pretrained = '/home/bert-base-uncased'
            args.featurePath = '/root/data/data/mosi/'
        # mosi 统一使用一个dropout
        args.a_lstm_dropout = args.dropout
        args.v_lstm_dropout = args.dropout
        # args.t_bert_dropout = args.dropout
        args.post_fusion_dropout = args.dropout
        args.post_text_dropout = args.dropout
        args.post_audio_dropout = args.dropout
        args.post_video_dropout = args.dropout

        args.weight_decay_bert = args.weight_decay
        args.weight_decay_audio = args.weight_decay
        args.weight_decay_video = args.weight_decay
        args.weight_decay_other = args.weight_decay

    logger = _set_logger(args.log_dir, model_name, dataset_name, args.verbose)

    for i, seed in enumerate(seeds):
        setup_seed(seed)

        logger.info("======================================== Program Start ========================================")
        
        # args = get_config_regression(model_name, dataset_name, config_file)
        args.model_save_path = Path(args.model_save_dir) / f'{model_name}-{dataset_name}.pth'
        args.device = assign_gpu(args.gpu_ids)
        args.train_mode = 'regression' # backward compatibility. TODO: remove all train_mode in code

        # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
        # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
        # It solves the bug of RNN always running on gpu 0.
        torch.cuda.set_device(args.device)

        logger.info("Running with args:")
        logger.info(args)
        logger.info(f"Seeds: {seeds}")
        
        res_save_dir = Path(args.res_save_dir) / "normal"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        
        args.cur_seed = i + 1
        logger.info(f"{'-'*30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-'*30}")

        # actual running
        results, results_text, results_audio, results_video, results_tav = _run(args, args.num_workers)

        logger.info(f"Result for seed {seed}: {results}")
        logger.info(f"Result for seed {seed}: {results_text}")
        logger.info(f"Result for seed {seed}: {results_audio}")
        logger.info(f"Result for seed {seed}: {results_video}")
        logger.info(f"Result for seed {seed}: {results_tav}")
        model_results.append(results)
        model_results_text.append(results_text)
        model_results_audio.append(results_audio)
        model_results_video.append(results_video)
        model_results_tav.append(results_tav)

    criterions = list(model_results[0].keys())
    # save result to csv
    csv_file = res_save_dir / f"{dataset_name}.csv"
    
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    
    text_results, audio_results, vision_results, tav_results = {}, {}, {}, {}
    # save results
    metrics = {}
    res = [model_name + 'IID   ']
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values)*100, 2)
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
        metrics[f'IID_{c}'] = mean
    df.loc[len(df)] = res
    res = [model_name + 'Text  ']
    for c in criterions:
        values = [r[c] for r in model_results_text]
        mean = round(np.mean(values)*100, 2)
        text_results[c] = mean
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
        metrics[f'Text_{c}'] = mean
    df.loc[len(df)] = res
    res = [model_name + 'Audio ']
    for c in criterions:
        values = [r[c] for r in model_results_audio]
        mean = round(np.mean(values)*100, 2)
        audio_results[c] = mean
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
        metrics[f'Audio_{c}'] = mean
    df.loc[len(df)] = res
    res = [model_name + 'Vision']
    for c in criterions:
        values = [r[c] for r in model_results_video]
        mean = round(np.mean(values)*100, 2)
        vision_results[c] = mean
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
        metrics[f'Vision_{c}'] = mean
    df.loc[len(df)] = res
    res = [model_name + 'TAV   ']
    for c in criterions:
        values = [r[c] for r in model_results_tav]
        mean = round(np.mean(values)*100, 2)
        tav_results[c] = mean
        std = round(np.std(values)*100, 2)
        res.append((mean, std))
        metrics[f'TAV_{c}'] = mean
    df.loc[len(df)] = res
    # wandb.log(metrics)

    res = [model_name + 'AVG   ']
    for c in criterions:
        value = (text_results[c] + audio_results[c] + vision_results[c] + tav_results[c]) / 4
        value = round(value, 2)
        res.append((value))
    df.loc[len(df)] = res

    df.to_csv(csv_file, index=None)
    logger.info(f"Results saved to {csv_file}.")