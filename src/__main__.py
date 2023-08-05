import argparse

from .run import MMSA_run

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model', type=str, default='lf_dnn', help='Name of model',
                        choices=['lf_dnn', 'ef_lstm', 'tfn', 'lmf', 'mfn', 'graph_mfn', 'mult', 'bert_mag', 
                                 'misa', 'mfm', 'mlf_dnn', 'mtfn', 'mlmf', 'self_mm', 'mmim',
                                 'ifl_es', 'ifl_es1', 'ifl_es2', 'ifl_es3', 'ifl_es4', 'ifl_es5','ifl_es6','ifl_ls'])
    parser.add_argument('-d', '--dataset', type=str, default='sims',
                        choices=['sims', 'mosi', 'mosei'], help='Name of dataset')
    parser.add_argument('-c', '--config', type=str, default='',
                        help='Path to config file. If not specified, default config file will be used.')
    parser.add_argument('-t', '--tune', action='store_true',
                        help='Whether to tune hyper parameters. Default: False')
    parser.add_argument('-tt', '--tune-times', type=int, default=50,
                        help='Number of times to tune hyper parameters. Default: 50')
    parser.add_argument('-s', '--seeds', action='append', type=int, default=[],
                        help='Random seeds. Specify multiple times for multiple seeds. Default: [1111, 1112, 1113, 1114, 1115]')
    parser.add_argument('-n', '--num-workers', type=int, default=16,
                        help='Number of workers used to load data. Default: 4')
    parser.add_argument('-v', '--verbose', type=int, default=1,
                        help='Verbose level of stdout. 0 for error, 1 for info, 2 for debug. Default: 1')
    parser.add_argument('--model-save-dir', type=str, default='',
                        help='Path to save trained models. Default: "~/MMSA/saved_models"')
    parser.add_argument('--res-save-dir', type=str, default='',
                        help='Path to save csv results. Default: "~/MMSA/results"')
    parser.add_argument('--log-dir', type=str, default='',
                        help='Path to save log files. Default: "~/MMSA/logs"')
    parser.add_argument('-g', '--gpu-ids', action='append', default=[],
                        help='Specify which gpus to use. If an empty list is supplied, will automatically assign to the most memory-free gpu. \
                              Currently only support single gpu. Default: []')
    parser.add_argument('-Ft', '--feature-T', type=str, default='',
                        help='Path to custom text feature file. Default: ""')
    parser.add_argument('-Fa', '--feature-A', type=str, default='',
                        help='Path to custom audio feature file. Default: ""')
    parser.add_argument('-Fv', '--feature-V', type=str, default='',
                        help='Path to custom video feature file. Default: ""')
    # 超参
    parser.add_argument('-ld', '--lambda-dis', type=int, default=15)
    parser.add_argument('--gce', type=str, default='')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('-se', '--swap-epochs', type=int, default=5)
    parser.add_argument('--q', type=int, default=1)

    parser.add_argument('-lrb', '--learning-rate-bert', type=float, default=5e-5)
    parser.add_argument('-lra', '--learning-rate-audio', type=float, default=0.005)
    parser.add_argument('-lrv', '--learning-rate-video', type=float, default=0.0001)
    parser.add_argument('-lro', '--learning-rate-other', type=float, default=0.001)

    parser.add_argument('-ptd', '--post-text-dim', type=int, default=32)
    parser.add_argument('-pfd', '--post-fusion-dim', type=int, default=128)
    parser.add_argument('-ao', '--audio-out', type=int, default=16)
    parser.add_argument('-vo', '--video-out', type=int, default=32)

    parser.add_argument('-alhs', '--a-lstm-hidden-size', type=int, default=16)
    parser.add_argument('-vlhs', '--v-lstm-hidden-size', type=int, default=32)
    
    parser.add_argument('-wdb', '--weight-decay-bert', type=float, default=0.001)
    parser.add_argument('-wda', '--weight-decay-audio', type=float, default=0)
    parser.add_argument('-wdv', '--weight-decay-video', type=float, default=0)
    parser.add_argument('-wdo', '--weight-decay-other', type=float, default=0.01)
    return parser.parse_args()


if __name__ == '__main__':
    cmd_args = parse_args()
    MMSA_run(
        model_name=cmd_args.model,
        dataset_name=cmd_args.dataset,
        config_file=cmd_args.config,
        seeds=cmd_args.seeds,
        is_tune=cmd_args.tune,
        tune_times=cmd_args.tune_times,
        feature_T=cmd_args.feature_T,
        feature_A=cmd_args.feature_A,
        feature_V=cmd_args.feature_V,
        model_save_dir=cmd_args.model_save_dir,
        res_save_dir=cmd_args.res_save_dir,
        log_dir=cmd_args.log_dir,
        gpu_ids=cmd_args.gpu_ids,
        num_workers=cmd_args.num_workers,
        verbose_level=cmd_args.verbose,
        lambda_dis=cmd_args.lambda_dis,
        gce=cmd_args.gce,
        weight=cmd_args.weight,
        swap_epochs=cmd_args.swap_epochs,
        learning_rate_bert=cmd_args.learning_rate_bert,
        learning_rate_audio=cmd_args.learning_rate_audio,
        learning_rate_video=cmd_args.learning_rate_video,
        learning_rate_other=cmd_args.learning_rate_other,
        post_text_dim=cmd_args.post_text_dim,
        post_fusion_dim=cmd_args.post_fusion_dim,
        audio_out=cmd_args.audio_out,
        video_out=cmd_args.video_out,
        weight_decay_bert=cmd_args.weight_decay_bert,
        weight_decay_audio=cmd_args.weight_decay_audio,
        weight_decay_video=cmd_args.weight_decay_video,
        weight_decay_other=cmd_args.weight_decay_other,
        q=cmd_args.q,
        a_lstm_hidden_size=cmd_args.a_lstm_hidden_size,
        v_lstm_hidden_size=cmd_args.v_lstm_hidden_size,
    )
