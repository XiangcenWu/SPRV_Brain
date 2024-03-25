import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    # Common params
    parser.add_argument("--config", help="The path of config file.", type=str)
    parser.add_argument(
        '--device',
        help='Set the device place for training model.',
        default='gpu',
        choices=['cpu', 'gpu', 'xpu', 'npu', 'mlu'],
        type=str)
    parser.add_argument(
        '--save_dir',
        help='The directory for saving the model snapshot.',
        type=str,
        default='./output')
    parser.add_argument(
        '--num_workers',
        help='Number of workers for data loader. Bigger num_workers can speed up data processing.',
        type=int,
        default=0)
    parser.add_argument(
        '--do_eval',
        help='Whether to do evaluation in training.',
        action='store_true')
    parser.add_argument(
        '--use_vdl',
        help='Whether to record the data to VisualDL in training.',
        action='store_true')
    parser.add_argument(
        '--use_ema',
        help='Whether to ema the model in training.',
        action='store_true')

    # Runntime params
    parser.add_argument(
        '--resume_model',
        help='The path of the model to resume training.',
        type=str)
    parser.add_argument('--iters', help='Iterations in training.', type=int)
    parser.add_argument(
        '--batch_size', help='Mini batch size of one gpu or cpu. ', type=int)
    parser.add_argument('--learning_rate', help='Learning rate.', type=float)
    parser.add_argument(
        '--save_interval',
        help='How many iters to save a model snapshot once during training.',
        type=int,
        default=1000)
    parser.add_argument(
        '--log_iters',
        help='Display logging information at every `log_iters`.',
        default=10,
        type=int)
    parser.add_argument(
        '--keep_checkpoint_max',
        help='Maximum number of checkpoints to save.',
        type=int,
        default=5)

    # Other params
    parser.add_argument(
        '--seed',
        help='Set the random seed in training.',
        default=None,
        type=int)
    parser.add_argument(
        "--precision",
        default="fp32",
        type=str,
        choices=["fp32", "fp16"],
        help="Use AMP (Auto mixed precision) if precision='fp16'. If precision='fp32', the training is normal."
    )
    parser.add_argument(
        "--amp_level",
        default="O1",
        type=str,
        choices=["O1", "O2"],
        help="Auto mixed precision level. Accepted values are “O1” and “O2”: O1 represent mixed precision, the input \
                data type of each operator will be casted by white_list and black_list; O2 represent Pure fp16, all operators \
                parameters and input data will be casted to fp16, except operators in black_list, don’t support fp16 kernel \
                and batchnorm. Default is O1(amp).")
    parser.add_argument(
        '--profiler_options',
        type=str,
        help='The option of train profiler. If profiler_options is not None, the train ' \
            'profiler is enabled. Refer to the paddleseg/utils/train_profiler.py for details.'
    )
    parser.add_argument(
        '--data_format',
        help='Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".',
        type=str,
        default='NCHW')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1,
        help="Repeat the samples in the dataset for `repeats` times in each epoch."
    )
    parser.add_argument(
        '--opts', help='Update the key-value pairs of all options.', nargs='+')
    # Set multi-label mode
    parser.add_argument(
        '--use_multilabel',
        action='store_true',
        default=False,
        help='Whether to enable multilabel mode. Default: False.')
    parser.add_argument(
        '--to_static_training',
        action='store_true',
        default=None,
        help='Whether to enable to_static in training')

    return parser.parse_args()