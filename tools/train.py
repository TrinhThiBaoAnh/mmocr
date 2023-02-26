# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmocr.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--work-dir', help='The dir to save logs and models')
    parser.add_argument(
        '--resume', action='store_true', help='Whether to resume checkpoint.')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='Enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='Whether to scale the learning rate automatically. It requires '
        '`auto_scale_lr` in config, and `base_batch_size` in `auto_scale_lr`')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='Job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    #############################################################################
    parser.add_argument('--lr', type=float,
        default=0.005,help='Change the learning rate.')
    parser.add_argument('--momentum', type=float,
        default=0.9,help='Change the momentum.')
    parser.add_argument('--weight-decay', type=float,
        default=0.0001,help='Change the weight decay.')
    parser.add_argument('--num-of-epoch', type=float,
        default=50,help='Change the number of epoch.')
    parser.add_argument('--fre-save-epoch', type=float,
        default=10,help='Change the fre_save_epoch.')
    # parser.add_argument('--phase', type=str, 
    #                     default='ocr', help='Choose ocr or det phase')
    parser.add_argument('--det_dataset_dir', type=str,
        default=None,help='Change the detection datadir')
    parser.add_argument('--ocr_dataset_dir', type=str,
        default=None,help='Change the ocr datadir')
    #############################################################################
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    # register all modules in mmdet into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    #############################################################################
    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
    # if args.momentum is not None:
    #     cfg.optim_wrapper.optimizer.momentum = args.momentum
    # if args.weight_decay is not None:
    #     cfg.optim_wrapper.optimizer.weight_decay = args.weight_decay
    if args.num_of_epoch is not None:
        cfg.train_cfg.max_epochs = args.num_of_epoch
    if args.fre_save_epoch is not None:
        cfg.train_cfg.val_interval = args.fre_save_epoch
    if args.det_dataset_dir is not None:
            cfg.toy_det_data_root = args.dataset_dir
            # cfg.toy_det_train = args.dataset_dir + '/Train'
            # cfg.toy_det_test = args.dataset_dir + '/Test'
    if args.ocr_dataset_dir is not None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>",args.ocr_dataset_dir)
            cfg.toy_data_root = args.ocr_dataset_dir
            cfg.toy_rec_train.data_root = args.ocr_dataset_dir 
            cfg.toy_rec_test.data_root = args.ocr_dataset_dir
            cfg.train_list[0].data_root = args.ocr_dataset_dir
            cfg.test_list[0].data_root = args.ocr_dataset_dir
            cfg.train_dataloader.dataset.datasets[0].data_root = args.ocr_dataset_dir
            cfg.val_dataloader.dataset.datasets[0].data_root = args.ocr_dataset_dir
            cfg.test_dataloader.dataset.datasets[0].data_root = args.ocr_dataset_dir
    #############################################################################
    # enable automatic-mixed-precision training
    if args.amp:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    if args.resume:
        cfg.resume = True

    # enable automatically scaling LR
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()
