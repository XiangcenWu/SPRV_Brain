import argparse
from utils.config import MainConfig
import os
import logging
import torch
from Training.train_seg import train_seg_epoch
from Training.train_ranking import train_ranking_epoch
from utils.loader import get_loader_seg, get_loader_ranking
from utils.data_dir_list import get_data_list
from utils.file_loader import ReadH5d
from Networks.ranking import RankUNet
from utils.ranking_loss import custom_wa_difference_loss
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.losses import DiceLoss
from Training.workflow_evaliation import inference_all_data_accuracy, workflow_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Model training')

    
    parser.add_argument("-c", "--config", help="The path of config file.", type=str)

    return parser.parse_args()


def config_log(cfg):
    FORMAT = '%(asctime)s, %(message)s'
    logging.basicConfig(
        filename=os.path.join(cfg.result_dir, cfg.nick_name+'.txt'),
        level=logging.INFO,
        filemode='w',
        format=FORMAT
    )
    logging.getLogger().addHandler(logging.StreamHandler())


def make_result_dir(cfg):
    cfg.add_dict_item({'result_dir': os.path.join(cfg.result_base_dir, cfg.nick_name)})
    if not os.path.exists(cfg.result_base_dir):
        os.mkdir(cfg.result_base_dir)
    if not os.path.exists(cfg.result_dir):
        os.mkdir(cfg.result_dir)
        
loader_transform = ReadH5d()

def seg(cfg: MainConfig, sup_list: list):
    logging.info("Segmentation training begins")
    SegModel = SwinUNETR(cfg.img_size, cfg.segmentation.num_channel, 1)
    SegModel.to(cfg.device)
    seg_loader = get_loader_seg(
        sup_list, 
        loader_transform, 
        cfg.segmentation.batch_size, 
        cfg.segmentation.shuffle, 
        cfg.segmentation.drop_last
    )
    seg_loss_function = DiceLoss(sigmoid=True)
    seg_model_save_path = os.path.join\
        (cfg.result_base_dir, cfg.nick_name + '_seg.pt')
    seg_optimizer = torch.optim.AdamW(SegModel.parameters(), cfg.segmentation.learning_rate)
    for e in range(cfg.segmentation.epoch):
        train_loss = train_seg_epoch(
            seg_model=SegModel,
            seg_loader=seg_loader,
            seg_optimizer=seg_optimizer,
            seg_loss_function=seg_loss_function,
            device=cfg.device
        )
        if e % 1 == 0:
            logging.info(f"On epoch {e}, seg loss is {train_loss:.2f}")

    torch.save(SegModel.state_dict(), seg_model_save_path)
    logging.info(f'Seg model saved in <{seg_model_save_path}>')


def ranking(cfg: MainConfig, qry_list: list):
    seg_model_save_path = os.path.join\
        (cfg.result_base_dir, cfg.nick_name + '_seg.pt')
    ranking_model_save_path = os.path.join\
        (cfg.result_base_dir, cfg.nick_name + '_rank.pt')
    
    SegModel = SwinUNETR(cfg.img_size, cfg.segmentation.num_channel, 1)
    SegModel.load_state_dict(torch.load(seg_model_save_path, map_location=cfg.device))
    SegModel.to(cfg.device)
    ranking_loader = get_loader_ranking(
        data_dir_list=qry_list,
        reader=loader_transform,
        batch_size=cfg.ranking.batch_size,
        sequence_length=cfg.ranking.sequence_length,
        shuffle=cfg.ranking.shuffle,
        drop_last=cfg.ranking.drop_last
    )
    query_set_true_performance = inference_all_data_accuracy(qry_list, SegModel, 1, cfg.device).view(-1, 1)
    query_set_true_mean = query_set_true_performance.mean()
    # query_set_true_std = query_set_true_performance.std()


    #############################
    # seg_loss_function = DiceLoss(sigmoid=True)


    Seg_model = SwinUNETR(cfg.img_size, cfg.segmentation.num_channel, 1)
    Seg_model.load_state_dict(torch.load(seg_model_save_path,\
        map_location=cfg.device))
    Seg_model.to(cfg.device)

    Rank_model = RankUNet(cfg.img_size, cfg.ranking.num_channel,\
        cfg.ranking.token_length, encoder_drop = 0, transformer_drop=0)
    Rank_model.to(cfg.device)
    ##############################
    rank_optimizer = torch.optim.AdamW(Rank_model.parameters(), lr=cfg.ranking.learning_rate)

    for e in range(cfg.ranking.epoch):

        
        rank_loss = train_ranking_epoch(
            rank_model=Rank_model,
            rank_loader=ranking_loader,
            rank_optimizer=rank_optimizer,
            true_mean=query_set_true_mean,
            sel_loss_function=custom_wa_difference_loss,
            seg_model=Seg_model,
            device=cfg.device
        )
        
        if e % 100 == 0:
            logging.info(f"This is epoch {e}, ranking loss is {rank_loss:.4f}")


    torch.save(Rank_model.state_dict(), ranking_model_save_path)
    logging.info('Rank model saved!')



def main(args):
    assert args.config is not None, \
        'No configuration file specified, please set --config'
    cfg = MainConfig(args.config)
    make_result_dir(cfg)
    config_log(cfg)
    sup_set, qry_set, dev_set, est_set = get_data_list(
        data_dir=cfg.data.data_path,
        seed=25,
        support_set_number=cfg.data.support_set_number,
        query_set_number=cfg.data.query_set_number,
        development_set_number=cfg.data.development_set_number,
        estimation_set_number=cfg.data.estimation_set_number
    )
    seg(cfg, sup_list=sup_set)
    ranking(cfg, qry_set)
    workflow_evaluation(
        cfg.img_size,
        cfg.ranking.token_length,
        t_values=cfg.workflow_evaluation.t_values,
        development_set=dev_set,
        estimation_set=est_set,
        seg_channel=cfg.segmentation.num_channel,
        ranking_channel=cfg.ranking.num_channel,
        sequence_length=cfg.ranking.sequence_length,
        num_experimrnts=cfg.workflow_evaluation.num_experiments,
        pre_train=cfg.workflow_evaluation.pre_train,
        seg_net_dir=os.path.join(cfg.result_base_dir, cfg.nick_name + '_seg.pt'),
        ranking_net_dir=os.path.join(cfg.result_base_dir, cfg.nick_name + '_rank.pt'),
        device=cfg.device
        
    )
    
    



if __name__ == '__main__':
    args = parse_args()
    main(args)