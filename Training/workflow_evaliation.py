import torch
from utils.loader import get_loader_ranking, get_loader_seg
from utils.dice_metric import dice_metric
from utils.file_loader import ReadH5d
import random
from Networks.ranking import SelectionUNet
from monai.networks.nets.swin_unetr import SwinUNETR
import logging
from monai.losses import DiceLoss
from Training.train_seg import train_seg_epoch
##############################################
# select based on selection net

loader_transform = ReadH5d()
# # 把所有的数据的预测dice值都存到一个list里面 
# # def pred_all_data_accuracy(data_dir, RankingModel, batch_size, sequence_length, device):
# #     RankingModel.eval()
# #     loader = get_loader_ranking(data_dir, loader_transform, batch_size, sequence_length, True, True)
# #     list = []
# #     for batch in loader:
# #         img = batch['image'].to(device)
# #         with torch.no_grad():
# #             accuracy = RankingModel(img)
# #             list.append(accuracy.flatten())
# #     all = torch.stack(list, dim=0).flatten()
# #     return all


def inference_all_data_accuracy(data_dir, SegModel, batch_size, device):
    SegModel.eval()
    loader = get_loader_seg(data_dir, loader_transform, batch_size, True, True)
    list = []
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = SegModel(img)
            accuracy = dice_metric(output, label, True, False)
            
            list.append(accuracy.flatten())
    all = torch.stack(list, dim=0).flatten()
    return all


# # 把传进来的数据dirlist变成dicelist
# # def pred_list_data_accuracy(data_dir, RankingModel, device):
# #     RankingModel.eval()
# #     loader = create_data_loader(data_dir, len(data_dir), True, False)
# #     for batch in loader:
# #         img = batch['image'].to(device)
# #         with torch.no_grad():
# #             accuracy = RankingModel(img)

# #     return accuracy.flatten()



def inference_list_data_accuracy(data_dir, SegModel, device):
    SegModel.eval()
    loader = get_loader_seg(data_dir, loader_transform, len(data_dir), True, False)
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = SegModel(img)
            accuracy = dice_metric(output, label, True, False)

    return accuracy.flatten()



# def selection_baseon_selection_net(all_data_dir, RankingModel, sequence_length, num_of_sequence_to_select, device):
#     l = []
#     all_data_accuracy = pred_all_data_accuracy(all_data_dir, RankingModel, 1, device)
#     for i in range(300):
#         data_dir = random.sample(all_data_dir, sequence_length)
#         accuracy = pred_list_data_accuracy(data_dir, RankingModel, device)
#         std = torch.abs(torch.std(accuracy) - torch.std(all_data_accuracy))
#         mean = torch.abs(torch.mean(accuracy) - torch.mean(all_data_accuracy))
#         l.append({'data_dir':data_dir, 'std':std.item(), 'mean': mean.item()})
#     newlist = sorted(l, key = lambda d: d['mean'])
#     top_mean = newlist[:20]
#     newlist = sorted(top_mean, key = lambda d: d['std'])
#     elements = newlist[:num_of_sequence_to_select]
#     newlist =[]
#     for item in elements:
#         newlist.append(item['data_dir'])
#     return [x  for sublist in newlist for x in sublist]
# ######################################################



# ########################################
# # validate the selection by segmentation net
# def calculate_mean_std(data_dir, SegModel, sequence_length, device):
#     loader = create_data_loader(data_dir, sequence_length, True, True)
#     l = []
#     for batch in loader:
#         img, label = batch['image'].to(device), batch['label'].to(device)
#         with torch.no_grad():
#             output = SegModel(img)
#             true_accuracy = dice_metric(output, label, post=True, mean=False)
#             l.append(true_accuracy)
#     result = torch.stack(l).flatten()
#     return torch.mean(result), torch.std(result)

def select(development_set, t_value, sequnece_length, RankingModel, device):
    selected_val_set = []
    for i in range(300):
        sample = random.sample(development_set, sequnece_length)
        loader = get_loader_seg(sample, loader_transform, sequnece_length, shuffle=False, drop_last=False)
        batch = next(iter(loader))
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():

            sel_output = RankingModel(torch.cat([img, label], dim=1))
            indexes_top = torch.topk(sel_output, 1, dim=0).indices.flatten().item()
            scores_top = torch.topk(sel_output, 1, dim=0).values.flatten().item()
            if scores_top > t_value:
                # print('accept')
            
                selected_data = [sample[indexes_top]]
                
                
                if sample[indexes_top] not in selected_val_set:
                    selected_val_set.append(sample[indexes_top])
                    
    return selected_val_set





def workflow_evaluation(
        img_size,
        token_length,
        t_values: list,
        development_set: list,
        estimation_set: list,
        seg_channel:int = 4,
        ranking_channel:int = 5,
        sequence_length:int=8,
        num_experimrnts: int=5,
        pre_train: bool=True,
        seg_net_dir: str=None,
        ranking_net_dir: str=None,
        device: str='cuda:0'
    ):
    for t in t_values:
        for i in range(num_experimrnts):
            random.shuffle(development_set)


            RankingModel = SelectionUNet(img_size, ranking_channel, token_length, encoder_drop = 0, transformer_drop=0)
            
            RankingModel.load_state_dict(torch.load(ranking_net_dir, map_location=device))
            RankingModel.to(device)
            RankingModel.eval()

            SegModel = SwinUNETR(img_size, 1, 1)
            
            SegModel.load_state_dict(torch.load(seg_net_dir, map_location=device))
            SegModel.to(device)
            SegModel.eval()
            selected_val_set = select(
                development_set=development_set,
                t_value=t,
                sequnece_length=sequence_length,
                RankingModel=RankingModel,
                device=device
            )
            #############################################
            del RankingModel

            selected_train_list = [i for i in development_set if i not in selected_val_set]
            logging.info("This result will not be on the paper")
            logging.info(f"t value: {t}, this is experiment {i}, inference on seg model")
            logging.info(f"selected_val_set accuracy: \
                {inference_all_data_accuracy(selected_val_set, SegModel, 1, device).mean().item():2f}")
            logging.info(f"True accuracy \
                {inference_all_data_accuracy(development_set, SegModel, 1, device).mean().item():2f}")
            del SegModel


            
            SegModel = SwinUNETR(img_size, seg_channel, 1)
            if pre_train:
                SegModel.load_state_dict(torch.load(seg_net_dir, map_location=device))
            SegModel.to(device)
            seg_optimizer = torch.optim.Adam(SegModel.parameters(), lr=1e-3)


            seg_loader = get_loader_seg(selected_train_list, loader_transform, 2, drop_last=True, shuffle=True)
            seg_loss_function = DiceLoss(sigmoid=True)
            for e in range(30):
                # print("This is epoch: ", e)
                train_loss = train_seg_epoch(
                    seg_model=SegModel,
                    seg_loader=seg_loader,
                    seg_optimizer=seg_optimizer,
                    seg_loss_function=seg_loss_function,
                    device=device
                )
                # print(train_loss)



            selected_val_scores = inference_all_data_accuracy(selected_val_set, SegModel, 1, device)
            estimation_set_list_scores = inference_all_data_accuracy(estimation_set, SegModel, 1, device)

            logging.info("This is the result for selected")
            logging.info(f'mean of select: {selected_val_scores.mean().item()}, mean of estimation set {estimation_set_list_scores.mean().item()}')
            logging.info(f'std of select: {selected_val_scores.std().item()}, std of estimation set {estimation_set_list_scores.std().item()}')
            ####random
            ###############################
            ###############################
            ###############################
            ###############################
            ###############################
            
            random_val_list = random.sample(development_set, len(selected_val_set))
            random_train_list = [i for i in development_set if i not in random_val_list]

            del SegModel
            SegModel = SwinUNETR(img_size, 1, 1)
            if pre_train:
                SegModel.load_state_dict(torch.load(seg_net_dir, map_location=device))
            SegModel.to(device)
            selected_optimizer = torch.optim.Adam(SegModel.parameters(), lr=1e-3)


            seg_loader = get_loader_seg(selected_train_list, loader_transform, 2, drop_last=True, shuffle=True)
            seg_loss_function = DiceLoss(sigmoid=True)
            for e in range(30):
                # print("This is epoch: ", e)
                # print(train_loss)
                train_loss = train_seg_epoch(
                    seg_model=SegModel,
                    seg_loader=seg_loader,
                    seg_optimizer=seg_optimizer,
                    seg_loss_function=seg_loss_function,
                    device=device
                )


            # random_val_list = [development_set[50:56][-2], development_set[0:6][-2]]  
            # random_train_list = [i for i in development_set if i not in random_val_list]
            random_val_scores = inference_all_data_accuracy(random_val_list, SegModel, 1, device)
            # query_set_list_scores
            estimation_set_list_scores = inference_all_data_accuracy(estimation_set, SegModel, 1, device)

            # print("This is the result for random sample")
            # print('mean: ', random_val_scores.mean().item(), estimation_set_list_scores.mean().item())
            # print('std: ', random_val_scores.std().item(), estimation_set_list_scores.std().item())

            logging.info("This is the result for random")
            logging.info(f'mean of random: {random_val_scores.mean().item()}, mean of estimation set {estimation_set_list_scores.mean().item()}')
            logging.info(f'std of random: {random_val_scores.std().item()}, std of estimation set {estimation_set_list_scores.std().item()}')
            

