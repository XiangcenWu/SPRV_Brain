import torch
from utils.loader import get_loader_ranking, get_loader_seg
from utils.dice_metric import dice_metric
from utils.file_loader import ReadH5d
import random
from Networks.ranking import SelectionUNet
from monai.networks.nets.swin_unetr import SwinUNETR
##############################################
# select based on selection net

loader_transform = ReadH5d()
# # 把所有的数据的预测dice值都存到一个list里面 
# # def pred_all_data_accuracy(data_dir, sel_model, batch_size, sequence_length, device):
# #     sel_model.eval()
# #     loader = get_loader_ranking(data_dir, loader_transform, batch_size, sequence_length, True, True)
# #     list = []
# #     for batch in loader:
# #         img = batch['image'].to(device)
# #         with torch.no_grad():
# #             accuracy = sel_model(img)
# #             list.append(accuracy.flatten())
# #     all = torch.stack(list, dim=0).flatten()
# #     return all


def inference_all_data_accuracy(data_dir, seg_model, batch_size, sequence_length, device):
    seg_model.eval()
    loader = get_loader_ranking(data_dir, loader_transform, batch_size, sequence_length, True, True)
    list = []
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, True, False)
            
            list.append(accuracy.flatten())
    all = torch.stack(list, dim=0).flatten()
    return all


# # 把传进来的数据dirlist变成dicelist
# # def pred_list_data_accuracy(data_dir, sel_model, device):
# #     sel_model.eval()
# #     loader = create_data_loader(data_dir, len(data_dir), True, False)
# #     for batch in loader:
# #         img = batch['image'].to(device)
# #         with torch.no_grad():
# #             accuracy = sel_model(img)

# #     return accuracy.flatten()



def inference_list_data_accuracy(data_dir, seg_model, device):
    seg_model.eval()
    loader = get_loader_seg(data_dir, loader_transform, len(data_dir), True, False)
    for batch in loader:
        img, label = batch['image'].to(device), batch['label'].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, True, False)

    return accuracy.flatten()



# def selection_baseon_selection_net(all_data_dir, sel_model, sequence_length, num_of_sequence_to_select, device):
#     l = []
#     all_data_accuracy = pred_all_data_accuracy(all_data_dir, sel_model, 1, device)
#     for i in range(300):
#         data_dir = random.sample(all_data_dir, sequence_length)
#         accuracy = pred_list_data_accuracy(data_dir, sel_model, device)
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
# def calculate_mean_std(data_dir, seg_model, sequence_length, device):
#     loader = create_data_loader(data_dir, sequence_length, True, True)
#     l = []
#     for batch in loader:
#         img, label = batch['image'].to(device), batch['label'].to(device)
#         with torch.no_grad():
#             output = seg_model(img)
#             true_accuracy = dice_metric(output, label, post=True, mean=False)
#             l.append(true_accuracy)
#     result = torch.stack(l).flatten()
#     return torch.mean(result), torch.std(result)







def workflow_evaluation(
        t_values: list,
        development_set: list,
        estimation_set: list,
        pre_train: bool=True,
        seg_net_dir: str=None,
        ranking_net_dir: str=None,
        device: str='cuda:0'
    ):
    t = 0.9
    for i in range(5):
        random.shuffle(holdout_test_list)


        Sel_model = SelectionUNet((96, 96, 64), 4096, encoder_drop = 0, transformer_drop=0)
        
        Sel_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_sel_results/sel_model_only_prostate_8_v2_yphsd.ptm', map_location=device))
        Sel_model.to(device)
        Sel_model.eval()

        Seg_model = SwinUNETR((96, 96, 64), 1, 1)
        Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_yphsd.ptm', map_location=device))
        Seg_model.to(device)
        Seg_model.eval()

        parent_set = holdout_test_list[:60]
        cousin_set = [i for i in holdout_test_list if i not in parent_set]


        ########################################
        selected_val_set = []
        for i in range(300):
            sample = random.sample(parent_set, 8)
            loader = create_data_loader(sample, 8, shuffle=False)
            batch = next(iter(loader))
            img, label = batch['image'].to(device), batch['label'].to(device)
            with torch.no_grad():
                # output = Seg_model(img)
                # accuracy = dice_metric(output, label, post=True, mean=False)
                sel_output = Sel_model(torch.cat([img, label], dim=1))
                indexes_top = torch.topk(sel_output, 1, dim=0).indices.flatten().item()
                scores_top = torch.topk(sel_output, 1, dim=0).values.flatten().item()
                if scores_top > t:
                    # print('accept')
                
                    selected_data = [sample[indexes_top]]
                    selected_dice = inference_all_data_accuracy(selected_data, Seg_model, 1, device)
                    # print(indexes_top, scores_top, selected_dice)
                    
                    
                    if sample[indexes_top] not in selected_val_set:
                        selected_val_set.append(sample[indexes_top])
        print(len(selected_val_set))
        #############################################
        del Sel_model

        selected_train_list = [i for i in parent_set if i not in selected_val_set]
        print(inference_all_data_accuracy(selected_val_set, Seg_model, 1, device))
        print(inference_all_data_accuracy(parent_set, Seg_model, 1, device).mean().item())
        del Seg_model


        # # train from scratch
        Seg_model = SwinUNETR((96, 96, 96), 1, 1)
        Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_yphsd.ptm', map_location=device))
        Seg_model.to(device)
        selected_optimizer = torch.optim.Adam(Seg_model.parameters(), lr=1e-3)


        train_loader = create_data_loader(selected_train_list, 2, drop_last=True, shuffle=True)
        seg_loss_function = DiceLoss(sigmoid=True)
        for e in range(30):
            # print("This is epoch: ", e)
            train_loss = train_seg_net_h5(Seg_model, train_loader, selected_optimizer, seg_loss_function, device)
            # print(train_loss)



        selected_val_scores = inference_all_data_accuracy(selected_val_set, Seg_model, 1, device)
        # query_set_list_scores
        cousin_set_list_scores = inference_all_data_accuracy(cousin_set, Seg_model, 1, device)

        print("This is the result for selected")
        print('mean: ', selected_val_scores.mean().item(), cousin_set_list_scores.mean().item())
        print('std: ', selected_val_scores.std().item(), cousin_set_list_scores.std().item())
        ####random
        ###############################
        ###############################
        ###############################
        ###############################
        ###############################
        
        random_val_list = random.sample(parent_set, len(selected_val_set))
        random_train_list = [i for i in parent_set if i not in random_val_list]

        del Seg_model
        Seg_model = SwinUNETR((96, 96, 96), 1, 1)
        Seg_model.load_state_dict(torch.load('/home/xiangcen/xiaoyao/prostate_training/train_seg_results/seg_model_only_yphsd.ptm', map_location=device))
        Seg_model.to(device)
        selected_optimizer = torch.optim.Adam(Seg_model.parameters(), lr=1e-3)


        train_loader = create_data_loader(random_train_list, 2, drop_last=True, shuffle=True)
        seg_loss_function = DiceLoss(sigmoid=True)
        for e in range(30):
            # print("This is epoch: ", e)
            train_loss = train_seg_net_h5(Seg_model, train_loader, selected_optimizer, seg_loss_function, device)
            # print(train_loss)


        # random_val_list = [holdout_test_list[50:56][-2], holdout_test_list[0:6][-2]]  
        # random_train_list = [i for i in parent_set if i not in random_val_list]
        random_val_scores = inference_all_data_accuracy(random_val_list, Seg_model, 1, device)
        # query_set_list_scores
        cousin_set_list_scores = inference_all_data_accuracy(cousin_set, Seg_model, 1, device)

        print("This is the result for random sample")
        print('mean: ', random_val_scores.mean().item(), cousin_set_list_scores.mean().item())
        print('std: ', random_val_scores.std().item(), cousin_set_list_scores.std().item())



        ###############################
        ###############################
        ###############################
        ###############################
        ###############################