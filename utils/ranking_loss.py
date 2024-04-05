import torch


def custom_wa_difference_loss(scores, true_accuracy, true_mean):
    
    # top_k_indices = torch.topk(scores, number_choice, dim=0).indices.flatten().tolist()

    tad = torch.abs(true_accuracy - true_mean)
    
    return (scores * tad).mean() 


