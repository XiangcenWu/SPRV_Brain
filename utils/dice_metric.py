import torch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric
import math


th = AsDiscrete(threshold=0.5)
def post_process(output):
    output = torch.sigmoid(output)
    return th(output)
dm = DiceMetric(reduction='mean', get_not_nans=True)
def dice_metric(y_pred, y_true, post=False, mean=False):
    """Calculate the dice score (accuracy) give the prediction from a trained segmentation network
    

    Args:
        y_pred (torch.tensor): output from the segmentation network 
        y_true (torch.tensor): ground truth of the output
        post (bool, optional): whether to do the post process (threshold, simoid etc...) 
                               before calculate the dice. Defaults to False.
        mean (bool, optional): calculate the mean of a batch of dice score. Defaults to False.

    Returns:
        torch.tensor: dice score of a single number or a batch of numbers
    """
    
    if post:
        y_pred = post_process(y_pred)
    # y_true = th(y_true)
        
    # if dm(y_pred, y_true).item() == float("nan"):
    # if math.isnan(dm(y_pred, y_true).item()):
    #     print(torch.unique(y_pred), torch.unique(y_true))
    
        

    dice_score = dm(y_pred, y_true).mean() if mean else dm(y_pred, y_true)
    
    return torch.where(torch.isnan(dice_score) , 0, dice_score)
