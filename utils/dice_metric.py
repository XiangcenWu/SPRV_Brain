import torch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric



th = AsDiscrete(threshold=0.5)
def post_process(output):
    output = torch.sigmoid(output)
    return th(output)
dm = DiceMetric(reduction='mean')
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


    return dm(y_pred, y_true).mean() if mean else dm(y_pred, y_true)
