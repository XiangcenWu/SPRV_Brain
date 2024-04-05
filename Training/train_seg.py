import torch
from utils.dice_metric import dice_metric


def train_seg_epoch(
        seg_model,
        seg_loader,
        seg_optimizer,
        seg_loss_function,
        device,
    ):
    # remember the loader should drop the last batch to prevent differenct sequence number in the last batch
    seg_model.train()
    
    step = 0.
    loss_a = 0.
    for batch in seg_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        # forward pass and calculate the selection


        # forward pass of selected data
        output = seg_model(img)
        
        loss = seg_loss_function(output, label)
        loss.backward()
        seg_optimizer.step()
        seg_optimizer.zero_grad()

        loss_a += loss.item()
        step += 1.
    loss_of_this_epoch = loss_a / step

    return loss_of_this_epoch



def test_seg(seg_model, test_loader, device):


    seg_model.eval()
    performance_a = 0.
    step = 0.

    for batch in test_loader:

        img, label = batch["image"].to(device), batch["label"].to(device)


        with torch.no_grad():
            seg_output = seg_model(img)
            performance = dice_metric(seg_output, label, post=True, mean=True)
            performance_a += performance.item()
            step += 1.

    performance_of_this_epoch = performance_a / step

    return performance_of_this_epoch

