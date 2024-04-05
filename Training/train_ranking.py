import torch
from utils.dice_metric import dice_metric


def train_sel_net_h5(
        sel_model,
        sel_loader,
        sel_optimizer,
        true_mean,
        sel_loss_function,
        seg_model,
        device='cpu',
    ):
    seg_model.eval()
    sel_model.train()


    step_e = 0.
    loss_e = 0.
    for batch in sel_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)

        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, post=True, mean=False)

        sel_output = sel_model(torch.cat([img, label], dim=1))

        loss = sel_loss_function(sel_output, accuracy, 2, true_mean)

        loss.backward()
        sel_optimizer.step()
        sel_optimizer.zero_grad()

        loss_e += loss.item()
        step_e += 1

    return loss_e / step_e


  
def test_sel_net(
        sel_model, 
        sel_loader,
        true_mean,
        sel_loss_function,
        seg_model,
        device='cpu',
    ):
    seg_model.eval()
    sel_model.eval()
    
    
    step_e = 0.
    loss_e = 0.
    for batch in sel_loader:
        img, label = batch["image"].to(device), batch["label"].to(device)
        with torch.no_grad():
            output = seg_model(img)
            accuracy = dice_metric(output, label, post=True, mean=False)
            # print(accuracy)
            
            sel_output = sel_model(torch.cat([img, label], dim=1))
            
            print('test true accuracy', accuracy)
            
            print('test sel output', sel_output)
            
            loss = sel_loss_function(sel_output, accuracy, 2, true_mean)


        loss_e += loss.item()
        step_e += 1
    return loss_e / step_e

