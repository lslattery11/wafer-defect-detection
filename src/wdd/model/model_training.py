import torch
import wandb
import torchmetrics
import time

def train_model(
    model,
    optimizer ,
    epochs: int ,
    batch_step_size: int = 1000,
    log: bool = False,
    ):
    assert(isinstance(model,torch.nn.Module)),"net must be an instance of torch.nn.Module"
    assert(isinstance(optimizer,torch.optim.Optimizer)),"optimizer must be an instance of torch.optim.Optimizer"

    device=next(model.parameters()).device

    for epoch in range(epochs):
        start=time.time()
        print('EPOCH {}:'.format(epoch + 1))
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model,optimizer,batch_step_size)

        # Stop trainining mode.
        model.train(False)

        #should change to be cleaean up.
        running_vloss = 0.0
        for i, vdata in enumerate(model.validLoader):
            vinputs, vlabels = vdata
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            voutputs = model(vinputs)
            vloss = model.validLossfn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)

        y_trues,y_preds,_ = model.predict(model.validLoader)
        balanced_f1=torchmetrics.F1Score(num_classes=9,average='macro')(y_trues,y_preds)


        if log==True:
            wandb.log({
                'training_loss' : avg_loss,
                'validation_loss': avg_vloss,
                'balanced_f1': balanced_f1,
            })
        stop=time.time()
        print('LOSS train {} valid {} ... took {} minutes'.format(avg_loss, avg_vloss,(stop-start)/60))

    return

def train_one_epoch(
    model,
    optimizer,
    batch_step_size,
    ):

    device=next(model.parameters()).device

    running_loss = 0.
    last_loss = 0.
    
    optimizer.zero_grad()
    for i, data in enumerate(model.trainingLoader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #optimizer.zero_grad()

        # Make predictions for this batch
        model.eval()
        outputs = model(inputs)
        model.train(True)
        # Compute the loss and its gradients
        loss = model.trainingLossfn(outputs, labels)
        loss.backward()
        
        # Gather data and report
        running_loss += loss.item()
        # Adjust learning weights
        if ((i+1)%batch_step_size==0) or (i+1==len(model.trainingLoader)):
            optimizer.step()
            optimizer.zero_grad()
            #print("running loss: ",running_loss/(i+1))
            
        # Gather data and report
        running_loss += loss.item()


    last_loss=running_loss/(i+1)

    return last_loss
