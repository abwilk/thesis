from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from scripts.data_loader import prep_data
from models.FCN import FCN

def train_CV(data, labels, k_folds, device, model_name, n_fams=20):
    """
    Trains k models using k-fold cross validation
    """
    models = []
    
    writer = SummaryWriter(f'runs/{model_name}')
    
    for fold, [train_index,val_index] in enumerate(k_folds):

        model = FCN(n_fams).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        loss_func = nn.CrossEntropyLoss()
        
        # prep data
        trainloader, valloader = prep_data(data,labels,train_index,val_index)
        # perform training
        model_trained = train(trainloader,valloader,fold+1,model,optimizer,loss_func,writer,model_name)
        models.append(model_trained)

    writer.close()

    return models
        
def train(trainloader, testloader, fold, model, optimizer, loss_func, writer, model_name):
    """
    Trains an independent fold in k-fold CV.
    """

    # Early stopping criteria
    patience = 7
    best_val_loss = float('inf')
    counter = 0 

    # Tracking performance across epochs
    train_losses = []
    val_losses = []
    
    num_epochs = 100
    for epoch in range(num_epochs):
        
        # Training Data
        model.train()
        train_loss = []
        for input, label in trainloader:
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss = np.mean(train_loss)
        train_losses.append(train_loss)

        # Val Data
        model.eval()
        val_loss = []
        with torch.no_grad():
            for input, label in testloader:
                output = model(input)
                loss = loss_func(output, label)
                val_loss.append(loss.item())
        val_loss = np.mean(val_loss)
        val_losses.append(val_loss)

        writer.add_scalars(
            f"Fold_{fold}",
            {
                "Train": train_loss,
                "Val":   val_loss
            },
            epoch
        )
    
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
        if counter >= patience:
            print(f"Early stopping after {epoch + 1} epochs.")
            break
    
    torch.save(model.state_dict(), f'models/saves/{model_name}/fold{fold}.pth')

    return model
        
