# helper functions
import torch
import torch.nn as nn
import torchmetrics
import time
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


def train_model(model: nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.device = 'mps',
               epoch: int = 1,
               writer: SummaryWriter = None):
    """Train the model

    Args:
        model (nn.Module): The model to train
        data_loader (torch._utils.data.data_loader): The data loader for the training data
        loss_fn (nn.Module): The loss function to calculate the loss
        optimizer (torch.optim.Optimizer): The optimizer to update the model's weights
        accuracy_fn (torchmetrics.Accuracy): The accuracy function to calculate the accuracy
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
        epoch (int, optional): The number of epochs to train the model. Defaults to 1.
        writter (SummaryWriter, optional): The tensorboard writter to log the training data. Defaults to None
    """
    
    model.train()
    running_loss, running_acc = 0.0, 0.0
    batch_result = []
    
    with tqdm(data_loader, desc=f"Epoch {epoch} | Training", leave=True) as t:
        for batch, data in enumerate(t):
            
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            
            # Calculate loss & accuracy
            loss = loss_fn(outputs, labels)
            acc = accuracy_fn(labels, preds)
            
            # Backward Pass + Optimize (Gradient Descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update data
            running_loss += loss.item()
            running_acc += acc.item()
            
            avg_loss = running_loss/(batch + 1)
            avg_acc = running_acc/(batch + 1)
            
            batch_result.append({
                "batch": batch + 1,
                "loss": avg_loss,
                "accuracy": avg_acc
            })
            
            # Log the loss and accuracy to TensorBoard
            if writer and (batch + 1) % 100 == 0:
                writer.add_scalar('Training Loss', running_loss, epoch * len(data_loader) + batch)
                writer.add_scalar('Training Accuracy', running_acc, epoch * len(data_loader) + batch)
            
            # Update tqdm description
            t.set_postfix(loss=f'{current_avg_loss:.4f}', acc=f'{current_avg_acc:.4f}')
    
    return {
        "name": model.__class__.__name__,
        "result": batch_result,
    }


def eval_model(model: nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              accuracy_fn: torchmetrics.Accuracy,
              device: torch.device = 'mps',
              epoch: int = 1):
    """eval the model

    Args:
        model (nn.Module): The model to test
        data_loader (torch.utils.data.DataLoader): The data loader for the testing data
        loss_fn (nn.Module): The loss function to calculate the loss
        accuracy_fn (torchmetrics.Accuracy): The accuracy function to calculate the accuracy
        device (torch.device, optional): The device used to train the model. Defaults to 'mps'
        epoch (int, optional): The number of epochs to train the model. Defaults to 1.
    """
    
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    batch_result = []
    
    with torch.inference_mode():
        with tqdm(data_loader, desc=f"Epoch {epoch} | Evaluating", leave=True) as t:
            for batch, data in enumerate(t):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                preds = outputs.softmax(dim=1).argmax(dim=1)
                
                # Calculate loss & accuracy
                loss = loss_fn(outputs, labels)
                acc = accuracy_fn(labels, preds)
                
                # Update data
                running_loss += loss
                running_acc += acc
                
                current_avg_loss = running_loss/(batch + 1)
                current_avg_acc = running_acc/(batch + 1)
                
                batch_result.append({
                    "batch": batch + 1,
                    "loss": current_avg_loss,
                    "accuracy": current_avg_acc
                })
                    
                # Update tqdm description
                t.set_postfix(loss=f'{current_avg_loss:.4f}', acc=f'{current_avg_acc:.4f}')
    
        return {
            "name": model.__class__.__name__,
            "result": batch_result,
        }
        

def save_model(model, model_name, directory="models", file_ext=".pth"):
    """
    Saves the PyTorch model's state dictionary to a specified directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        model_name (str): The name of the model file.
        directory (str): The directory where the model will be saved. Default is 'models'.
        file_ext (str): The file extension for the saved model. Default is '.pth'.
    
    Returns:
        Path: The path to the saved model file.
    """
    # Create a directory to save the model
    model_path = Path(directory)
    model_path.mkdir(parents=True, exist_ok=True)

    # Create the full model save path
    model_save_path = model_path / f"{model_name}{file_ext}"

    # Save the model
    torch.save(model.state_dict(), model_save_path)

    return model_save_path