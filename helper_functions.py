# helper functions
import torch
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from datetime import datetime
from torchvision import transforms, datasets
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import time
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

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
               data_loader: DataLoader,
               loss_fn: nn.Module,
               optimizer: optim.Optimizer,
               accuracy_fn: Accuracy,
               device: torch.device = 'mps',
               epoch: int = 1,
               ) -> tuple[float, float]:
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
    total_loss, total_acc = 0.0, 0.0
    num_batches = len(data_loader)
    
    with tqdm(data_loader, desc=f"Epoch {epoch} | Training", unit="batch") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
    
            outputs = model(inputs)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            
            # Calculate loss & accuracy
            loss = loss_fn(outputs, labels)
            acc = accuracy_fn(preds, labels)
            
            # Backward Pass + Optimize (Gradient Descent)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update data
            total_loss += loss.item()
            total_acc += acc.item()
            
            # Update tqdm description
            current_batch = pbar.n + 1
            pbar.set_postfix(loss=f'{total_loss/current_batch:.4f}', acc=f'{total_acc/current_batch:.4f}')
    
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def eval_model(model: nn.Module,
              data_loader: DataLoader,
              loss_fn: nn.Module,
              accuracy_fn: Accuracy,
              device: torch.device = 'mps',
              epoch: int = 1
              ) -> tuple[float, float]:
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
    total_loss, total_acc = 0.0, 0.0
    num_batches = len(data_loader)
    
    with torch.inference_mode(), tqdm(data_loader, desc=f"Epoch {epoch} | Evaluating", unit="batch") as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
                        
            outputs = model(inputs)
            preds = outputs.softmax(dim=1).argmax(dim=1)
            
            # Calculate loss & accuracy
            loss = loss_fn(outputs, labels)
            acc = accuracy_fn(preds, labels)
            
            # Update data
            total_loss += loss.item()
            total_acc += acc.item()
            
            # Update tqdm description
            current_batch = pbar.n + 1
            pbar.set_postfix(loss=f'{total_loss/current_batch:.4f}', acc=f'{total_acc/current_batch:.4f}')
                
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    return avg_loss, avg_acc


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    accuracy_fn: Accuracy,
    device: torch.device,
    scheduler: optim.lr_scheduler = None,
    start_epoch: int = 1,
    end_epoch: int = 10,
    writer: SummaryWriter = None,
    name: str = "model",
) -> dict[str, list[dict[str, float]]]:
    training_results = []
    evaluation_results = []
    
    for epoch in tqdm(range(start_epoch, end_epoch + 1), desc="Epochs", unit="epoch"):
        train_loss, train_acc = train_model(
            model=model,
            data_loader=train_loader,
            loss_fn=criterion,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
            epoch=epoch,
        )
        
        eval_loss, eval_acc = eval_model(
            model=model,
            data_loader=test_loader,
            loss_fn=criterion,
            accuracy_fn=accuracy_fn,
            device=device,
            epoch=epoch
        )
        
        # Update the learning rate
        if scheduler:
            scheduler.step()
        
        training_results.append({"epoch": epoch, "loss": train_loss, "accuracy": train_acc})
        evaluation_results.append({"epoch": epoch, "loss": eval_loss, "accuracy": eval_acc})
        
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', eval_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', eval_acc, epoch)
        
        result = {
            "train": training_results,
            "eval": evaluation_results
        }
        
        # save checkpoint
        save_checkpoint(model=model, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        history=result, 
                        epoch=epoch,
                        model_name=name)
        
    return result
        

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


def save_history(data, history_name, directory="historys", file_ext=".csv"):
    """
    Saves the training history to a specified directory.

    Args:
        history (dict): The training history to be saved.
        directory (str): The directory where the history will be saved. Default is 'history'.
        file_ext (str): The file extension for the saved history. Default is '.json'.
    
    Returns:
    """
    # Create a directory to save the history
    history_path = Path(directory)
    history_path.mkdir(parents=True, exist_ok=True)

    # Create the full history save path
    history_save_path = history_path / f"{history_name}{file_ext}"

    # Save the history
    data.to_csv(history_save_path, index=False)


def save_checkpoint(model, optimizer, scheduler, history, epoch, model_name, directory="checkpoints", file_ext=".pth"):
    """
    Saves the PyTorch model's state dictionary and optimizer's state dictionary to a specified directory.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer to be saved.
        epoch (int): The epoch number.
        model_name (str): The name of the model file.
        directory (str): The directory where the model will be saved. Default is 'models'.
        file_ext (str): The file extension for the saved model. Default is '.pth'.
    
    Returns:
    """
    # Create a directory to save the model
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    model_path = Path(directory)
    model_path.mkdir(parents=True, exist_ok=True)

    # Create the full model save path
    model_save_path = model_path / f"{model_name}_{current_time}{file_ext}"

    # Save the model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
    }, model_save_path)


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
    dir_path (str): target directory

    Returns:
    A print out of:
      number of subdiretories in dir_path
      number of images (files) in each subdirectory
      name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})

def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time


# Plot loss curves of a model
def plot_loss_curves(results):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


# Pred and plot image function from notebook 04
# See creation: https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function
from typing import List
import torchvision


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: List[str] = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.

    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".
    
    Returns:
        Matplotlib plot of target image and model prediction as title.

    Example usage:
        pred_and_plot_image(model=model,
                            image="some_image.jpeg",
                            class_names=["class_1", "class_2", "class_3"],
                            transform=torchvision.transforms.ToTensor(),
                            device=device)
    """

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.0

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(
        target_image.squeeze().permute(1, 2, 0)
    )  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    
    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        with open(data_path / target_file, "wb") as f:
            request = requests.get(source)
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
            print(f"[INFO] Unzipping {target_file} data...") 
            zip_ref.extractall(image_path)

        # Remove .zip file
        if remove_source:
            os.remove(data_path / target_file)
    
    return image_path
