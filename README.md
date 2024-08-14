# Initialization

## Environment setup
install all the environment based on the ```environment.yml``` file using conda or others

### for conda user can run these code.
```
conda env create -f environment.yml -p ./env
```

## Dataset Download
1. obtain your kaggle api from kaggle website.
2. open ```eda.ipynb``` run all commands and enter your kaggle's username and password.

# Preprocessing and Model Building
1. Clone the ```sample.ipynb``` with your name ```<name>.ipynb```.
2. Code review, the first section is data preprocessing, involves
    - Resizing
    - Normalization [0, 255] -> [0, 1]
    - Data Augmentation
        i. horizontal flip
        ii. rotation
3. test all functions is working, run all the entire file
4. if no error, remember search 'REMARK' and comment out the testing part which is reducing the dataloader for testing purpose
```
### REMARK: for testing purposes, reduce the dataset to 10% of the original size
train_dataset.dataset.samples = train_dataset.dataset.samples[:int(len(train_dataset)*0.1)]
test_dataset.dataset.samples = test_dataset.dataset.samples[:int(len(test_dataset)*0.1)]
valid_dataset.dataset.samples = valid_dataset.dataset.samples[:int(len(valid_dataset)*0.1)]

len(train_dataset), len(test_dataset), len(valid_dataset)
```
5. remember save every trained model and history, screenshot all the training and evaluation as well.
6. remember rename each trained model version to avoid overwritting.
7. run ```tensorboard --logdir=runs``` in terminal to view the accuracy and loss result.



