import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from model import build_model, modify_model, resume_model, load_chackpoint
from datasets import create_listdataset, split_datasets, create_datasets, create_data_loaders
from utils import save_model, save_plots, SaveBestModel
from torch.utils.tensorboard import SummaryWriter
from datasets import trans_img
import os
import pandas as pd
import PIL



def main():
    # construct the argument parser
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train our network for')
    my_parser.add_argument('--gpu', type=int, default=0, help='Number GPU 0,1')
    my_parser.add_argument('--base_path', type=str, help='Path to CSV. [train.csv, test.csv]')
    my_parser.add_argument('--save_dir', type=str, help='Path to Save output training model')
    my_parser.add_argument('--name', type=str, help='Name to save output in save_dir')
    my_parser.add_argument('--resume', action='store_true')
    my_parser.add_argument('--modelPATH', type=str, default='', help='path/to/checkpoint/...pth')
    my_parser.add_argument('--lr', type=float, default=1e-4)
    
    
    args = my_parser.parse_args()

    ## set gpu
    gpu = args.gpu
    ## get my_parser
    save_dir = args.save_dir
    name = args.name
    base_path = args.base_path
    ## get dataset
    ##train set
    train_path = f'{base_path}/MSDT_datatrain.csv'
    ##test set
    test_path = f'{base_path}/MSDT_datatest.csv'
    # Prepare the Required Data Loaders and Define the Learning Parameters
    # data constants ## create dataset
    train_img_list, train_label_list, val_img_list, val_label_list, test_img_list, test_label_list = split_datasets(train_path, test_path,  0.2)
    ## Create dataset
    transforms_img = trans_img()
    #transforms_valid = trans_valid()

    dataset_train = create_datasets(img_list=train_img_list, label_list=train_label_list, transforms=transforms_img)
    dataset_valid = create_datasets(img_list=val_img_list, label_list=val_label_list, transforms=transforms_img)
    dataset_test = create_datasets(img_list=test_img_list, label_list=test_label_list, transforms=transforms_img)
    ## Data loader
    train_loader, valid_loader, test_loader = create_data_loaders(dataset_train, dataset_valid, dataset_test)

    
    # computation device
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}" 
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")
    
    ## Create Directory to save output
    import imageio
    mkdir_weight = f'{save_dir}/{name}/weight'
    os.makedirs(mkdir_weight, exist_ok=True) 
    mkdir_checkpoint = f'{save_dir}/{name}/checkpoint'
    os.makedirs(mkdir_checkpoint, exist_ok=True) 
    save_pic = f'{save_dir}/{name}/picture'
    os.makedirs(save_pic, exist_ok=True) 
    save_tensorboard = f'{save_dir}/{name}/tensorboard_logs'
    os.makedirs(save_tensorboard, exist_ok=True)
    ## Seting SummaryWriter tensorboard pytorch
    writer = SummaryWriter(save_tensorboard)
    
    # learning_parameters 
    lr = args.lr
    epochs = args.epochs
    
    ##** Check if Resume to train model.
    if args.resume:
        PATH = args.modelPATH
        ModelFinalName = 'RheoVitRegress_final_R2model.pth'
        ModelbestName =  'RheoVitRegress_best_R2model.pth'
        EntireModelName = 'RheoVitRegress_Entire_R2model.pth'
        ## load checkpoint
        model = load_chackpoint(PATH, dropout=True)
        print(model)
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.\n")
    else:
        ModelFinalName = 'RheoVitRegress_final_R1model.pth'
        ModelbestName =  'RheoVitRegress_best_R1model.pth'
        EntireModelName = 'RheoVitRegress_Entire_R1model.pth'
        # build the model
        model = modify_model(pretrained=True, fine_tune=False, dropout=False).to(device)
        print(model)
        # total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.\n")
 
    
    save_weight = f'{mkdir_weight}/{ModelFinalName}'
    save_checkpoint = f'{mkdir_checkpoint}/{ModelbestName}'
    save_entire = f'{mkdir_weight}/{EntireModelName}'
    
    # Define the loss function
    criterion = nn.MSELoss()
    # Define the optimizer
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel(save_dir=save_checkpoint)
    
    # training
    def train(model, trainloader, optimizer, criterion):
        model.train()
        print('Training')
        train_running_loss = 0.0
        counter = 0
        for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
            counter += 1
            inputs, targets = data
            image = inputs.to(device)
            labels = targets.to(device)
            labels = labels.reshape((labels.shape[0], 1))
            labels = labels.float()

            optimizer.zero_grad()
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()
            # backpropagation
            loss.backward()
            # update the optimizer parameters
            optimizer.step()

        # loss for the complete epoch
        epoch_loss = train_running_loss / counter
 
        return epoch_loss

    # validation
    def validate(model, testloader, criterion):
        model.eval()
        print('Validation')
        valid_running_loss = 0.0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(testloader), total=len(testloader)):
                counter += 1
                inputs, targets = data
                image = inputs.to(device)
                labels = targets.to(device)
                labels = labels.reshape((labels.shape[0], 1))
                labels = labels.float()
                # forward pass
                outputs = model(image)
                # calculate the loss
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()
                
        # loss for the complete epoch
        epoch_loss = valid_running_loss / counter

        return epoch_loss


    # lists to keep track of losses: MSE
    train_loss, valid_loss = [], []
    # start the training #train_loader, valid_loader, test_loader
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss = train(model, train_loader,optimizer, criterion)
        valid_epoch_loss = validate(model, valid_loader, criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        print(f"Training loss: {train_epoch_loss:.4}, Validation loss: {valid_epoch_loss:.4}")
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(valid_epoch_loss, epoch, model, optimizer, criterion, save_dir=save_checkpoint)
        print('-'*50)

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/valid', valid_epoch_loss, epoch)
        
    # save the trained model weights for a final time
    save_model(epochs, model, optimizer, criterion, save_dir=save_weight)
    # Save the Entire Model
    torch.save(model, save_entire)
    # save the loss and plots
    save_plots(train_loss, valid_loss, save_dir=save_pic)
    print('='*125)
    print('TRAINING COMPLETE')
    print(f'Save [Rheology] Visiontranformer Linear Regression Model as : --> {mkdir_weight}')


## Run Function 
if __name__ == '__main__':
    main()
    
    
