import torch
import matplotlib.pyplot as plt
import PIL

plt.style.use('ggplot')

## save the best model while training.
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, save_dir, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        self.save_dir = save_dir
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, save_dir):
        #self.save_dir = save_dir
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_dir)  ##---** 1.RheoVitRegress_best_model.pth
            

## the code to save the model after the training completes, that is, the last epoch’s model.
def save_model(epochs, model, optimizer, criterion, save_dir):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_dir) ##---** 2. RheoVitRegress_final_model.pth
    
## function is for saving the loss and accuracy graphs for training and validation.
def save_plots(train_loss, valid_loss, save_dir):
    """
    Function to save the loss and accuracy plots to disk.
    """
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-', 
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-', 
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/loss.png')  ####---**
    
    
    