import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import PIL


# data constants
BATCH_SIZE = 64
NUM_WORKERS = 0
IMG_SIZE = 384


## Function Create dataset list
def create_listdataset(df):
    img_list = df['pathimg'].tolist()
    label_list = df['MSDT'].tolist()
    return img_list, label_list


def split_datasets(train_path, test_path, VALID_SPLIT): 
    train = pd.read_csv(train_path)
    print(f"Dataset Train Set : {train.shape[0]} images")
    test = pd.read_csv(test_path)
    print(f"Dataset Test Set : {test.shape[0]} images")
    print(f"*"*125)
    ## Split train, validation set
    trainset, validset = train_test_split(train, test_size=VALID_SPLIT, random_state=42, shuffle=True)
    print(f"Train set : {trainset.shape[0]} images")
    print(f"Validation set : {validset.shape[0]} images")
    ## Crate data list
    train_img_list, train_label_list = create_listdataset(trainset)
    print(f"For train set ; Images {len(train_img_list)} , Label; {len(train_label_list)}")
    val_img_list, val_label_list = create_listdataset(validset)
    print(f"For Validation set ; Images {len(val_img_list)} , Label; {len(val_label_list)}")
    test_img_list, test_label_list =  create_listdataset(test)
    print(f"For test set ; Images {len(test_img_list)} , Label; {len(test_label_list)}")
    
    return train_img_list, train_label_list, val_img_list, val_label_list, test_img_list, test_label_list
    

#The Training and Validation Transforms
# create image augmentations
# def trans_train():
#     transforms_train = transforms.Compose(
#         [
#             transforms.Resize((IMG_SIZE, IMG_SIZE)),
#             transforms.RandomHorizontalFlip(p=0.3),
#             transforms.RandomVerticalFlip(p=0.3),
#             transforms.RandomResizedCrop(IMG_SIZE),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ]
#     )
#     return transforms_train 

# def trans_valid():    
#     transforms_valid = transforms.Compose(
#         [
#             transforms.Resize((IMG_SIZE, IMG_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         ]
#     )
#     return transforms_valid 

def trans_img():    
    transforms_img = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms_img 


## Function to Create the Dataset
class create_datasets(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, img_list, label_list, transforms=None):
        self.img_list = img_list
        self.label_list = label_list
        #self.mode = mode  # train 
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        label = self.label_list[index]
        image = self.img_list[index]
        
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = PIL.Image.open(image).convert('RGB')
        
        if self.transforms is not None:
            transformed_img = self.transforms(img)
        
        return transformed_img, label

    
## Data Loader    
def create_data_loaders(dataset_train, dataset_valid, dataset_test):  
    """
    Function to build the data loaders.
    Parameters:
    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.
    :param dataset_test: The test dataset.
    """
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, 
                                                   num_workers=NUM_WORKERS)
    valid_loader = torch.utils.data.DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, 
                                                   num_workers=NUM_WORKERS)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, 
                                                  num_workers=NUM_WORKERS)
  
    return train_loader, valid_loader, test_loader
    
      