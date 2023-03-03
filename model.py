import torchvision.models as models
import torch.nn as nn
import timm
import torch


def set_dropout_p(m, p):
    if isinstance(m, nn.Dropout):
        m.p = p

        
def modify_model(pretrained, fine_tune, dropout):
    """
    Function to build the neural network model. Returns the final model.
    Parameters
    :param pretrained (bool): Whether to load the pre-trained weights or not.
    :param fine_tune (bool): Whether to train the hidden layers or not.
    :param num_classes (int): Number of classes in the dataset. 
    :dropout :Dropout to be True for all the different layers with prob == 0.1.
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
    model = timm.create_model("vit_large_patch32_384", pretrained=pretrained)
    if fine_tune:
        print('[INFO]: Fine-tuning all layers...')
        for params in model.parameters():
            params.requires_grad = True
    elif not fine_tune:
        print('[INFO]: Freezing hidden layers...')
        for params in model.parameters():
            params.requires_grad = False
            
    # Remove the final classification layer
   #model.head = nn.Identity()
    # Add a new regression head
    #model.regressor = nn.Linear(in_features=model.embed_dim, out_features=1)
    model.head = nn.Linear(in_features=model.embed_dim, out_features=1)
    
    if dropout:
        print('[INFO]: Dropout to be True for all the different layers ...')
        model.apply(lambda m: set_dropout_p(m, p=0.1))
  
    return model 

 
def build_model(pretrained, n_classes, dropout):
    """
    Function to build the neural network model. Returns the final model.
    Parameters
    :param pretrained (bool): Whether to load the pre-trained weights or not.
    :param num_classes (int): Number of classes in the dataset. 
    """
    if pretrained:
        print('[INFO]: Loading pre-trained weights')
    elif not pretrained:
        print('[INFO]: Not loading pre-trained weights')
        
    model = timm.create_model("vit_large_patch32_384", pretrained=pretrained)    
    # Remove the final classification layer
#     model.head = nn.Identity()
#     # Add a new regression head
#     model.regressor = nn.Linear(in_features=model.embed_dim, out_features=1)
    model.head = nn.Linear(in_features=model.embed_dim, out_features=1)
    if dropout:
        print('[INFO]: Dropout to be True for all the different layers ...')
        model.apply(lambda m: set_dropout_p(m, p=0.1))
  
    return model 


def resume_model(PATH, n_classes, dropout, lr):
    model = timm.create_model("vit_large_patch32_384", num_classes=n_classes, pretrained=False)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    model.train()
    if dropout:
        print('[INFO]: Dropout to be True for all the different layers ...')
        model.apply(lambda m: set_dropout_p(m, p=0.1))
        
    return model 


def load_chackpoint(PATH, dropout):
    model = torch.load(PATH)
    model = model.to(torch.device('cuda'))
    model.train()
    if dropout:
        print('[INFO]: Dropout to be True for all the different layers ...')
        model.apply(lambda m: set_dropout_p(m, p=0.1))
    
    return model







