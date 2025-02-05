import os
import torch
from torch import nn, optim
from torchvision import models

def build_model(arch='vgg16', hidden_units=512, learning_rate=0.001, device='cpu'):
    
    """
    Build a model according to what specified in the args
    """
    if arch=="vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features
    elif arch == 'googlenet':
        model = models.googlenet(pretrained=True)
        input_size = model.fc.in_features
    else:
        raise ValueError(f"Unsupported architecture: {arch}. try choosing from resnet18, resnet50, vgg16, or googlenet")
        
    # freeze the feature extraction params
    for param in model.parameters():
        param.requires_grad = False
        
    # define the new classifier
    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # set the model clssifier/fc
    if arch in ['vgg16']:
        model.classifier = classifier
    elif arch in ['googlenet', 'resnet50', 'resnet18']:
        model.fc = classifier
    
    # define learning funcitons
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters() if arch in ['vgg16'] else model.fc.parameters(), lr=learning_rate)
    
    # ensure model to moved to device 
    model.to(device)
    return model, criterion, optimizer
    
    
def save_checkpoint(model, optimizer, class_to_idx, save_path='checkpoint_.pth', arch='vgg16', hidden_units=512, learning_rate=0.001, epochs=5):
    """
    Save model checkpoint after training
    """
    checkpoint = {
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'hidden_units': hidden_units,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'class_to_idx': class_to_idx
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint successfully saved to path: {save_path}")
    
    
def load_checkpoint(filepath, device='cpu'):
    """
    Load a pretrained model from a checkpoint
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at path: '{filepath}'!")
      
    # load checkpoint to device 
    checkpoint = torch.load(filepath, map_location=device)
    
    # rebuild the model with our previous function passing the chekpoint keys
    if all(key in checkpoint for key in ['arch', 'hidden_units', 'learning_rate']):
        model, _, _ = build_model(arch=checkpoint['arch'], hidden_units=checkpoint['hidden_units'], learning_rate=checkpoint['learning_rate'], device=device)
    else:
        model, _, _ = build_model()
    
    # load saved parameters insteaad
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    print(f"Model successfully loaded from {filepath}")
    
    return model
    
    
    