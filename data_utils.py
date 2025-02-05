import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np


def load_data(data_dir):
    """
    Load train, validation, and test datasets while apply transformations
    """
    # define data directories 
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # define transforms 
    train_transforms = transforms.Compose([
            transforms.RandomRotation(35),
            transforms.RandomResizedCrop(224), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    eval_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = ImageFolder(valid_dir, transform=eval_transforms)
    test_dataset =  ImageFolder(test_dir, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    
    
    # get class_to_idx mapping 
    class_to_idx = train_dataset.class_to_idx  

    return train_loader, valid_loader, test_loader, class_to_idx


def process_image(image_path):
    """
    Process an image preparing it for use in the app model
    """
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File '{image_path}' not found!")

    image = Image.open(image_path).convert("RGB")

    # define transformation as expected by the model
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # return a tensor image
    image = transform(image)

    return image  
    
