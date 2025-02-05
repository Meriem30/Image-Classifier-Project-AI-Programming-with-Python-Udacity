import os
import argparse
import json

import torch
from torchvision import models
from PIL import Image

import numpy as np
from model_utils import load_checkpoint
from data_utils import process_image

def predict(image_path, model, top_k=5, category_names=None, gpu=False):
    """Return the predicted class of an image using a pretrained model"""
    # set and move model to device
    device = torch.device("cuda" if ( gpu and torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()
    
    # prepare and move the data to device
    image = process_image(image_path)
    image = image.unsqueeze(0).float()
    image = image.to(device)
    
    # process to inference
    with torch.no_grad():
        output = model(image)
        
    # get the probs and top_classes from the output
    probs = torch.exp(output)
    top_probs, top_classes = probs.topk(top_k, dim=1)
    
    # convert tensors to list
    top_probs = top_probs.cpu().numpy().flatten()
    top_classes = top_classes.cpu().numpy().flatten()
    
    # convert class indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_classes]
    
    # get classes names if categories provided
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name.get(str(cls), "NonDefined") for cls in top_classes]
        
    return top_probs, top_classes



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run predict function to classify an image on a trained model")
    parser.add_argument("image_path", type=str, help="Your path to image file")
    parser.add_argument("checkpoint", type=str, help="Your path of saved model checkpoint")
    parser.add_argument("--top_k", type=int, default=5, help="Precise the number of top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Your path to JSON file mapping categories to names")
    parser.add_argument("--gpu", action="store_true", help="Enable using a GPU for inference")
    
    args = parser.parse_args()
    
    # load cached model
    model = load_checkpoint(args.checkpoint)
    
    # predict top classes
    probs, classes = predict(args.image_path, model, args.top_k, args.category_names, args.gpu)
    
    # print predict() outputs
    print("Probabilities: ", probs)
    print("Clasees : ", classes)