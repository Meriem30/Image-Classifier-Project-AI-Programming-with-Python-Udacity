import argparse
import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.optim as optim
from data_utils import load_data
from model_utils import build_model, save_checkpoint



def train(data_dir, arch, hidden_units, learning_rate, epochs, save_dir, gpu):
    # set device
    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    
    # load data from the dataloader function passing the data directory
    train_loader, valid_loader, _ , class_to_idx = load_data(data_dir)
    
    # prepare the model
    model, criterion, optimizer = build_model(arch, hidden_units, learning_rate)
    model.class_to_idex = class_to_idx 
    
    # move model to device 
    model.to(device)
    
    print("Start Training now ...")

    # Training loop
    for epoch in range(epochs):
        # reset the training loss to 0 at the begining of each epoch
        running_loss = 0
        # switch back to training mode for current iteration (after eval mode for validation)
        model.train()
        for inputs, labels in train_loader:
            # move loaded data to cuda
            inputs, labels = inputs.to(device), labels.to(device)
            # clean gradient
            optimizer.zero_grad()

            # training step
            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Validation step
            # set the model to the evaluation mode
            model.eval()
            valid_loss = 0
            accuracy = 0

            with torch.no_grad():
                for valid_inputs, valid_labels in valid_loader:
                    # move data to GPU
                    valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                    # perform a forwardpass
                    valid_outputs = model(valid_inputs)
                    # calculate the loss of this batch
                    batch_loss = criterion(valid_outputs, valid_labels)
                    # acuumulate
                    valid_loss += batch_loss.item()

                # Accuracy Calculation
                # convert the logits back into actual probabilities since we used logsoftmax
                probs = torch.exp(valid_outputs)
                # return the top '1' element from the probs tensor (the highest probability and indice)
                top_p, top_class = probs.topk(1, dim=1)
                # compare to the ground truth classes resulting in a boolean tensor 
                equals = top_class == valid_labels.view(*top_class.shape)
                # use the mean of the converted to float tensor items
                #to get the nbr of correct predictions and add the current batch acc to total acc
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train Loss: {running_loss/len(train_loader):.3f}.. "
              f"Val Loss: {valid_loss/len(valid_loader):.3f}.. "
              f"Val Accuracy: {accuracy/len(valid_loader):.3f}")
    # dave the model checkpoint before leaving the     
    save_checkpoint(model, save_dir, arch, hidden_units, learning_rate, epochs)
    print(f"Training completed for {epochs} epochs. Resulted model saved at {save_dir}")    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hello ! This is you application to train a neural netwrok model.")
    parser.add_argument('data_dir', type=str, help="Your dataset directory")
    parser.add_argument('--save_dir', type=str, help="Your directory for saving model checkpoints")
    parser.add_argument('--arch', type=str, default="vgg16", help="Your chosen model architecture from torchvision.models(e.g. vgg16, resnet50, googlenet)")
    parser.add_argument('--hidden_units', type=int, default=512, help="The number of your hidden units")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="The learning rate of your training")
    parser.add_argument('--epochs', type=int, default=5, help="The number of epochs of your training")
    parser.add_argument('--gpu', action='store_true', help="Enable using GPU for your training")
    
    
    args = parser.parse_args()
    train(args.data_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.save_dir, args.gpu)

    