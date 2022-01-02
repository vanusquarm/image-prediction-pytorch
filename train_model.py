#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# TODO: Import dependencies for Debugging andd Profiling
# import smdebug.pytorch as smd

def test(model, test_loader):
    model.eval()
    for (inputs, labels) in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        print(torch.sum(preds == labels.data))
    

def train(model, train_loader, epochs, criterion, optimizer):
    # https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md
    model.train()
    count = 0
    for e in range(epochs):
        print(e)
        for (inputs, labels) in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            count += len(inputs)
            print(".", end="")
            break
        print("\n")
    return model

def net():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model

def create_data_loaders(data, batch_size, test_batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    
    trainset = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    validset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    testset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(validset, batch_size=test_batch_size, shuffle=False),
        torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False))

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model=net()
    model=model.to(device)
    loss_criterion = nn.CrossEntropyLoss()
    # nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    # optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)

    # https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md
    # hook = smd.Hook.create_from_json_file()
    # hook.register_module(net)
    # hook.register_loss(loss_criterion)
    
    train_loader, valid_loader, test_loader = create_data_loaders(args.data, args.batch_size, args.test_batch_size)
    
    model=train(model, train_loader, args.epochs, loss_criterion, optimizer)
    
    test(model, test_loader)
    
    torch.save(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument('--data', type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument('--model_dir', type=str, default="s3://sagemaker-us-east-1-709614815312/dogmodel")
    
    args=parser.parse_args()
    
    main(args)
