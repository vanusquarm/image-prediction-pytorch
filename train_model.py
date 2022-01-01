#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

# TODO: Import dependencies for Debugging andd Profiling
import smdebug.pytorch as smd

def test(model, test_loader):
    for (inputs, labels) in train_loader:
        outputs = net(inputs)
        _, preds = torch.max(outputs, 1)
        torch.sum(preds == labels.data)
    

def train(model, train_loader, criterion, optimizer):
    # https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md
    for (inputs, labels) in train_loader:
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
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
    trainset = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    trainset = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(validset, batch_size=test_batch_size, shuffle=True),
        torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True))

def main(args):
    model=net()    
    loss_criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr = args.lr, momentum=args.momentum)

    # https://github.com/awslabs/sagemaker-debugger/blob/master/docs/pytorch.md
    hook = smd.Hook.create_from_json_file()
    hook.register_module(net)
    hook.register_loss(loss_criterion)
    
    train_loader, test_loader = create_data_loaders(args.data, args.batch_size, args.test_batch_size)
    
    model=train(model, train_loader, loss_criterion, optimizer)
    
    test(model, test_loader, criterion)
    
    torch.save(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )
    parser.add_argument('--data', type=str)
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
