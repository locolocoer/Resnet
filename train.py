from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from resnet import *

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                normalize])

train_data = datasets.CIFAR10("./data", train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

val_data = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    normalize,
]), download=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

if __name__ == "__main__":
    losses=[]
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Resnet(BasicBlock, [3, 3, 3])
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01, weight_decay=0.2)
    for eopch in range(10):
        for i, (input, target) in enumerate(train_loader):
            input,target=input.to(device),target.to(device)
            output = model(input)
            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if (i % 3 == 0):
               print("{}epoch,{}batchs,loss:{}".format(eopch,i,loss.item()))
