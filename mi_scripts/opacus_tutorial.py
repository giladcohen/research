import warnings
import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision import models
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description='Training DP networks using PyTorch')
parser.add_argument('--checkpoint_dir', default='/data/gilad/logs/mi/debug', type=str, help='checkpoint dir')

# dataset
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset: cifar10, cifar100, svhn, tiny_imagenet')
parser.add_argument('--train_size', default=0.5, type=float, help='Fraction of train size out of entire trainset')
parser.add_argument('--val_size', default=0.05, type=float, help='Fraction of validation size out of entire trainset')

# optimization:
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--mom', default=0.9, type=float, help='weight momentum of SGD optimizer')
parser.add_argument('--epochs', default='400', type=int, help='number of epochs')
parser.add_argument('--num_workers', default=4, type=int, help='Data loading threads')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')

# LR scheduler
parser.add_argument('--lr_scheduler', default='reduce_on_plateau', type=str, help='reduce_on_plateau/multi_step')
parser.add_argument('--factor', default=0.9, type=float, help='LR schedule factor')
parser.add_argument('--patience', default=3, type=int, help='LR schedule patience')
parser.add_argument('--cooldown', default=0, type=int, help='LR cooldown')

parser.add_argument('--mode', default='null', type=str, help='to bypass pycharm bug')
parser.add_argument('--port', default='null', type=str, help='to bypass pycharm bug')

args = parser.parse_args()


MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20
LR = 1e-3

BATCH_SIZE = 128
MAX_PHYSICAL_BATCH_SIZE = 32

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
# CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
# CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])

DATA_ROOT = '/data/dataset/cifar10'
train_dataset = CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

model = models.resnet18(num_classes=10)

errors = ModuleValidator.validate(model, strict=False)
# errors[-5:]
model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)

def accuracy(preds, labels):
    return (preds == labels).mean()


privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    train(model, train_loader, optimizer, epoch + 1, device)

top1_acc = test(model, test_loader, device)
print('test accuracy: {}'.format(top1_acc))
