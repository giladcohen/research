import matplotlib.pyplot as plt
import datetime
import glob
import os
import pickle
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj
from captum.influence._utils.common import _load_flexible_state_dict
from sklearn.metrics import auc, roc_curve
from torch.utils.data import DataLoader, Dataset, Subset

warnings.filterwarnings("ignore")

######################################### Identifying Influential Examples #########################################
# Define net
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
inverse_normalize = transforms.Compose([
    transforms.Normalize(mean = [0., 0., 0.], std = [1/0.5, 1/0.5, 1/0.5]),
    transforms.Normalize(mean = [-0.5, -0.5, -0.5], std = [1., 1., 1.]),
])
correct_dataset_path = "/Users/giladcohen/data/dataset/cifar10"
correct_dataset = torchvision.datasets.CIFAR10(root=correct_dataset_path, train=True, download=True, transform=normalize)
test_dataset = torchvision.datasets.CIFAR10(root=correct_dataset_path, train=False, download=True, transform=normalize)

def train(net, num_epochs, train_dataloader, test_dataloader, checkpoints_dir, save_every):

    start_time = datetime.datetime.now()

    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        epoch_loss = 0.0
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            # get the inputs
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % 100 == 0:  # print every 100 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                epoch_loss += running_loss
                running_loss = 0.0

        if epoch % save_every == 0:
            checkpoint_name = "-".join(["checkpoint", str(epoch) + ".pt"])
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": epoch_loss,
                },
                os.path.join(checkpoints_dir, checkpoint_name),
            )

        # Calcualate validation accuracy
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print("Accuracy of the network on test set at epoch %d: %d %%" % (epoch, 100 * correct / total))

    total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
    print("Finished training in %.2f minutes" % total_minutes)

correct_dataset_checkpoints_dir = '/data/gilad/logs/mi/cifar10/tracin_tutorial/checkpoints/cifar_10_correct_dataset'
num_epochs = 26
do_training = False # change to `True` if you want to do training
if do_training:
    train(net, num_epochs, DataLoader(correct_dataset, batch_size=128, shuffle=True), DataLoader(test_dataset, batch_size=128, shuffle=True), correct_dataset_checkpoints_dir, save_every=5)
elif not os.path.exists(correct_dataset_checkpoints_dir):
    # this should download the zipped folder of checkpoints from the S3 bucket
    # then unzip the folder to produce checkpoints in the folder `checkpoints/cifar_10_correct_dataset`
    # this is done if checkpoints do not already exist in the folder
    # if the below commands do not work, please manually download and unzip the folder to produce checkpoints in that folder
    os.makedirs(correct_dataset_checkpoints_dir)
    # !wget https://pytorch.s3.amazonaws.com/models/captum/influence-tutorials/cifar_10_correct_dataset.zip -O checkpoints/cifar_10_correct_dataset.zip
    # !unzip -o checkpoints/cifar_10_correct_dataset.zip -d checkpoints

correct_dataset_checkpoint_paths = glob.glob(os.path.join(correct_dataset_checkpoints_dir, "*.pt"))

def checkpoints_load_func(net, path):
    _load_flexible_state_dict(net, path, keyname="model_state_dict")
    return 1.

correct_dataset_final_checkpoint = os.path.join(correct_dataset_checkpoints_dir, "-".join(['checkpoint', str(num_epochs - 1) + '.pt']))
checkpoints_load_func(net, correct_dataset_final_checkpoint)

# Now, we define test_examples_batch, the batch of test examples to identify influential examples for, and also store the correct as well as predicted labels.
test_examples_indices = [0,1,2,3]
test_examples_batch = torch.stack([test_dataset[i][0] for i in test_examples_indices])
test_examples_predicted_probs, test_examples_predicted_labels = torch.max(F.softmax(net(test_examples_batch), dim=1), dim=1)
test_examples_true_labels = torch.Tensor([test_dataset[i][1] for i in test_examples_indices]).long()

# Choosing the TracInCP implementation to use
tracin_cp_fast = TracInCPFast(
    model=net,
    final_fc_layer=net.fc3,  # a reference or the name of the last fully-connected layer whose
                             # gradients will be used to calculate influence scores.
                             # This must be the last layer.
    influence_src_dataset=correct_dataset,
    checkpoints=correct_dataset_checkpoint_paths,
    checkpoints_load_func=checkpoints_load_func,
    loss_fn=nn.CrossEntropyLoss(reduction="sum"),
    batch_size=2048,
    vectorize=False,
)

# Compute the proponents / opponents using TracInCPFast
k = 10
start_time = datetime.datetime.now()
proponents_indices, proponents_influence_scores = tracin_cp_fast.influence(
    test_examples_batch, test_examples_true_labels, k=k, proponents=True
)
opponents_indices, opponents_influence_scores = tracin_cp_fast.influence(
    test_examples_batch, test_examples_true_labels, k=k, proponents=False
)
total_minutes = (datetime.datetime.now() - start_time).total_seconds() / 60.0
print(
    "Computed proponents / opponents over a dataset of %d examples in %.2f minutes"
    % (len(correct_dataset), total_minutes)
)
# proponents_indices.shape = [4, 10]
# proponents_influence_scores.shape = [4, 10]

# Define helper functions for displaying results
label_to_class = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trimshow_transformuck')
imshow_transform = lambda tensor_in_dataset: inverse_normalize(tensor_in_dataset.squeeze()).permute(1, 2, 0)
def display_test_example(example, true_label, predicted_label, predicted_prob, label_to_class):
    fig, ax = plt.subplots()
    print('true_class:', label_to_class[true_label])
    print('predicted_class:', label_to_class[predicted_label])
    print('predicted_prob', predicted_prob)
    ax.imshow(torch.clip(imshow_transform(example), 0, 1))
    plt.show()

def display_training_examples(examples, true_labels, label_to_class, figsize=(10,4)):
    fig = plt.figure(figsize=figsize)
    num_examples = len(examples)
    for i in range(num_examples):
        ax = fig.add_subplot(1, num_examples, i+1)
        ax.imshow(torch.clip(imshow_transform(examples[i]), 0, 1))
        ax.set_title(label_to_class[true_labels[i]])
    plt.show()
    return fig

def display_proponents_and_opponents(test_examples_batch, proponents_indices, opponents_indices, test_examples_true_labels, test_examples_predicted_labels, test_examples_predicted_probs):
    for (
            test_example,
            test_example_proponents,
            test_example_opponents,
            test_example_true_label,
            test_example_predicted_label,
            test_example_predicted_prob,
    ) in zip(
        test_examples_batch,
        proponents_indices,
        opponents_indices,
        test_examples_true_labels,
        test_examples_predicted_labels,
        test_examples_predicted_probs,
    ):

        print("test example:")
        display_test_example(
            test_example,
            test_example_true_label,
            test_example_predicted_label,
            test_example_predicted_prob,
            label_to_class,
        )

        print("proponents:")
        test_example_proponents_tensors, test_example_proponents_labels = zip(
            *[correct_dataset[i] for i in test_example_proponents]
        )
        display_training_examples(
            test_example_proponents_tensors, test_example_proponents_labels, label_to_class, figsize=(20, 8)
        )

        print("opponents:")
        test_example_opponents_tensors, test_example_opponents_labels = zip(
            *[correct_dataset[i] for i in test_example_opponents]
        )
        display_training_examples(
            test_example_opponents_tensors, test_example_opponents_labels, label_to_class, figsize=(20, 8)
        )

display_proponents_and_opponents(
    test_examples_batch,
    proponents_indices,
    opponents_indices,
    test_examples_true_labels,
    test_examples_predicted_labels,
    test_examples_predicted_probs,
)