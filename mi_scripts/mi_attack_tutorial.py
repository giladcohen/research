import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from art.utils import load_nursery

(x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)

# Train random forest model
from sklearn.ensemble import RandomForestClassifier
from art.estimators.classification.scikitlearn import ScikitlearnRandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train, y_train)
art_classifier = ScikitlearnRandomForestClassifier(model)
print('Base model accuracy: ', model.score(x_test, y_test))

# Rule-based attack
import numpy as np
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased

attack = MembershipInferenceBlackBoxRuleBased(art_classifier)
# infer attacked feature
inferred_train = attack.infer(x_train, y_train)
inferred_test = attack.infer(x_test, y_test)
# check accuracy
train_acc = np.sum(inferred_train) / len(inferred_train)
test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
print(train_acc)
print(test_acc)
print(acc)

def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1

    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall

print(calc_precision_recall(np.concatenate((inferred_train, inferred_test)),
                            np.concatenate((np.ones(len(inferred_train)), np.zeros(len(inferred_test))))))

# Black-box attack
#
# The black-box attack basically trains an additional classifier (called the attack model) to predict the membership
# status of a sample. It can use as input to the learning process probabilities/logits or losses, depending on the type
# of model and provided configuration.
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

attack_train_ratio = 0.5
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

bb_attack = MembershipInferenceBlackBox(art_classifier)

# train attack model
bb_attack.fit(x_train[:attack_train_size], y_train[:attack_train_size],
              x_test[:attack_test_size], y_test[:attack_test_size])
# Infer sensitive feature and check accuracy
# get inferred values
inferred_train_bb = bb_attack.infer(x_train[attack_train_size:], y_train[attack_train_size:])
inferred_test_bb = bb_attack.infer(x_test[attack_test_size:], y_test[attack_test_size:])
# check accuracy
train_acc = np.sum(inferred_train_bb) / len(inferred_train_bb)
test_acc = 1 - (np.sum(inferred_test_bb) / len(inferred_test_bb))
acc = (train_acc * len(inferred_train_bb) + test_acc * len(inferred_test_bb)) / (len(inferred_train_bb) + len(inferred_test_bb))
print(train_acc)
print(test_acc)
print(acc)
# black-box
print(calc_precision_recall(np.concatenate((inferred_train_bb, inferred_test_bb)),
                            np.concatenate((np.ones(len(inferred_train_bb)), np.zeros(len(inferred_test_bb))))))

# Train neural network model
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from art.estimators.classification.pytorch import PyTorchClassifier

# reduce size of training set to make attack slightly better
train_set_size = 500
x_train = x_train[:train_set_size]
y_train = y_train[:train_set_size]
x_test = x_test[:train_set_size]
y_test = y_test[:train_set_size]
attack_train_size = int(len(x_train) * attack_train_ratio)
attack_test_size = int(len(x_test) * attack_train_ratio)

class ModelToAttack(nn.Module):

    def __init__(self, num_classes, num_features):
        super(ModelToAttack, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Tanh(), )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(), )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(), )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)

mlp_model = ModelToAttack(4, 24)
mlp_model = torch.nn.DataParallel(mlp_model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)

class NurseryDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.from_numpy(x.astype(np.float64)).type(torch.FloatTensor)

        if y is not None:
            self.y = torch.from_numpy(y.astype(np.int8)).type(torch.LongTensor)
        else:
            self.y = torch.zeros(x.shape[0])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if idx >= len(self.x):
            raise IndexError("Invalid Index")

        return self.x[idx], self.y[idx]

train_set = NurseryDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)

for epoch in range(20):
    for (input, targets) in train_loader:
        input, targets = torch.autograd.Variable(input), torch.autograd.Variable(targets)
        input = input.cuda()
        targets = targets.cuda()

        optimizer.zero_grad()
        outputs = mlp_model(input)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

mlp_art_model = PyTorchClassifier(model=mlp_model, loss=criterion, optimizer=optimizer, input_shape=(24,), nb_classes=4)
pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(x_test.astype(np.float32))])
print('Base model accuracy: ', np.sum(pred == y_test) / len(y_test))

# rule based attack
mlp_attack = MembershipInferenceBlackBoxRuleBased(mlp_art_model)

# infer
mlp_inferred_train = mlp_attack.infer(x_train.astype(np.float32), y_train)
mlp_inferred_test = mlp_attack.infer(x_test.astype(np.float32), y_test)

# check accuracy
mlp_train_acc = np.sum(mlp_inferred_train) / len(mlp_inferred_train)
mlp_test_acc = 1 - (np.sum(mlp_inferred_test) / len(mlp_inferred_test))
mlp_acc = (mlp_train_acc * len(mlp_inferred_train) + mlp_test_acc * len(mlp_inferred_test)) / (len(mlp_inferred_train) + len(mlp_inferred_test))
print(mlp_train_acc)
print(mlp_test_acc)
print(mlp_acc)

print(calc_precision_recall(np.concatenate((mlp_inferred_train, mlp_inferred_test)),
                            np.concatenate((np.ones(len(mlp_inferred_train)), np.zeros(len(mlp_inferred_test))))))

# Black-box attack
mlp_attack_bb = MembershipInferenceBlackBox(mlp_art_model, attack_model_type='rf')

# train attack model
mlp_attack_bb.fit(x_train[:attack_train_size].astype(np.float32), y_train[:attack_train_size],
                  x_test[:attack_test_size].astype(np.float32), y_test[:attack_test_size])

# infer
mlp_inferred_train_bb = mlp_attack_bb.infer(x_train[attack_train_size:].astype(np.float32), y_train[attack_train_size:])
mlp_inferred_test_bb = mlp_attack_bb.infer(x_test[attack_test_size:].astype(np.float32), y_test[attack_test_size:])

# check accuracy
mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))
print(mlp_train_acc_bb)
print(mlp_test_acc_bb)
print(mlp_acc_bb)

print(calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)),
                            np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb))))))
