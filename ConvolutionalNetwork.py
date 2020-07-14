import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetRegressor
PATH = "trained_model.pt"


class ConvolutionalNetwork(nn.Module):
    """ My model 3 convolutional layers and 2 fully connected layers."""
    def __init__(self):
        super(ConvolutionalNetwork, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1))
        self.fc1 = nn.Linear(4 * 4 * 64, 200)
        self.fc2 = nn.Linear(200, 7)

    def forward(self, x):
        """ Forward pass through network."""
        x = x.unsqueeze(dim=1)
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.reshape(out.size(0), -1)
        out = torch.flatten(out,start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def train(model, samples, targets, num_epochs=8 ):
    """ Trainning function with optimal number of epochs and learning rate and batchsize."""
    batch_size = 1
    learning_rate = 0.001
    loss_list = []
    acc_list = []
    all_predictions = []
    all_targets = []
    accuracy_track = []
    progress_track = []
    correct = 0
    print("Training.")
    trainset = torch.utils.data.TensorDataset(samples, targets)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i,(vectors, labels) in enumerate(train_loader):
            # Run the forward pass
            outputs = model(vectors)
            loss = criterion(outputs.view(batch_size,7), labels.view(batch_size).type(torch.LongTensor))
            loss_list.append(loss.item())

            # Backprop and perform Adam optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            d = outputs.data
            _, predicted = torch.max(d,1)
            all_predictions.append(predicted.view(batch_size).type(torch.float32))
            all_targets.append(labels)
            correct = correct + (predicted.view(batch_size).type(torch.float32) == labels).sum().item()
            acc_list.append(correct / 10)
            if (i + 1) % 1000 == 0:
                acc = correct / (1000*batch_size) * 100
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),acc))
                if acc <= 100:
                    accuracy_track.append(acc)
                    progress_track.append(total_step*epoch + i)
                correct = 0
    torch.save(model.state_dict(), PATH)
    plt.figure()
    plt.plot(progress_track, accuracy_track)
    plt.title("Accuracy graph")
    plt.ylabel("Accuracy")
    plt.xlabel("Training progress")
    return model

def test(model, samples, targets):
    """ Testing function """
    correct = 0
    testset = torch.utils.data.TensorDataset(samples, targets)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False)
    total_step = len(samples)
    total_correct = 0
    target_conf = []
    pred_conf = []
    class_labels = ["AddToPlaylist","BookRestaurant","GetWeather","PlayMusic","RateBook","SearchCreativeWork","SearchScreeningEvent"]
    for i, (vectors, labels) in enumerate(test_loader):
        # Run the forward pass
        outputs = model(vectors)
        d = outputs.data
        _, predicted = torch.max(d,1)
        correct = correct + (predicted.view(1).type(torch.float32) == labels).sum().item()
        target_conf.append(int(labels.item()))
        pred_conf.append(predicted.item())
        # Update on accuracy
        if (i + 1) % 10 == 0:
            print('Testing step [{}/{}], Accuracy: {:.2f}%'
                  .format( i + 1, total_step,
                          (correct / 10) * 100))
            total_correct = total_correct + correct
            correct = 0
    conf_mat = confusion_matrix(target_conf,pred_conf)
    plt.figure()
    sns.heatmap(conf_mat, cmap="RdYlBu", annot=True, cbar=False, square=True,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel("Predicted values")
    plt.ylabel("Actual values")
    plt.show()
    plt.savefig("confusion_matrix.png")
    # Final results and report
    print('Test completed \n Results: [{}/{}], {:.2f}'.format(total_correct,total_step,(total_correct / total_step) * 100))
    print(classification_report(target_conf, pred_conf))

def hyper_tunning(model, samples, target):
    """ Hyperparameters tuning using grid seacrh from sklearn."""
    params = {
        'lr':[0.001, 0.0005, 0.005, 0.05],
        'max_epochs': list(range(1, 12, 4)),
        'batch_size': [1]
    }
    binary_target = []
    for t in target:
        tem = np.zeros(7)
        index = int(t.item())
        tem[index] = 1
        binary_target.append(tem)
    binary_target = torch.FloatTensor(binary_target)
    net = NeuralNetRegressor(model
                             , max_epochs=12
                             , lr=0.001
                             , verbose=1)
    gs = GridSearchCV(net, params, refit=False, scoring='r2', verbose=1, cv=7)
    target = target.reshape(-1, 1)
    gs.fit(samples, binary_target)
