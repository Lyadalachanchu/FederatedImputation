import os
import pickle

import numpy as np
import torch
from src.impute import impute_cvae_naive
from src.utils import get_dataset
from src.vae.mnist_vae import ConditionalVae

def load_dataset(dirichlet):
    filepath = f'C:\\Users\\LohithSai\\Desktop\\FederatedImputation\\src\\datasets\\gen_dataset_{dirichlet}.pkl'
    if not os.path.exists(filepath):
        # Load cvae that is trained federatedly
        vae_model = ConditionalVae(dim_encoding=3)
        checkpoint = torch.load(f"C:\\Users\\LohithSai\\Desktop\\FederatedImputation\\vae_data\\models\\0_cvae_{dirichlet}_cvae.pth")
        vae_model.load_state_dict(checkpoint)
        gen_dataset = impute_cvae_naive(k=70000, trained_cvae = vae_model, initial_dataset = torch.tensor([]))
        with open(filepath, 'wb') as f:
            pickle.dump(gen_dataset, f)
    with open(filepath, 'rb') as f:
        gen_dataset = pickle.load(f)
    return gen_dataset

def train(generated_dataset, real_test_dataset):
    from src.global_model_logic import get_dirch_datalaoders
    from tqdm import tqdm
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from torch.version import cuda
    from src.models import ExquisiteNetV1
    import torch
    from torch import nn, optim
    device = 'cuda:0'
    print(device)

    trainloader = DataLoader(generated_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(real_test_dataset, batch_size=32, shuffle=True)
    # Assuming 'model' is your model
    model = ExquisiteNetV1(class_num=10, img_channels=1)
    model = model.to(device)  # Move model to GPU if available

    # Define the loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs to train the model
    n_epochs = 15
    train_losses = []
    test_losses = []
    f1_scores = []
    accuracies = []
    correct_predictions = 0
    total_predictions = 0
    for epoch in tqdm(range(n_epochs)):
        train_loss = 0.0
        pred_labels = []
        actual_labels = []
        for data, target in trainloader:  # Assuming 'trainloader' is your DataLoader
            data, target = data.to(device), target.to(device)  # Move data to GPU if available

            # Clear the gradients of all optimized variables
            optimizer.zero_grad()

            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            pred_labels.append(output.argmax(dim=1))
            actual_labels.append(target)

            # Calculate the loss
            loss = criterion(output, target)

            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()

            # Perform single optimization step (parameter update)
            optimizer.step()

            # Update running training loss
            train_loss += loss.item() * data.size(0)

        # Switch to evaluation mode
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            test_pred_labels = []
            test_actual_labels = []
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                test_pred_labels.append(output.argmax(dim=1))
                test_actual_labels.append(target)
                # Compare with actual classes
                total_predictions += output.argmax(dim=1).size(0)
                # correct_predictions += (predicted == labels).sum().item()
                correct_predictions += (output.argmax(dim=1) == target).sum().item()
        # Compute average test loss
        train_loss = train_loss / len(trainloader.dataset)
        test_loss = test_loss / len(testloader.dataset)
        test_losses.append(test_loss)
        train_losses.append(train_loss)
        # Calculate F1 score for the test data
        test_pred_labels = torch.cat(test_pred_labels).to('cpu').numpy()
        test_actual_labels = torch.cat(test_actual_labels).to('cpu').numpy()
        test_f1_score = f1_score(test_actual_labels, test_pred_labels, average='macro')
        f1_scores.append(test_f1_score)
        accuracy = correct_predictions / total_predictions
        accuracies.append(accuracy)

        print(f'Accuracy: {accuracy * 100}%')
        print('Epoch: {} \tTraining Loss: {:.6f} \t Test Loss: {:.6f} \tF1 Test Macro: {:.6f}'.format(
            epoch + 1,
            train_loss,
            test_loss,
            test_f1_score
        ))
    results = {"train_losses":train_losses, "test_losses":test_losses, "f1_scores": f1_scores, "accuracies":accuracies}
    return model, results

class args:
    def __init__(self):
        self.num_channels = 1
        self.iid = 1
        self.num_classes = 10
        self.num_users = 10
        self.dataset = 'mnist'

if __name__ == "__main__":
    dirichlets = [0.1, 0.3, 0.5, 0.8, 1.0]
    train_dataset, test_dataset, user_groups = get_dataset(args())
    real_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    for dirichlet in dirichlets:
        gen_dataset = load_dataset(dirichlet)
        trained_model, results = train(gen_dataset, real_dataset)
        torch.save(trained_model, f"classifier_global_{dirichlet}.pth")
        with open(f'./results/results_{dirichlet}', 'wb') as f:
            pickle.dump(results, f)

