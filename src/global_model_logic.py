import pickle

import numpy as np
from tqdm import tqdm


def get_dirch_datalaoders(dirichlet_param):
    from torch.utils.data import random_split
    from torch.utils.data import DataLoader, Dataset

    lengths = [60000, 10000]
    data = pickle.load(open(f'C:\\Users\\LohithSai\\Desktop\\FederatedImputation\\src\\jupyter_notebook\\gen_dataset_{dirichlet_param}.pkl', 'rb'))

    generated_train_dataset, generated_test_dataset = random_split(data, lengths)

    trainloader = DataLoader(generated_train_dataset,
                             batch_size=32, shuffle=True)
    testloader = DataLoader(generated_test_dataset,
                            batch_size=32, shuffle=True)
    return trainloader, testloader

def train_model(dirichlet):
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader
    from torch.version import cuda
    from src.models import ExquisiteNetV1
    import torch
    from torch import nn, optim
    device = 'cuda:0'
    print(device)

    trainloader, testloader = get_dirch_datalaoders(dirichlet)
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
        print('Epoch: {} \tTraining Loss: {:.6f} \t Test Loss: {:.6f} \tF1 Test Macro: {:.6f}'.format(
            epoch + 1,
            train_loss,
            test_loss,
            test_f1_score
        ))
    torch.save(model.state_dict(), f'C:\\Users\\LohithSai\\Desktop\\FederatedImputation\\vae_data\\global_exq_{dirichlet}.pth')
    return train_losses, test_losses, f1_scores

def main_train_for_all_dirichlet_params():
    import pickle
    results = {}
    for dirichlet in tqdm(np.linspace(0.1, 1, 10)):
        dirichlet = round(dirichlet, 1)
        train_losses, test_losses, f1_scores = train_model(dirichlet)
        results[dirichlet] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'f1_scores': f1_scores
        }
    print(results)
    with open('results.pkl', 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    main_train_for_all_dirichlet_params()