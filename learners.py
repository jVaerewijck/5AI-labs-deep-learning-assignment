import os
import math
import torch

from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader

from classification_experiments import criterion


class Learner():

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def save(self, label):
        if not os.path.exists(os.path.join("checkpoints")):
            os.makedirs(os.path.join("checkpoints"))
        torch.save(self.model.state_dict(), os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label))

    def load(self, label):
        self.model.load_state_dict(torch.load(os.path.join("checkpoints", self.model.__class__.__name__ + "_" + label), weights_only=True))
    

class RegressionLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []

        for epoch in range(epochs):

            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in dataloader:
                data = data.to(device)
                targets = targets.to(device)

                # TODO: Implement the batch update step.
                self.optimizer.zero_grad()                                                      # Reset all the gradients to zero (else new and ol gradients will be summed up)
                predictions = self.model(data)                                                  # Run the model and store its predictions
                loss = self.criterion(predictions, targets)                                     # Calculate the loss and store it as scalar
                loss.backward()                                                                 # Calculate the gradients
                self.optimizer.step()                                                           # Apply the gradients to the weights and the biases

                train_loss += loss.item() * len(data)                                           # Multiplies the loss-value with the amount of items it the batch

            train_loss = train_loss / len(dataloader.dataset)                                   # Divide the loss by the amount of total items during the run, giving us our mean-value for the epoch
            train_losses.append(train_loss)                                                     # Add the loss from this epoch to a list which contains the loss of all the epochs

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.6f}")   

            # If the current training-loss is smaller or equal to
            # the smallest loss in the list of training-losses, mark it as "leader".
            if train_loss <= min(train_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return train_losses

    @torch.no_grad()
    def test(self, dataset: Dataset, device: torch.device):

        self.model.to(device)                                   # Move the model to CPU / GPU
        self.model.eval()                                       # set model in evaluation mode
        test_loss = 0                                           # Initialize the test-loss variable

        # TODO: Compute the model's RMSE on the test dataset.
        criterion = torch.nn.MSELoss()                          # Create the MSE loss function
        for data, targets in DataLoader(dataset):
            data = data.to(device)                              # Move the data to the device CPU/GPU
            targets = targets.to(device)                        # Move the targets to the device CPU/GPU

            predictions = self.model(data)                      # Run the model and us e the recently moved data
            loss = criterion(predictions, targets)              # Calculate the mean squared error

            test_loss += loss.item() * len(data)                # Multiply the batch-loss with the amount of data in each batch

        test_loss = (test_loss / len(dataset)) ** 0.5           # Calculate the RMSE over the entire dataset
        return test_loss

    @torch.no_grad()
    def predict(self, dataset, device):
        self.model.to(device)
        self.model.eval()
        data = dataset[:][0]
        data = data.to(device)
        results = self.model(data)
        return results


class ClassificationLearner(Learner):

    def __init__(self, model: Module, criterion: Module, optimizer: Optimizer):
        super().__init__(model, criterion, optimizer)

    def train(self, train_dataloader: DataLoader, valid_dataloader: DataLoader, epochs: int, device: torch.device):

        self.model.to(device)

        train_losses = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(epochs):
            
            self.model.train()  # set model in training mode

            train_loss = 0

            for data, targets in train_dataloader:
                data = data.to(device)
                targets = targets.to(device)
                
                # TODO: Implement the batch update step.
                loss = torch.tensor([0])
                self.optimizer.zero_grad()

                predictions = self.model(data)

                loss = self.criterion(predictions, targets)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * len(data)

            train_loss = train_loss / len(train_dataloader.dataset)

            self.model.eval()  # set model in evaluation mode

            # evaluate the model on the validation dataset
            valid_loss, valid_accuracy = self._evaluate(valid_dataloader, device)    

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            valid_accuracies.append(valid_accuracy)

            print(f"train epoch [{epoch + 1}/{epochs}]:\ttrain loss = {train_loss:.4f}\tvalid loss = {valid_loss:.4f}\tvalid accuracy = {100 * valid_accuracy:.2f}%")   

            if valid_loss <= min(valid_losses):
                self.save("leader")

        print()
        
        self.save("final")

        return (train_losses, valid_losses, valid_accuracies)

    def test(self, test_dataloader: DataLoader, device: torch.device):

        self.model.to(device)

        self.model.eval()  # set model in evaluation mode

        # evaluate the model on the test dataset
        test_loss, test_accuracy = self._evaluate(test_dataloader, device)

        return (test_loss, test_accuracy)

    @torch.no_grad()
    def _evaluate(self, dataloader, device):
        eval_loss = 0
        eval_accuracy = 0
        correct = 0

        # TODO: Compute the model's loss and accuracy over the dataset.
        self.model.eval() 

        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            predictions = self.model(data)
            loss = self.criterion(predictions, targets)

            eval_loss += loss.item() * len(data)
            correct += (predictions.argmax(dim=1) == targets).sum().item()

        n = len(dataloader.dataset)
        eval_loss /= n
        eval_accuracy = correct / n

        return (eval_loss, eval_accuracy)
