import torch


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define the linear regression layer.
        self.linear_layer = torch.nn.Linear(1,1)

    def forward(self, x: torch.Tensor):
        # TODO: Implement the forward pass.
        return self.linear_layer(x)


class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Define the three classification layers.
        self.layer1 = torch.nn.Linear(784, 128)
        self.layer2 = torch.nn.Linear(128, 64)
        self.layer3 = torch.nn.Linear(64, 10)

    def forward(self, x: torch.Tensor):
        # TODO: Implement the forward pass.
        relu = torch.nn.ReLU()
        x = torch.flatten(x, start_dim=1)
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        return self.layer3(x)
