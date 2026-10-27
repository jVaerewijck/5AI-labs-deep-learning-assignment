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
        self.layer1 = None
        self.layer2 = None
        self.layer3 = None

    def forward(self, x: torch.Tensor):
        # TODO: Implement the forward pass.
        return x
