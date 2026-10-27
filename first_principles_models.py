import torch

from torch import Tensor, Size


class LinearModel():

    def __init__(self, input_shape: Size, output_shape: Size):
        
        if len(output_shape) != 1:
            raise ValueError("the output shape must be one-dimensional")
        
        self.weights = self.init_weights(input_shape, output_shape)
        self.input_shape = Size(input_shape)
        self.output_shape = Size(output_shape)

    def init_weights(self, input_shape: Size, output_shape: Size) -> Tensor:
        """Create the model's weight tensor."""
        # TODO: Initialize one small random weight for every input-output connection.
        return 0.01 * torch.randn(*input_shape, *output_shape)

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""
        # TODO: Calculate the unrestricted linear output without flattening the input.
        len_in = len(self.input_shape)
        return torch.tensordot(
            input,
            self.weights,
            dims=(list(range(input.ndim - len_in, input.ndim)), list(range(len_in))),
        )

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Measure the discrepancy between a model output and its target."""
        # TODO: Calculate the mean squared error.
        return ((output - target) ** 2).mean()

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        error = output - target
        d_loss = 2 * error / error.numel()

        len_in = len(self.input_shape)
        # batch dimensions, otherwise plain outer product
        batch_dims = list(range(input.ndim - len_in))

        return torch.tensordot(input, d_loss, dims=(batch_dims, batch_dims))


    def update_weights(self, gradients: Tensor, learning_rate: float) -> Tensor:
        """Update the model parameters using the supplied gradients."""
        # TODO: Perform one gradient-descent step and return the updated weights.
        self.weights = self.weights - learning_rate * gradients
        return self.weights

    def train_step(self, input: Tensor, target: Tensor, learning_rate: float) -> float:
        """Run one optimization step for an input-target pair."""
        output = self.calculate_output(input)
        loss = self.calculate_loss(output, target)
        gradients = self.calculate_gradients(input, output, target)
        self.update_weights(gradients, learning_rate)

        return loss.item()

    def fit(self, input: Tensor, target: Tensor, steps: int, learning_rate: float) -> list[float]:
        """Optimize the model over multiple passes through an input-target pair."""
        loss_history = []

        for step in range(steps):
            loss = self.train_step(input, target, learning_rate)
            loss_history.append(loss)
            print(f"[{self.__class__.__name__}] fit step {step + 1}/{steps}: loss = {loss}")

        return loss_history


class LinearSigmoidModel(LinearModel):

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""
        # TODO: Calculate the linear logits and apply the sigmoid activation.
        len_in = len(self.input_shape)
        logits =  torch.tensordot(
            input,
            self.weights,
            dims=(list(range(input.ndim - len_in, input.ndim)), list(range(len_in))),
        )
        return torch.sigmoid(logits)

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE, sigmoid, and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        error = output - target
        d_loss = 2 * error / error.numel()
        d_logits = d_loss * output * (1 - output)

        len_in = len(self.input_shape)
        # batch dimensions, otherwise plain outer product
        batch_dims = list(range(input.ndim - len_in))

        return torch.tensordot(input, d_logits, dims=(batch_dims, batch_dims))

