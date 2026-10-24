import torch

from torch import Tensor, Size

class LinearModel():

    def __init__(self, input_shape: Size, output_shape: Size):
        
        if len(output_shape) != 1:
            raise ValueError("the output shape must be one-dimensional")

        # This is what you call when you call the weights function
        self.weights = self.init_weights(input_shape, output_shape)

    def init_weights(self, input_shape: Size, output_shape: Size) -> Tensor:
        """Create the model's weight tensor."""

        weight_shape = (input_shape + output_shape)                                             # We add the new dimensions form the output-shape to the dimension from teh input shape, giving us the shape of our weight
        weight_tens_rand = torch.randn(weight_shape, dtype=torch.float32)                       # Create a new tensor with the calculated weight-shape and initialize it with random values given the standard distribution
        weight_multiplier = 0.1                                                                 # This is the multiplier to make our generated random weights smaller, keeping them more to the center (0)
        weight_tens_adj = torch.mul(weight_tens_rand, weight_multiplier)                        # Here we multiply our tensor with random values with a constant, this way we can bring our values closer to "0" without losing the direction and absolute distance between weights

        return weight_tens_adj                                                                  # Return the adjusted weights

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""
        # TODO: Calculate the unrestricted linear output without flattening the input.
        raise NotImplementedError()

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Measure the discrepancy between a model output and its target."""
        # TODO: Calculate the mean squared error.
        raise NotImplementedError()

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        raise NotImplementedError()

    def update_weights(self, gradients: Tensor, learning_rate: float) -> Tensor:
        """Update the model parameters using the supplied gradients."""
        # TODO: Perform one gradient-descent step and return the updated weights.
        raise NotImplementedError()

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
        raise NotImplementedError()

    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE, sigmoid, and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        raise NotImplementedError()

