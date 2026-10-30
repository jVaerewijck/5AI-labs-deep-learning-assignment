import torch

from torch import Tensor, Size
from torch._C import dtype


class LinearModel():

    def __init__(self, input_shape: Size, output_shape: Size):
        
        if len(output_shape) != 1:
            raise ValueError("the output shape must be one-dimensional")

        # Initialize the model weights and store the resulting tensor
        self.weights = self.init_weights(input_shape, output_shape)
        self.input_shape = Size(input_shape)
        self.output_shape = Size(output_shape)

    def init_weights(self, input_shape: Size, output_shape: Size) -> Tensor:
        """Create the model's weight tensor."""

        weight_shape = (input_shape + output_shape)                                             # We add the new dimensions form the output-shape to the dimension from teh input shape, giving us the shape of our weight
        weight_tens_rand = torch.randn(weight_shape, dtype=torch.float32)                       # Create a new tensor with the calculated weight-shape and initialize it with random values given the standard distribution
        weight_multiplier = 0.1                                                                 # This is the multiplier to make our generated random weights smaller, keeping them more to the center (0)
        weight_tens_adj = torch.mul(weight_tens_rand, weight_multiplier)                        # Here we multiply our tensor with random values with a constant, this way we can bring our values closer to "0" without losing the direction and absolute distance between weights

        return weight_tens_adj   
    def init_weights(self, input_shape: Size, output_shape: Size) -> Tensor:
        """Create the model's weight tensor."""
        # TODO: Initialize one small random weight for every input-output connection.
        return 0.01 * torch.randn(*input_shape, *output_shape)                                                               # Return the adjusted weights

    def calculate_output(self, input: Tensor) -> Tensor:
        """Apply the model to an input tensor."""

        input_adj = torch.unsqueeze(input, 3)                                               # add a 4th dimension to the input-data, this makes it compatible with the 4D-tensor from our weights
        mul_in_w = torch.mul(input_adj, self.weights)                                           # element wise multiplication on the un-squeezed input and the weights
        sum_out = torch.sum(mul_in_w, dim=(0,1,2), dtype=torch.float32)                         # Sum up all the dimensions until only the output dimension is left, also explicitly make the tensor a 32-bit float

        return sum_out                                                                          # Return the 32-bit tensor with shape (2)

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

        ind_sq = (output - target) ** 2                                                         # We first subtract our target value from our input value, then we execute element wise squared operations to each datapoint in our tensor
        mse = torch.mean(ind_sq,dtype=torch.float32)                                            # Then we take the mean-value of the previous tensor and, we get a 0D-tensor wit only 1 value which is our MSE

        return mse                                                                              # Then we return the tensor (IMPORTANT => it's still a tensor!)

    def calculate_loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Measure the discrepancy between a model output and its target."""
        # TODO: Calculate the mean squared error.
        return ((output - target) ** 2).mean()
    
    def calculate_gradients(self, input: Tensor, output: Tensor, target: Tensor) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""

        predict_error = (output - target)                                                       # Output and target tensor ar both 1D-tensors at this moment!

        print("Output:", output, "| Shape:", output.shape)                                      # Some visual aid
        print("Target:", target, "| Shape:", target.shape)                                      #   ""
        print("Predict error:", predict_error, "| Shape:", predict_error.shape)                 #   ""

        in_unsq = torch.unsqueeze(input, dim=3)                                                 # Here we unsqueeze the input tensor which is 3D -> 4D, where the default value will be 1 in the 4th dimension
        in_mul_loss = torch.mul(in_unsq, predict_error)                                         # Here we multiply our 3D-tensor with our 1D-tensor, so we basically get (in this case) 2 3D-tensors within a 4D-tensor.

        return in_mul_loss                                                                      # Here we return our gradients in 4D-tensor format
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

        adjusted_gradients = torch.mul(gradients, learning_rate)
        adjusted_weights = torch.sub(self.weights, adjusted_gradients)

        self.weights = adjusted_weights

        return adjusted_weights

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

    def calculate_gradients(
        self, input: Tensor, output: Tensor, target: Tensor
    ) -> Tensor:
        """Calculate the gradient of the loss with respect to every model weight."""
        # TODO: Use the MSE, sigmoid, and linear-layer derivatives to calculate d(loss)/d(weights) manually.
        error = output - target
        d_loss = 2 * error / error.numel()
        d_logits = d_loss * output * (1 - output)

        len_in = len(self.input_shape)
        # batch dimensions, otherwise plain outer product
        batch_dims = list(range(input.ndim - len_in))

        return torch.tensordot(input, d_logits, dims=(batch_dims, batch_dims))

