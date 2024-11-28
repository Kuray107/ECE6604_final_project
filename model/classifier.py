import torch
import torch.nn as nn
import einops

class LinearClassifier(nn.Module):
    """
    A fully connected neural network for binary classification of input symbols 
    into bits using a sequence of linear layers.

    Args:
        symbol_length (int, optional): The length of each input symbol. Defaults to 32.
        bit_length (int, optional): The length of the output bit sequence. Defaults to 29.
        node_list (list of int, optional): A list defining the number of nodes in 
            each hidden layer. Defaults to [32, 64, 128].
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.0.
    """
    def __init__(
        self, 
        symbol_length=32,
        bit_length=29,
        node_list=[32, 64, 128],
        dropout=0.0, 
    ):
        super().__init__()
        self.symbol_length = symbol_length
        self.bit_length = bit_length
        self.node_list = [symbol_length] + node_list  # Include input symbol length

        # Define the layers
        self.layers = nn.ModuleList()
        
        # Add hidden layers
        for idx in range(len(node_list)):
            self.layers.append(
                nn.Sequential(
                    nn.Linear(
                        in_features=self.node_list[idx],
                        out_features=self.node_list[idx + 1]
                    ),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Add the output layer
        self.layers.append(
            nn.Sequential(
                nn.Linear(
                    in_features=self.node_list[-1],
                    out_features=bit_length
                ),
                nn.Sigmoid(),  # Outputs probabilities for binary classification
            )
        )

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, bit_length * sequence_length).
        """
        assert inputs.ndim == 2, "Input tensor must be 2-dimensional (batch_size, sequence_length)."

        # Rearrange inputs to match expected dimensions
        outputs = einops.rearrange(inputs, "B (D T) -> B T D", D=self.symbol_length)

        # Pass through each layer
        for layer in self.layers:
            outputs = layer(outputs)

        # Rearrange back to flattened output
        outputs = einops.rearrange(outputs, "B T D -> B (D T)")

        return outputs
