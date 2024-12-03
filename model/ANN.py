import torch
import torch.nn as nn

from model.DSSM_modules.components import TransposedLN

class EncoderLayer(nn.Module):
    """
    Defines a single encoder layer consisting of a 1D convolution, 
    max pooling, and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the convolution.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass through the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after convolution, pooling, and dropout.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class DecoderLayer(nn.Module):
    """
    Defines a single decoder layer consisting of a 1D convolution,
    upsampling, and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the convolution.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )
        self.norm = nn.BatchNorm1d(num_features=out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass through the decoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            torch.Tensor: Output tensor after convolution, upsampling, and dropout.
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

class ANN(nn.Module):
    """
    Implements an autoencoder-like network with a series of encoder
    and decoder layers.

    Args:
        in_channels (int, optional): Number of input channels. Defaults to 1.
        channels_list (list of int, optional): List specifying the number of 
            channels in each encoder/decoder layer. Defaults to [32, 64, 128].
        kernel_size (int, optional): Kernel size for the convolutional layers. Defaults to 5.
        dropout (float, optional): Dropout probability. Defaults to 0.0.
    """
    def __init__(
        self, 
        in_channels=1,
        channels_list=[64, 128, 128],
        final_conv_channels=128,
        kernel_size=5,
        dropout=0.0, 
    ):
        super().__init__()

        # Prepare the list of channel sizes for encoder and decoder
        enc_channels_list = [in_channels] + channels_list
        self.num_layers = len(enc_channels_list) - 1

        # Define encoder layers
        self.encoder = nn.ModuleList()
        for idx in range(self.num_layers):
            self.encoder.append(
                EncoderLayer(
                    in_channels=enc_channels_list[idx],
                    out_channels=enc_channels_list[idx + 1],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )

        # Define decoder layers
        self.decoder = nn.ModuleList()
        dec_channels_list = [final_conv_channels] + channels_list
        for idx in range(self.num_layers - 1, -1, -1):
            self.decoder.append(
                DecoderLayer(
                    in_channels=dec_channels_list[idx + 1],
                    out_channels=dec_channels_list[idx],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )

        self.norm = TransposedLN(d=dec_channels_list[0])
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(final_conv_channels, in_channels, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Forward pass through the autoencoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length).
        """
        assert inputs.ndim == 2, "Input tensor must be 2-dimensional (batch_size, sequence_length)."

        # Add a channel dimension
        x = inputs.unsqueeze(1)

        # Pass through encoder layers
        for enc_layer in self.encoder:
            x = enc_layer(x)

        # Pass through decoder layers
        for dec_layer in self.decoder:
            x = dec_layer(x) 
        

        x = self.norm(x)
        x = self.final_conv(x)


        # Remove the channel dimension
        return x.squeeze(1)
