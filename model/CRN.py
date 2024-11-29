import torch
import torch.nn as nn
import einops

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

class CRN(nn.Module):
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
        symbol_length=32,
        channels_list= [64, 128],
        rnn_units=64,
        rnn_layers=1,
        final_conv_channels=64,
        kernel_size=5,
        dropout=0.0, 
    ):
        super().__init__()

        # Prepare the list of channel sizes for encoder and decoder
        enc_channels_list = [symbol_length] + channels_list
        self.num_layers = len(enc_channels_list) - 1
        self.symbol_length = symbol_length

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

        rnn_input_dim = channels_list[-1]

        self.lstm_prenet = nn.Linear(rnn_input_dim, rnn_units)
        self.lstm = nn.LSTM(
            input_size=rnn_units,
            hidden_size=rnn_units,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_postnet = nn.Linear(rnn_units * 2, rnn_input_dim)

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
            nn.Conv1d(final_conv_channels, symbol_length, kernel_size=1),
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
        outputs = einops.rearrange(inputs, "B (D T) -> B D T", D=self.symbol_length)

        # Pass through encoder layers
        for enc_layer in self.encoder:
            outputs = enc_layer(outputs)

        outputs = einops.rearrange(outputs, "B D T -> B T D")
        outputs = self.lstm_prenet(outputs)
        outputs, _ = self.lstm(outputs)
        outputs = self.lstm_postnet(outputs)
        outputs = einops.rearrange(outputs, "B T D -> B D T")

        # Pass through decoder layers
        for dec_layer in self.decoder:
            outputs = dec_layer(outputs) 
        
        outputs = self.norm(outputs)
        outputs = self.final_conv(outputs)

        # Rearrange back to flattened output
        outputs = einops.rearrange(outputs, "B D T -> B (D T)")

        return outputs
