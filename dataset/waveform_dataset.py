import os
import random
import torch
import pyldpc
import librosa
import numpy as np
import pandas as pd
from torch.utils import data

class Dataset(data.Dataset):
    """
    Custom Dataset for LDPC-based encoding of speech data.

    Args:
        csv_file (str): Path to CSV file containing paths to clean wav files.
        sample_rate (int): Sample rate for loading wav files.
        is_training (bool): Whether the dataset is used for training.
        LDPC_cfg (dict): Configuration for LDPC encoding.
        snr_values (list): List of SNR values for testing scenarios.
        num_blocks (int): Number of LDPC blocks per example.
        limit (int, optional): Limit on the number of examples.
        offset (int): Offset to skip rows in the CSV file.
        seed (int): Random seed for reproducibility.
    """
    def __init__(
        self,
        csv_file,
        sample_rate,
        is_training,
        LDPC_cfg: dict,
        snr_values=[7],
        num_blocks=100,
        limit=None,
        offset=0,
        seed=1234
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.is_training = is_training
        self.snr_values = snr_values
        self.num_blocks = num_blocks
        self.seed = seed

        # Load dataset using pandas
        df = pd.read_csv(csv_file, skiprows=range(offset))
        if limit:
            df = df.head(limit)
        self.dataset = df
        self.length = len(self.dataset)

        # Generate LDPC matrices
        self.H, self.G = pyldpc.make_ldpc(
            n_code=LDPC_cfg["n_code"],
            d_v=LDPC_cfg["d_v"],
            d_c=LDPC_cfg["d_c"],
            seed=seed,
            systematic=True,
            sparse=True
        )
        self.n, self.k = self.G.shape
        
        if not self.is_training:
            self.rng = random.Random()
            self.rng.seed(self.seed)

    def __len__(self):
        return self.length

    def __getitem__(self, item: int):
        """
        Get an example for training or testing.

        Args:
            item (int): Index of the dataset.

        Returns:
            Tuple: y_t (noisy signal), x_t (encoded signal), 
                   bit_sequence (original bit sequence), name (file name).
        """
        row = self.dataset.iloc[item]
        wav_path = row["clean_path"]
        name = os.path.splitext(os.path.basename(wav_path))[0]

        wav, _ = librosa.load(wav_path, sr=self.sample_rate)
        bit_sequence = self.wav_to_bit_sequence(wav)

        # Prepare blocks for encoding
        bit_sequence = bit_sequence[:self.num_blocks * self.k]
        bit_matrix = np.reshape(bit_sequence, (self.k, self.num_blocks))

        clean = pyldpc.encode(self.G, bit_matrix, snr=50, seed=self.seed)
        clean = np.reshape(clean, (self.n * self.num_blocks))

        if self.is_training:
            snr = random.choice(self.snr_values)
            noisy = self.add_noise(clean, snr=snr)
        else:
            # We need to fix the testing and validation noise for fair comparison
            snr = self.rng.choice(self.snr_values)
            noisy = pyldpc.encode(self.G, bit_matrix, snr=snr, seed=self.seed)
            noisy = np.reshape(noisy, (self.n * self.num_blocks))

        return noisy.astype(np.float32), clean.astype(np.float32), bit_sequence.astype(np.float32), name

    def ldpc_decode(self, y: np.array):
        """
        Decode an LDPC-encoded signal.

        Args:
            y_t (np.array): Noisy encoded signal.

        Returns:
            np.array: Decoded bit sequence.
        """
        D = pyldpc.decode(self.H, y, snr=20)
        bit_preds = [pyldpc.get_message(self.G, D[:, i]) for i in range(D.shape[1])]
        return np.stack(bit_preds, axis=1)

    def wav_to_bit_sequence(self, wav, quantization_bits=16):
        """
        Convert a waveform to a bit sequence.

        Args:
            wav (np.array): Waveform data.
            quantization_bits (int): Number of bits for quantization.

        Returns:
            np.array: Bit sequence.
        """
        if wav.dtype == np.float32:
            data = wav
        elif wav.dtype in [np.int16, np.int8]:
            max_val = 2**(quantization_bits - 1) - 1
            data = wav.astype(np.float32) / max_val
        else:
            raise ValueError("Unsupported WAV format")

        data = (data + 1) / 2  # Scale to [0, 1]
        quantized_data = np.round(data * (2**quantization_bits - 1)).astype(np.uint16)
        bit_sequence = np.unpackbits(quantized_data.view(np.uint8), axis=0)
        return bit_sequence

    def add_noise(self, signal, snr=7):
        """
        Add white Gaussian noise to a signal.

        Args:
            signal (np.array): Input signal.
            snr (float): Signal-to-noise ratio in dB.

        Returns:
            np.array: Noisy signal.
        """
        signal_power = np.mean(signal**2)
        noise_power = signal_power / (10**(snr / 10))
        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for DataLoader.

        Args:
            batch (list): List of dataset examples.

        Returns:
            Tuple: Stacked tensors for y_t, x_t, and file names.
        """
        noisy_list, clean_list, bit_list, names = [], [], [], []
        for noisy, clean, bit, name in batch:
            noisy_list.append(torch.FloatTensor(noisy))
            clean_list.append(torch.FloatTensor(clean))
            bit_list.append(torch.FloatTensor(bit))
            names.append(name)

        return (
            torch.stack(noisy_list),
            torch.stack(clean_list),
            torch.stack(bit_list),
            names
        )