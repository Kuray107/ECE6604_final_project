import ipdb
import unittest
import os
import pandas as pd
import numpy as np
import soundfile as sf


from torch.utils.data import DataLoader
from waveform_dataset import Dataset, collate_fn  # Assuming your dataset code is in `dataset.py`

class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary CSV file with dummy data
        cls.temp_csv = "temp_dataset.csv"
        data = {"clean_path": ["audio1.wav", "audio2.wav"]}
        pd.DataFrame(data).to_csv(cls.temp_csv, index=False)

        # Create dummy audio files
        cls.sample_rate = 16000
        cls.duration = 1  # seconds
        for i in range(1, 3):
            wav_data = (np.random.rand(cls.sample_rate * cls.duration) * 2 - 1).astype(np.float32)
            sf.write(f"audio{i}.wav", wav_data, samplerate=cls.sample_rate)

        # LDPC configuration
        cls.LDPC_cfg = {
            "n_code": 32,
            "d_v": 2,
            "d_c": 16,
        }

    @classmethod
    def tearDownClass(cls):
        # Clean up temporary files
        os.remove(cls.temp_csv)
        for i in range(1, 3):
            os.remove(f"audio{i}.wav")

    def test_dataset_initialization(self):
        # Test dataset initialization
        dataset = Dataset(
            csv_file=self.temp_csv,
            sample_rate=self.sample_rate,
            is_training=True,
            LDPC_cfg=self.LDPC_cfg,
            snr_values=[7],
            num_of_blocks=10,
        )
        self.assertEqual(len(dataset), 2)  # We created 2 rows in the CSV

    def test_getitem_training_mode(self):
        # Test __getitem__ in training mode
        dataset = Dataset(
            csv_file=self.temp_csv,
            sample_rate=self.sample_rate,
            is_training=True,
            LDPC_cfg=self.LDPC_cfg,
            snr_values=[7],
            num_of_blocks=10,
        )
        y_t, x_t, bit_sequence, name = dataset[0]
        ipdb.set_trace()
        self.assertEqual(len(bit_sequence), dataset.num_of_blocks * dataset.k)
        self.assertEqual(len(x_t), dataset.num_of_blocks * dataset.n)
        self.assertEqual(len(y_t), len(x_t))
        self.assertIsInstance(name, str)

    def test_collate_fn(self):
        # Test the custom collate function
        dataset = Dataset(
            csv_file=self.temp_csv,
            sample_rate=self.sample_rate,
            is_training=True,
            LDPC_cfg=self.LDPC_cfg,
            snr_values=[7],
            num_of_blocks=10,
        )
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        for batch in dataloader:
            y_t, x_t, bit_sequences, names = batch
            self.assertEqual(y_t.shape[0], 2)  # Batch size
            self.assertEqual(x_t.shape[0], 2)  # Batch size
            self.assertEqual(bit_sequences.shape[0], 2)  # Batch size
            self.assertEqual(len(names), 2)

if __name__ == "__main__":
    unittest.main()
