import os
import unittest

import config
from data.download_dataset import download_sample_dataset
from env.real_data_loader import RealDataLoader


class TestDataLoader(unittest.TestCase):
    def test_loader_returns_8760_records(self):
        if not os.path.exists(config.REAL_DATASET_PATH):
            download_sample_dataset()

        loader = RealDataLoader(config.REAL_DATASET_PATH)
        self.assertIsNotNone(loader.data)
        self.assertEqual(len(loader.data), 8760)


if __name__ == "__main__":
    unittest.main()
