import torch

from torch import Tensor
from torch.utils.data import Dataset


def get_logo_as_rgb_image_tensor() -> Tensor:

    logo = torch.tensor([
        [
            [0.9333, 0.1569, 0.2510],
            [0.9333, 0.1569, 0.2510],
            [0.9333, 0.1569, 0.2510],
            [0.9333, 0.1569, 0.2510],
        ],
        [
            [1.0, 1.0, 1.0],
            [0.9333, 0.1569, 0.2510],
            [1.0, 1.0, 1.0],
            [0.9333, 0.1569, 0.2510],
        ],
        [
            [1.0, 1.0, 1.0],
            [0.9333, 0.1569, 0.2510],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        [
            [0.9333, 0.1569, 0.2510],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.9333, 0.1569, 0.2510],
        ],
    ], dtype=torch.float32)

    return logo



class HousesDataset(Dataset):
    def __init__(self, dataset_size: int):
        self.dataset_size = dataset_size

        self.min_house_size = 30
        self.max_house_size = 70
        self.noise_house_price = 50000

        house_sizes = torch.randint(self.min_house_size, self.max_house_size, (self.dataset_size,))
        house_prices = 5000 * house_sizes + 100000 + torch.randn(self.dataset_size) * self.noise_house_price

        self.data = house_sizes.float().unsqueeze(1)
        self.targets = house_prices.float().unsqueeze(1)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx, :], self.targets[idx, :]