import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# --- MOVE THIS CLASS DEFINITION HERE (OUTSIDE THE FUNCTION) ---
class TransformedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
# --- END OF MOVED CLASS ---


def get_paddy_data_loaders(data_dir, batch_size=32, val_split=0.2, test_split=0.1, random_seed=42):
    """
    Creates PyTorch DataLoaders for paddy disease images.
    ... (docstring remains the same) ...
    """

    # Define transformations for training (with data augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define transformations for validation/testing (no augmentation)
    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset (without initial transform, so it can be applied to subsets)
    base_dataset = datasets.ImageFolder(root=data_dir)

    # Get class names
    class_names = base_dataset.classes # Use base_dataset to get classes
    print(f"Detected classes: {class_names}")
    print(f"Total images found: {len(base_dataset)}")

    # Calculate split sizes
    total_size = len(base_dataset)
    test_size = int(test_split * total_size)
    val_size = int(val_split * (total_size - test_size))
    train_size = total_size - test_size - val_size

    # Ensure consistent splits
    generator = torch.Generator().manual_seed(random_seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create datasets with specific transforms using the now globally defined TransformedSubset
    train_dataset_final = TransformedSubset(train_subset, transform=train_transform)
    val_dataset_final = TransformedSubset(val_subset, transform=val_test_transform)
    test_dataset_final = TransformedSubset(test_subset, transform=val_test_transform)

    # Create DataLoaders
    # num_workers=0 can be used as a fallback if multiprocessing issues persist,
    # but it will be slower as data loading happens in the main process.
    train_loader = DataLoader(train_dataset_final, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count())
    val_loader = DataLoader(val_dataset_final, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())
    test_loader = DataLoader(test_dataset_final, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count())

    return train_loader, val_loader, test_loader, class_names


# Example usage (in main.py or for testing)
if __name__ == '__main__':
    # Create a dummy data directory for testing
    dummy_data_dir = 'dummy_paddy_data'
    os.makedirs(os.path.join(dummy_data_dir, 'Healthy'), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_dir, 'Brown_Spot'), exist_ok=True)
    os.makedirs(os.path.join(dummy_data_dir, 'Bacterial_Blight'), exist_ok=True)

    # Create dummy images (requires Pillow)
    from PIL import Image
    for i in range(20):
        img = Image.new('RGB', (256, 256), color = (i*10, i*5, i*2))
        img.save(os.path.join(dummy_data_dir, 'Healthy', f'healthy_{i}.png'))
        img.save(os.path.join(dummy_data_dir, 'Brown_Spot', f'brown_spot_{i}.png'))
        img.save(os.path.join(dummy_data_dir, 'Bacterial_Blight', f'blight_{i}.png'))

    train_loader, val_loader, test_loader, class_names = get_paddy_data_loaders(
        data_dir=dummy_data_dir,
        batch_size=16,
        val_split=0.2,
        test_split=0.2
    )

    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Class names: {class_names}")

    # Clean up dummy data
    import shutil
    shutil.rmtree(dummy_data_dir)