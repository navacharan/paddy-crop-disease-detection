import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models # Import pre-trained models
from tqdm import tqdm

class PaddyTransferLearningModel(nn.Module):
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        super(PaddyTransferLearningModel, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(f"Model {model_name} not supported.")

        # Freeze all parameters in the pre-trained model (optional, good for initial training)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # Replace the final classification layer
        # Get the number of features in the last layer
        if model_name.startswith('resnet'):
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
        elif model_name == 'mobilenet_v2':
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, val_loader, optimizer, criterion, epochs, device):
        # This function is identical to the one in PaddyCNNModel
        # You can copy-paste it directly or put it in a utility file if desired.
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training"):
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            epoch_train_loss = running_loss / len(train_loader.dataset)
            epoch_train_acc = correct_train / total_train

            self.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation"):
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            epoch_val_loss = val_loss / len(val_loader.dataset)
            epoch_val_acc = correct_val / total_val

            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
            print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)

        print("Training finished!")
        return history

    def plot_history(self, history):
        # This function is identical to the one in PaddyCNNModel
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

# Example usage with transfer learning (in main.py)
if __name__ == '__main__':
    from data_loader import get_paddy_data_loaders # Import your data loader

    # Assume dummy_paddy_data_for_model_test is already created from previous run
    dummy_data_dir = 'dummy_paddy_data_for_model_test'
    if not os.path.exists(dummy_data_dir):
        print(f"Creating dummy data for {dummy_data_dir}...")
        os.makedirs(os.path.join(dummy_data_dir, 'Healthy'), exist_ok=True)
        os.makedirs(os.path.join(dummy_data_dir, 'Brown_Spot'), exist_ok=True)
        os.makedirs(os.path.join(dummy_data_dir, 'Bacterial_Blight'), exist_ok=True)
        from PIL import Image
        for i in range(20):
            Image.new('RGB', (256, 256), color = (i*10, i*5, i*2)).save(os.path.join(dummy_data_dir, 'Healthy', f'healthy_{i}.png'))
            Image.new('RGB', (256, 256), color = (i*2, i*10, i*5)).save(os.path.join(dummy_data_dir, 'Brown_Spot', f'brown_spot_{i}.png'))
            Image.new('RGB', (256, 256), color = (i*5, i*2, i*10)).save(os.path.join(dummy_data_dir, 'Bacterial_Blight', f'blight_{i}.png'))


    train_loader, val_loader, test_loader, class_names = get_paddy_data_loaders(
        data_dir=dummy_data_dir,
        batch_size=16,
        val_split=0.2,
        test_split=0.2
    )

    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use a pre-trained ResNet-18 model
    model = PaddyTransferLearningModel(num_classes=num_classes, model_name='resnet18', pretrained=True).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) # You might use a smaller LR for fine-tuning

    history = model.train_model(train_loader, val_loader, optimizer, criterion, epochs=5, device=device)
    model.plot_history(history)

    # Evaluate on the test set
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct_test / total_test:.2f}%")

    # Save the trained model weights
    torch.save(model.state_dict(), 'paddy_resnet18_weights.pth')
    print("Trained ResNet18 model weights saved to paddy_resnet18_weights.pth")

    # Clean up dummy data
    import shutil
    shutil.rmtree(dummy_data_dir)