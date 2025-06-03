import torch
import torch.nn as nn
import torch.optim as optim
import os
import json # Import json for saving class names

from data_loader import get_paddy_data_loaders
# Choose one of the model implementations:
# Make sure the one you want to use is UNCOMMENTED
# from cnn_model import PaddyCNNModel # For training from scratch
from cnn_model import PaddyTransferLearningModel # For training with transfer learning (recommended)

# --- Configuration ---
DATA_DIR = r'data_dir' # <--- IMPORTANT: Change this to your actual dataset path
BATCH_SIZE = 32
NUM_EPOCHS = 1 # Set to 5 epochs as requested
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_SEED = 42
MODEL_SAVE_PATH = 'paddy_disease_model_weights.pth' # Name for your saved model weights file
CLASS_NAMES_SAVE_PATH = 'paddy_class_names.json' # Path to save class names JSON file

def main():
    # 1. Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please ensure you have downloaded the dataset and provided the correct path.")
        return # Exit the function if path is incorrect

    # 3. Load data
    print("\nLoading and preprocessing data...")
    train_loader, val_loader, test_loader, class_names = get_paddy_data_loaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        random_seed=RANDOM_SEED
    )
    num_classes = len(class_names)
    print("Data loaded successfully.")
    print(f"Number of disease classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # --- IMPORTANT FIX: Save class_names here, inside the main function ---
    try:
        with open(CLASS_NAMES_SAVE_PATH, 'w') as f:
            json.dump(class_names, f)
        print(f"Class names saved to {CLASS_NAMES_SAVE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save class names to {CLASS_NAMES_SAVE_PATH}: {e}")
    # --- END FIX ---

    # 4. Initialize model
    # Choose either PaddyCNNModel (from scratch) or PaddyTransferLearningModel
    # model = PaddyCNNModel(num_classes=num_classes).to(device) # From scratch
    model = PaddyTransferLearningModel(num_classes=num_classes, model_name='resnet18', pretrained=True).to(device) # Transfer Learning (recommended)

    print("\nModel Architecture:")
    print(model)

    # 5. Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Train the model
    print("\nStarting training...")
    history = model.train_model(train_loader, val_loader, optimizer, criterion, NUM_EPOCHS, device)

    # 7. Plot training history
    model.plot_history(history)

    # 8. Evaluate on the test set
    print("\nEvaluating on test set...")
    model.eval() # Set model to evaluation mode
    correct_test = 0
    total_test = 0
    with torch.no_grad(): # Disable gradient calculation for inference
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the max log-probability
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # 9. Save the trained model weights
    try:
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Trained model weights saved to {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Warning: Could not save model weights to {MODEL_SAVE_PATH}: {e}")


if __name__ == '__main__':
    # This guard is crucial for multiprocessing on Windows
    main()