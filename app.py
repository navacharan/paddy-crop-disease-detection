import os
import streamlit as st
st.set_page_config(layout="wide")
import torch
from torchvision import transforms
from PIL import Image
import io
import base64
import json
import torch.nn.functional as F

# Import your CNN model class.
# Ensure this matches the model you trained and saved weights for.
from cnn_model import PaddyTransferLearningModel # Assuming you're using this one
# from cnn_model import PaddyCNNModel # Uncomment if you're using the CNN from scratch


# --- Configuration ---
# Path to your saved model weights
# Make sure this matches the file name from your training script (main.py)
MODEL_PATH = 'paddy_disease_model_weights.pth'
# Path to the JSON file containing your class names
CLASS_NAMES_PATH = 'paddy_class_names.json'

# Determine the device (CPU or GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Class Names ---
class_labels = {}
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        loaded_class_names = json.load(f)
    # Create a dictionary mapping index to class name
    # We replace underscores with spaces for cleaner display, if your original names had them.
    # Given your provided list ["Bacterial leaf blight", "Brown spot", "Leaf smut"],
    # they already have spaces, so replace() might not do anything but it's safe to keep.
    class_labels = {i: name.replace('_', ' ') for i, name in enumerate(loaded_class_names)}
    print(f"Loaded class labels: {class_labels}")
except FileNotFoundError:
    print(f"Error: Class names file not found at {CLASS_NAMES_PATH}.")
    print("Please ensure you have trained the model and saved the class names (e.g., by running main.py).")
    exit() # Exit if class names not found
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

# --- DISEASE PRESCRIPTIONS ---
# IMPORTANT: Customize these based on real agricultural advice for your region and specific diseases.
# Ensure the keys here EXACTLY MATCH the names your model predicts and are in class_labels.
DISEASE_PRESCRIPTIONS = {
    'Healthy': "The crop appears healthy. Continue with regular fertilization and watering practices. Monitor for any early signs of disease.",
    'Bacterial leaf blight': "For Bacterial Leaf Blight: Use resistant rice varieties. Avoid excessive nitrogen fertilizer. Apply bactericides like copper-based compounds (e.g., Copper Oxychloride). Manage irrigation water effectively to reduce humidity. Remove and destroy severely infected plant debris.",
    'Brown spot': "For Brown Spot: Apply fungicides containing Carbendazim, Mancozeb, or Propiconazole. Ensure balanced fertilization, particularly adequate potassium and silicon. Improve field drainage and aeration. Practice crop rotation and remove infected plant debris.",
    'Leaf smut': "For Leaf Smut: Implement fungicide seed treatment (e.g., Carbendazim or Thiram) before planting. Avoid excessive nitrogen application, especially during panicle initiation. Remove smutted panicles promptly before harvesting to prevent spore dissemination. Practice good field sanitation."
}


# --- Load the trained model ---
model = None
try:
    # Instantiate your model.
    # num_classes should match the number of classes in your dataset (len of class_labels).
    # 'resnet18' is used here as an example; change if you used a different backbone.
    model = PaddyTransferLearningModel(num_classes=len(class_labels), model_name='resnet18', pretrained=False)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set the model to evaluation mode
    model.to(DEVICE) # Move model to the appropriate device
    print(f"Model loaded successfully from {MODEL_PATH} on {DEVICE}")
except FileNotFoundError:
    print(f"Error: Model weights file not found at {MODEL_PATH}.")
    print("Please ensure you have trained the model and saved its weights (e.g., by running main.py).")
    exit() # Exit if model not found
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- Define transformations for inference (MUST match data_loader.py's val_test_transform) ---
inference_transform = transforms.Compose([
    transforms.Resize((128, 128)), # Ensure this matches your training/validation data size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Routes ---

# No index route needed with Streamlit

# Streamlit app
st.title("Paddy Disease Prediction")

uploaded_file = st.file_uploader("Upload a paddy leaf image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        img_bytes = uploaded_file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        input_tensor = inference_transform(img)
        input_batch = input_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_batch)
            probabilities = F.softmax(output, dim=1)
            _, predicted_class_idx = torch.max(output, 1)

        prediction_label = class_labels.get(predicted_class_idx.item(), "Unknown")
        confidence = probabilities[0, predicted_class_idx.item()].item() * 100

        # --- Get the prescription ---
        # This will now correctly map to 'Bacterial leaf blight', 'Brown spot', 'Leaf smut'
        # If 'Healthy' is also a class and predicted, it will also map correctly.
        prescription = DISEASE_PRESCRIPTIONS.get(prediction_label, "No specific recommendation available for this. Please consult an agricultural expert.")

        st.markdown(
            """
            <style>
            img {
                width: 100%;
                height: auto;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.image(img, caption="Uploaded Image", use_container_width=True)
        st.write(f"Prediction: {prediction_label}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.write("Prescription:")
        st.write(prescription)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    print("Starting Streamlit application...")
    # No need to run app.run() with Streamlit
    # Streamlit apps are run from the command line: streamlit run app.py
    pass