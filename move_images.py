# move_images.py
import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import DocumentClassifier

def classify_image(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),   # Resize the image to 512x512 (same size as the trained model)
        transforms.ToTensor(),           # Convert the image to a tensor
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add a batch dimension for the single image
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model(image)
    probabilities = torch.softmax(output, dim=1)
    _, predicted_class = torch.max(probabilities, 1)

    return predicted_class.item(), probabilities.squeeze().tolist()

def main(source_folder, destination_folder):
    model = DocumentClassifier()
    model.load_state_dict(torch.load('document_classifier_model.pth'))
    model.eval()

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, _, files in os.walk(source_folder):
        for file in files:
            file_path = os.path.join(root, file)
            predicted_class, _ = classify_image(model, file_path)
            if predicted_class == 0:  # Class 0 represents documents
                try:
                    shutil.move(file_path, destination_folder)
                    print(f"Moved {file_path} to {destination_folder}")
                except Exception as e:
                    print(f"Failed to move {file_path} to {destination_folder}: {e}")

if __name__ == "__main__":
    source_folder = "path/to/source/folder"                # Replace with the source folder containing images
    destination_folder = "path/to/destination/folder/docs" # Replace with the destination folder for documents

    main(source_folder, destination_folder)
