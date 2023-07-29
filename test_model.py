import torch
import torch.nn as nn
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

if __name__ == "__main__":
    model = DocumentClassifier()
    model.load_state_dict(torch.load('document_classifier_model.pth'))
    model.eval()

    image_path = "test1.jpg"  # Replace with the path to sample image

    predicted_class, class_probabilities = classify_image(model, image_path)

    if predicted_class == 0:
        print("The sample image is classified as a document.")
    else:
        print("The sample image is classified as a non-document.")

    print(f"Class probabilities: {class_probabilities}")
