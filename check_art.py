import torch
from torchvision import transforms
from PIL import Image
from model import ArtClassifier


def classify_image(image_path):
    # Load the trained model
    model = ArtClassifier()
    model.load_state_dict(torch.load("Train/model_epoch_10.pth"))
    model.eval()

    # Define image transformation
    # Add normalization to the transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        # Open and preprocess the user-input image
        user_image = Image.open(image_path)
        preprocessed_image = transform(user_image).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(preprocessed_image)
            _, predicted = torch.max(output.data, 1)

        # Interpret the result
        if predicted.item() == 1:
            return "The image appears to be human-generated."
        else:
            return "The image appears to be AI-generated."

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    # Replace 'path/to/your/image.jpg' with the actual path to the user's image
    user_image_path = 'Test/ai2.jpg'

    # Perform classification
    result = classify_image(user_image_path)

    # Display the result
    print(result)
