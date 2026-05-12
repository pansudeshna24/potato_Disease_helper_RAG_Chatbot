import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image

MODEL_PATH = "src/cnn/models/potato_disease_cnn.pth"

class CNNDiseaseClassifier:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device
        )

        self.classes = checkpoint["classes"]

        self.model = efficientnet_b0(weights=None)

        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            len(self.classes)
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image: Image.Image):

        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():

            outputs = self.model(tensor)

            probs = F.softmax(outputs, dim=1)

            confidence, pred_idx = torch.max(probs, 1)

        prediction = self.classes[pred_idx.item()]

        return {
            "prediction": prediction,
            "confidence": round(confidence.item() * 100, 2)
        }