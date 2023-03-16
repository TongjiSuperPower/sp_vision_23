import cv2
import torch
import torch.nn as nn
from torchvision.transforms import Resize


class_names = ('big_one', 'big_three', 'big_four', 'big_five', 'big_base',
               'small_two', 'small_three', 'small_four', 'small_five',
               'small_base', 'small_santry', 'small_outpost',
               'no_pattern')


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.BatchNorm2d(6),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 9 * 9, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        logits = self.classifier(x)
        return logits


class Classifier:
    def __init__(self) -> None:
        self.model = LeNet(len(class_names))
        self.tranform = Resize(50)
        self.model.load_state_dict(torch.load('assets/model.pth'))
        self.model.eval()

    def classify(self, pattern_img: cv2.Mat) -> tuple[float, str]:
        pattern_img = torch.from_numpy(pattern_img)
        pattern_img.to(torch.float32)
        pattern_img = pattern_img / 255.0
        pattern_img = pattern_img.unsqueeze(0)
        pattern_img = self.tranform(pattern_img)

        output = self.model(pattern_img.unsqueeze(0))
        probabilities = torch.nn.functional.softmax(output, dim=1)
        max_probability, predicted_id = torch.max(probabilities, dim=1)
        confidence = max_probability.item()
        class_id = predicted_id.item()

        return confidence, class_names[class_id]
