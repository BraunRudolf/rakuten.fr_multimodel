import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(self, text_model, image_model, num_classes):
        super(FusionModel, self).__init__()
        lstm_layer = list(text_model.children())[1]
        self.text_hidden_size = lstm_layer.hidden_size
        self.text_model = text_model
        self.image_model = image_model
        self.fusion = nn.Linear(text_model.fc1.out_features + image_model[9].out_features, 1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, inputs):
        texts, images = inputs

        text_features = self.text_model(texts)
        image_features = self.image_model(images).flatten(start_dim=1)
        combined = torch.cat((text_features, image_features), dim=1)
        fused = torch.relu(self.fusion(combined))
        output = self.classifier(fused)

        return output
