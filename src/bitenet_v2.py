import os
import torch

from PIL import Image
from torch import nn
from torchvision.models import alexnet, AlexNet_Weights
from src.utils import make_conversion_dict


class BiteNetV2(nn.Module):
    def __init__(self):
        super(BiteNetV2, self).__init__()

        self.weights = AlexNet_Weights.DEFAULT
        self.structure = alexnet(weights = AlexNet_Weights.DEFAULT)

        in_features = self.structure.classifier[6].in_features
        self.structure.classifier[6] = nn.Linear(in_features, 27)

        self.path_to_conversion_file = None

    def forward(self, x):
        return self.structure(x)

    def predict(self, img_path):
        if not os.path.exists(img_path):
            raise ValueError(f'Path {img_path} does not exist')
        
        ckpt_file_path = 'results/bitenetv2/model/bitenetv2.pth'
        ckpt_file = torch.load(ckpt_file_path, map_location=torch.device('cpu'))
        self.path_to_conversion_file = 'data/classes.csv'
        self.load_state_dict(ckpt_file['model'])

        self.eval()
        img = Image.open(img_path).convert('RGB')
        transf = self.weights.transforms()
        transformed_img = transf(img).unsqueeze(0)
        conversion_dict = make_conversion_dict(self.path_to_conversion_file, inverted = True)
        with torch.no_grad():
            model_output = self.forward(transformed_img)
            prediction = model_output.to('cpu').max(1)[1]

        index = str(prediction.item())
        print(f'{conversion_dict[index]}')
        return prediction

