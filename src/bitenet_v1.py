import torch

from torch import nn
from src.utils import transform_input_img
from src.utils import make_conversion_dict

class BiteNetV1(nn.Module):
    def __init__(self):
        super(BiteNetV1,self).__init__()

        self.conv_1 = nn.Sequential(
            # 3x224x224 --> 18x110x110
            nn.Conv2d(in_channels = 3, out_channels = 18, kernel_size = 6, stride = 2),
            # 18x110x110 --> 18x55x55
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            # 18x55x55 --> 28x26x26
            nn.Conv2d(in_channels = 18, out_channels = 28, kernel_size = 5, stride = 2),
            # 28x26x26 --> 28x13x13
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            # 28x13x13 --> 28x6x6
            nn.Conv2d(in_channels = 28, out_channels = 28, kernel_size = 3, stride = 2),
            # 28x6x6 --> 28x3x3
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.classifier = nn.Linear(252, 27)


        self.path_to_norm_params_file = None
        self.path_to_conversion_file = None

    def forward(self,x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x

    def predict(self, img_path):
        self.eval()
        transformed_img = transform_input_img(img_path, self.path_to_norm_params_file)
        conversion_dict = make_conversion_dict(self.path_to_conversion_file, inverted = True)
        with torch.no_grad():
            model_output = self.forward(transformed_img)
            prediction = model_output.to('cpu').max(1)[1]
        index = str(prediction.item())

        print(f'{conversion_dict[index]}')
        return prediction

