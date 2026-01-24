from torch import nn


class BiteNetV1(nn.Module):
    def __init__(self):
        super(BiteNetV1,self).__init__()

        self.conv_1 = nn.Sequential(
            # 3x384x384 --> 18x190x190
            nn.Conv2d(in_channels = 3, out_channels = 18, kernel_size = 6, stride = 2),
            # 18x190x190 --> 18x95x95
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv_2 = nn.Sequential(
            # 18x95x95 --> 28x46x46
            nn.Conv2d(in_channels = 18, out_channels = 28, kernel_size = 5, stride = 2),
            # 28x46x46 --> 28x23x23
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.conv_3 = nn.Sequential(
            # 28x23x23 --> 28x10x10
            nn.Conv2d(in_channels = 28, out_channels = 28, kernel_size = 5, stride = 2),
            # 28x10x10 --> 28x5x5
            nn.MaxPool2d(2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(700,360),
            nn.ReLU(),
            nn.Linear(360,252),
            nn.ReLU(),
            nn.Linear(252,101)
        )
            
    def forward(self,x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.classifier(x.view(x.shape[0],-1))
        return x

