from torch import nn

x_layer = 3
y_layer = 3
unit_layer = 32

#layer of generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.for_layer1 = nn.Sequential(
            nn.Conv2d(x_layer, unit_layer * 2, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(unit_layer * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.for_layer2 = nn.Sequential(
            nn.Conv2d(unit_layer * 2, unit_layer * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(unit_layer * 4, unit_layer * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.for_layer3 = nn.Sequential(
            nn.Conv2d(unit_layer * 4, unit_layer * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(unit_layer * 8, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(unit_layer * 8, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(unit_layer * 8, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8)
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(unit_layer * 8, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(unit_layer * 8, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8)
        )
        self.back_layer1 = nn.Sequential(
            nn.ConvTranspose2d(unit_layer * 8, unit_layer * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(unit_layer * 4, unit_layer * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.back_layer2 = nn.Sequential(
            nn.ConvTranspose2d(unit_layer * 4, unit_layer * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(unit_layer * 2, unit_layer * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.back_layer3 = nn.Sequential(
            nn.ConvTranspose2d(unit_layer * 2, x_layer, kernel_size=7, stride=1, padding=3, bias=False)
        )

    def forward(self, x):
        out = self.for_layer1(x)
        out = self.for_layer2(out)
        out = self.for_layer3(out)
        res1 = self.residual1(out)
        out = out + res1
        res2 = self.residual2(out)
        out = out + res2
        out = self.back_layer1(out)
        out = self.back_layer2(out)
        out = self.back_layer3(out)

        return out


#layer of discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(x_layer, unit_layer, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(unit_layer, unit_layer * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(unit_layer * 2, unit_layer * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(unit_layer * 4, unit_layer * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(unit_layer * 4, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(unit_layer * 8, unit_layer * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(unit_layer * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(unit_layer * 8, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = nn.Sigmoid()(out)
        out = nn.AvgPool2d(out.size()[2:])(out)
        return out
