import torch.nn as nn
from model.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

class FPN(nn.Module):
    def __init__(self, resnet_factory):
        super(FPN, self).__init__()
        self.resnet = resnet_factory()
        
        # top-down layers
        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        #self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # lateral convs
        self.lat_c5p5 = nn.Conv2d(512 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.lat_c4p4 = nn.Conv2d(256 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.lat_c3p3 = nn.Conv2d(128 * 4, 256, kernel_size=1, stride=1, padding=0)
        self.lat_c2p2 = nn.Conv2d( 64 * 4, 256, kernel_size=1, stride=1, padding=0)

        # between fpn layers
        self.upsample = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2,padding=1)

    def forward(self, x):
        C1, C2, C3, C4, C5 = self.resnet(x)
        P5 = self.lat_c5p5(C5)
        P4 = self.lat_c4p4(C4) + self.upsample(P5)
        P3 = self.lat_c3p3(C3) + self.upsample(P4)
        P2 = self.lat_c2p2(C2) + self.upsample(P3)

        #P5 = self.fpn_p5(P5)
        P4 = self.fpn_p4(P4)
        P3 = self.fpn_p3(P3)
        P2 = self.fpn_p2(P2)

        return P2, P3, P4, P5


def fpn18():
    """Constructs a ResNet-18+FPN model.
    """
    model = FPN(resnet18)
    return model


def fpn34():
    """Constructs a ResNet-34+FPN model.
    """
    model = FPN(resnet34)
    return model


def fpn50():
    """Constructs a ResNet-50+FPN model.
    """
    model = FPN(resnet50)
    return model


def fpn101():
    """Constructs a ResNet-101+FPN model.
    """
    model = FPN(resnet101)
    return model


def fpn152():
    """Constructs a ResNet-152+FPN model.
    """
    model = FPN(resnet152)
    return model

