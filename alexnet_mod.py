import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from pruning.layers import MaskedLinear, MaskedConv2d 


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = MaskedConv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = MaskedConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = MaskedConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = MaskedConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = MaskedConv2d(256, 256, kernel_size=3, padding=1)
        self.linear6 = MaskedLinear(256 * 6 * 6, 4096)
        self.linear7 = MaskedLinear(4096, 4096)
        self.linear8 = MaskedLinear(4096, num_classes)
        self.features = nn.Sequential(
            self.conv1,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.conv2,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.conv3,
            nn.ReLU(inplace=True),
            self.conv4,
            nn.ReLU(inplace=True),
            self.conv5,
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            self.linear6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            self.linear7,
            nn.ReLU(inplace=True),
            self.linear8,
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        # self.conv1.set_mask(torch.from_numpy(masks[0]))
        # self.conv2.set_mask(torch.from_numpy(masks[1]))
        # self.conv4.set_mask(torch.from_numpy(masks[2]))
        # self.conv5.set_mask(torch.from_numpy(masks[3]))
        # self.linear6.set_mask(torch.from_numpy(masks[4]))
        # self.linear7.set_mask(torch.from_numpy(masks[5]))
        # self.linear8.set_mask(torch.from_numpy(masks[6]))
        self.conv1.set_mask(masks[0])
        self.conv2.set_mask(masks[1])
        self.conv3.set_mask(masks[2])
        self.conv4.set_mask(masks[3])
        self.conv5.set_mask(masks[4])
        self.linear6.set_mask(masks[5])
        self.linear7.set_mask(masks[6])
        self.linear8.set_mask(masks[7])


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']), strict=False)
    return model
