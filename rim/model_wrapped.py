from torch import nn

class model_wrapped(nn.Module):
    def __init__(self, preprocess_net, main_net):
        super(model_wrapped, self).__init__()
        self.preprocess_net = preprocess_net
        self.main_net = main_net

    def forward(self, x, image, args, imgname=None, hx=None):
        x0 = self.preprocess_net(x)
        x = self.main_net(x0, image, args, imgname, hx)
        return [x0, x]

