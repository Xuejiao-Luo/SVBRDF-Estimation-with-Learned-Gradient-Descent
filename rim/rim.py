import torch
from torch import nn

from torch.autograd import Variable
from utils.loss import rmse
from utils.rendererMG import renderTex

class RIM(nn.Module):

    def __init__(self, rnn, n_steps=1):
        super(RIM, self).__init__()
        self.rnn = rnn
        self.n_steps = n_steps

    def forward(self, eta, image, args, imgname, hx=None):

        etas = []
        for i in range(self.n_steps):
            with torch.enable_grad():
                eta_pro = Variable(eta.clone(), requires_grad=True)

                lp = torch.tensor([0,0,250], dtype=torch.int, device=eta_pro.device)
                cp = torch.tensor([0,0,250], dtype=torch.int, device=eta_pro.device)
                L = torch.tensor([329288,], dtype=torch.int, device=eta_pro.device)
                re_renderedimage = renderTex(eta_pro[:,0:9,:,:], lp, cp, L)

                rmse(re_renderedimage, image).backward()
                grad_eta = eta_pro.grad * 100000
                print("step ", i )
                print("eta.abs().mean(): ", eta.abs().mean())
                print("grad_eta.abs().mean(): ", grad_eta.abs().mean())

            x_in = torch.cat((eta, grad_eta), 1)
            delta, hx = self.rnn.forward(x_in, hx)

            eta = eta+delta

        etas = eta
        return etas