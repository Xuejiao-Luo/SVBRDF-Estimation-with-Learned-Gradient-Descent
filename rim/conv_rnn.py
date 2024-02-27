import torch
import torch.nn as nn

class InputRNN(torch.nn.Module):

    def __init__(self, rnn_cell=None, input_fun=None):
        super(InputRNN, self).__init__()

        self.rnn_cell = rnn_cell
        self.input_fun = input_fun

    def forward(self, x, hx=None):

        if self.input_fun is not None:
            x = self.input_fun.forward(x, )
        if self.rnn_cell is not None:
            x = self.rnn_cell.forward(x, )
            hx = x

        return x, hx

class InputRNN_highlightaware(torch.nn.Module):

    def __init__(self, rnn_cell=None, input_fun=None, input_fun2=None, input_fun3=None, instance_normalization=None):
        super(InputRNN_highlightaware, self).__init__()

        self.rnn_cell = rnn_cell
        self.input_fun = input_fun
        self.input_fun2 = input_fun2
        self.input_fun3 = input_fun3
        self.sigmoid = nn.Sigmoid()
        self.instance_normalization = instance_normalization

    def forward(self, x, hx=None):

        if self.input_fun is not None:
            x1 = self.instance_normalization(self.input_fun.forward(x, ))
        if self.input_fun2 is not None:
            x2 = self.sigmoid(self.input_fun2.forward(x, ))
        if self.input_fun3 is not None:
            x3 = self.input_fun3.forward(x, )
        x = x1 * x2 + x3
        if self.rnn_cell is not None:
            x = self.rnn_cell.forward(x, )
            hx = x

        return x, hx


class ConvNonlinear(nn.Module):

    def __init__(self, input_size, features, conv_dim, kernel_size, dilation, bias, nonlinear='relu'):
        super(ConvNonlinear, self).__init__()

        self.input_size = input_size
        self.features = features
        self.bias = bias
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)
        if nonlinear is not None and nonlinear.upper() == 'RELU':
            self.nonlinear = torch.nn.ReLU(inplace=False)
        elif nonlinear is not None and nonlinear.upper() == 'TANH':
            self.nonlinear = torch.nn.Tanh()
        elif nonlinear is not None and nonlinear.upper() == 'LRELU':
            self.nonlinear = torch.nn.LeakyReLU()
        elif nonlinear is None:
            self.nonlinear = lambda x: x
        else:
            ValueError('Please specify a proper')

        self.padding = [torch.nn.ReplicationPad1d(dilation * (kernel_size - 1) // 2),
                        torch.nn.ReplicationPad2d(dilation * (kernel_size - 1) // 2),
                        torch.nn.ReplicationPad3d(dilation * (kernel_size - 1) // 2)][conv_dim - 1]
        self.conv_layer = self.conv_class(in_channels=input_size, out_channels=features,
                                          kernel_size=kernel_size, padding=0,
                                          dilation=dilation, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity='tanh')

        if self.conv_layer.bias is not None:
            nn.init.zeros_(self.conv_layer.bias)

    def determine_conv_class(self, n_dim):
        if n_dim is 1:
            return nn.Conv1d
        elif n_dim is 2:
            return nn.Conv2d
        elif n_dim is 3:
            return nn.Conv3d
        else:
            NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        s = '{input_size}, {features}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinear' in self.__dict__ and self.nonlinear != "tanh":
            s += ', nonlinearity={nonlinear}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def forward(self, input):

        return self.nonlinear(self.conv_layer(self.padding(input)))


class ConvRNNCellBase(nn.Module):

    def __init__(self, input_size, hidden_size, num_chunks, conv_dim, kernel_size,
                 dilation, bias):
        super(ConvRNNCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)
        self.ih = self.conv_class(in_channels=input_size, out_channels=num_chunks * hidden_size,
                                  kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2,
                                  dilation=dilation, bias=bias)
        self.hh = self.conv_class(in_channels=hidden_size, out_channels=num_chunks * hidden_size,
                                  kernel_size=kernel_size, padding=dilation * (kernel_size - 1) // 2,
                                  dilation=dilation, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        self.hh.weight.data = self.orthotogonalize_weights(self.hh.weight.data)

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)
            nn.init.zeros_(self.hh.bias)

    def orthotogonalize_weights(self, weights, chunks=1):
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks, 0)], 0)

    def determine_conv_class(self, n_dim):

        if n_dim is 1:
            return nn.Conv1d
        elif n_dim is 2:
            return nn.Conv2d
        elif n_dim is 3:
            return nn.Conv3d
        else:
            NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class ConvGRUCell(ConvRNNCellBase):
    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        super(ConvGRUCell, self).__init__(input_size=input_size, hidden_size=hidden_size,
                                          num_chunks=3, conv_dim=conv_dim, kernel_size=kernel_size,
                                          dilation=dilation, bias=bias)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros((input.size(0), self.hidden_size) + input.size()[2:],
                                 requires_grad=False)

        ih = self.ih(input).chunk(3, dim=1)
        hh = self.hh(hx).chunk(3, dim=1)

        z = torch.sigmoid(ih[0] + hh[0])
        r = torch.sigmoid(ih[1] + hh[1])
        n = torch.tanh(ih[2] + r * hh[2])

        hx = (1. - z) * hx + z * n

        return hx


class IndRNN(ConvRNNCellBase):
    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        super(IndRNN, self).__init__(input_size=input_size, hidden_size=hidden_size,
                                     num_chunks=3, conv_dim=conv_dim, kernel_size=kernel_size,
                                     dilation=dilation, bias=bias)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if isinstance(kernel_size, int):
            kernel = (kernel_size, kernel_size)
            dilation = (dilation, dilation)

        padding = [int((k + (k - 1) * (d - 1)) / 2) for k, d in zip(kernel, dilation)]

        if padding[0] > 0:
            self.padding = nn.ReplicationPad2d(padding)

        self.ih = nn.Conv2d(input_size, hidden_size, kernel[0], dilation=dilation[0], bias=bias)
        self.hh = nn.Parameter(
            nn.init.normal_(torch.empty(1, hidden_size, 1, 1), std=1. / (hidden_size * (1 + kernel[1] ** 2))))

        nn.init.normal_(self.ih.weight, std=1. / (hidden_size * (1 + kernel[0] ** 2)))

        if bias:
            nn.init.constant_(self.ih.bias, 0)

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros((input.size(0), self.hidden_size) + input.size()[2:],
                                 requires_grad=False)
        if hasattr(self, 'padding'):
            input = self.padding(input)
        ih = self.ih(input)
        return ih + self.hh * hx


class ConvRNN(nn.Module):
    def __init__(self, input_size, recurrent_layer='gru',
                 conv_params={
                     'features': [256, 256, 10],
                     'k_size': [5, 3, 3],
                     'dilation': [1, 2, 1],
                      'bias': [True, True, False],
                     # 'bias': [True, True, True],
                     # 'nonlinear': ['lrelu', 'lrelu', None]
                      'nonlinear': ['lrelu', 'lrelu', 'tanh']
                 },
                 rnn_params={
                     'features': [64, 64, 0],
                     'k_size': [1, 1, 0],
                     'dilation': [1, 1, 0],
                     'bias': [True, True, False],
                     'rnn_type': ['gru', 'gru', None]
                 },
                 conv_dim=2):

        super(ConvRNN, self).__init__()

        self.input_size = input_size
        self.recurrent_layer = recurrent_layer
        self.conv_dim = conv_dim
        self.conv_params = conv_params
        self.rnn_params = rnn_params

        conv_params = zip(*(self.conv_params[k] for k in ['features', 'k_size', 'dilation', 'bias', 'nonlinear']))
        rnn_params = zip(*(self.rnn_params[k] for k in ['features', 'k_size', 'dilation', 'bias', 'rnn_type']))

        self.layers = nn.ModuleList()
        for (conv_features, conv_k_size, conv_dilation, conv_bias, nonlinear), \
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type) in zip(conv_params, rnn_params):
            conv_layer = None
            rnn_layer = None

            if conv_features > 0:
                conv_layer = ConvNonlinear(input_size, conv_features, conv_dim=self.conv_dim,
                                           kernel_size=conv_k_size, dilation=conv_dilation, bias=conv_bias,
                                           nonlinear=nonlinear)
                input_size = conv_features

            if rnn_features > 0 and rnn_type is not None:
                if self.recurrent_layer != rnn_type:
                    rnn_type = self.recurrent_layer

                if rnn_type == 'gru':
                    rnn_type = ConvGRUCell
                elif rnn_type == 'indrnn':
                    rnn_type = IndRNN
                elif issubclass(rnn_type, ConvRNNCellBase):
                    rnn_type = rnn_type
                else:
                    ValueError('Please speacify a proper rrn_type')

                rnn_layer = rnn_type(input_size, rnn_features, conv_dim=self.conv_dim,
                                     kernel_size=rnn_k_size, dilation=rnn_dilation, bias=rnn_bias)
                input_size = rnn_features

            self.layers.append(InputRNN(rnn_layer, conv_layer))

    def forward(self, input, hx=None):
        if not hx:
            hx = [None] * len(self.layers)

        hidden_new = []

        for layer, local_hx in zip(self.layers, hx):
            input, new_hx = layer.forward(input, local_hx)
            hidden_new.append(new_hx)

        return input, hidden_new


class ConvRNN_highlightaware(nn.Module):
    def __init__(self, input_size, recurrent_layer='gru',
                 conv_params={
                     'features': [128, 128, 9],
                     'k_size': [5, 3, 3],
                     'dilation': [1, 2, 1],
                     'bias': [True, True, False],
                     'nonlinear': ['lrelu', 'lrelu', None]
                 },
                 conv_params2={
                     'features': [128, 128, 9],
                     'k_size': [5, 3, 3],
                     'dilation': [1, 2, 1],
                     'bias': [True, True, False],
                     'nonlinear': [None, None, None]
                 },
                 rnn_params={
                     'features': [64, 64, 0],
                     'k_size': [1, 1, 0],
                     'dilation': [1, 1, 0],
                     'bias': [True, True, False],
                     'rnn_type': ['gru', 'gru', None]
                 },
                 conv_dim=2):
        super(ConvRNN_highlightaware, self).__init__()

        self.input_size = input_size
        self.recurrent_layer = recurrent_layer
        self.conv_dim = conv_dim
        self.conv_params = conv_params
        self.conv_params2 = conv_params2
        self.rnn_params = rnn_params

        conv_params = zip(*(self.conv_params[k] for k in ['features', 'k_size', 'dilation', 'bias', 'nonlinear']))
        conv_params2 = zip(*(self.conv_params2[k] for k in ['features', 'k_size', 'dilation', 'bias', 'nonlinear']))
        rnn_params = zip(*(self.rnn_params[k] for k in ['features', 'k_size', 'dilation', 'bias', 'rnn_type']))

        self.layers = nn.ModuleList()
        for (conv_features, conv_k_size, conv_dilation, conv_bias, nonlinear), \
            (conv_features2, conv_k_size2, conv_dilation2, conv_bias2, nonlinear2), \
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type) in zip(conv_params, conv_params2, rnn_params):
            conv_layer = None
            rnn_layer = None

            if conv_features > 0:
                instance_normalization = nn.InstanceNorm2d(input_size)
                conv_layer = ConvNonlinear(input_size, conv_features, conv_dim=self.conv_dim,
                                           kernel_size=conv_k_size, dilation=conv_dilation, bias=conv_bias,
                                           nonlinear=nonlinear)
                conv_layer2 = ConvNonlinear(input_size, conv_features2, conv_dim=self.conv_dim,
                                           kernel_size=conv_k_size2, dilation=conv_dilation2, bias=conv_bias2,
                                           nonlinear=nonlinear2)
                conv_layer3 = ConvNonlinear(input_size, conv_features, conv_dim=self.conv_dim,
                                           kernel_size=conv_k_size, dilation=conv_dilation, bias=conv_bias,
                                           nonlinear=nonlinear)
                input_size = conv_features

            if rnn_features > 0 and rnn_type is not None:
                if self.recurrent_layer != rnn_type:
                    rnn_type = self.recurrent_layer

                if rnn_type == 'gru':
                    rnn_type = ConvGRUCell
                elif rnn_type == 'indrnn':
                    rnn_type = IndRNN
                elif issubclass(rnn_type, ConvRNNCellBase):
                    rnn_type = rnn_type
                else:
                    ValueError('Please speacify a proper rnn_type')

                rnn_layer = rnn_type(input_size, rnn_features, conv_dim=self.conv_dim,
                                     kernel_size=rnn_k_size, dilation=rnn_dilation, bias=rnn_bias)
                input_size = rnn_features

            self.layers.append(InputRNN_highlightaware(rnn_layer, conv_layer, conv_layer2, conv_layer3, instance_normalization))

    def forward(self, input, hx=None):
        if not hx:
            hx = [None] * len(self.layers)

        hidden_new = []

        for layer, local_hx in zip(self.layers, hx):
            input, new_hx = layer.forward(input, local_hx)
            hidden_new.append(new_hx)

        return input, hidden_new
