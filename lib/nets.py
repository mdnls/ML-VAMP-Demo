import torch
from torch import nn
import math


class ReLU_MLP(nn.Module):
    # MLP net with set input dimensionality, given intermediate layer dims, and fixed outputs
    def __init__(self, layer_dims, output="linear"):
        '''
        A generic ReLU MLP network.

        Arguments:
            - layer_dims: a list [d1, d2, ..., dn] where d1 is the dimension of input vectors and d1, ..., dn
                        is the dimension of outputs of each of the intermediate layers.
            - output: output activation function, either "sigmoid" or "linear".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''
        super(ReLU_MLP, self).__init__()
        layers = []
        for i in range(1, len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i - 1], layer_dims[i]))
            layers.append(nn.LayerNorm(normalized_shape= layer_dims[i]))
            layers.append(nn.ReLU())
        if (output == "sigmoid"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))
            layers.append(nn.Sigmoid())
        if (output == "linear"):
            layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp, *args):
        if (type(inp) == tuple):
            args = inp[1:]
            inp = inp[0]
        if (len(args) > 0):
            inp = torch.cat([inp] + list(args), dim=1)
        return self.out(inp)

    def clip_weights(self, c):
        for layer in self.layers:
            if (isinstance(layer, nn.Linear) or isinstance(layer, nn.LayerNorm)):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)

class FCImageCritic(nn.Module):
    def __init__(self, inp_WC, hidden_layer_dims):
        super(FCImageCritic, self).__init__()
        self.input_W, self.input_C = inp_WC
        self.hidden_layer_dims = hidden_layer_dims
        self.inp_projector = nn.Linear(in_features=self.input_W ** 2 * self.input_C,
                                       out_features=hidden_layer_dims[0])
        self.outp_projector = nn.Linear(in_features = hidden_layer_dims[-1], out_features=1)
        self.hidden = ReLU_MLP(layer_dims=hidden_layer_dims)
    def forward(self, inp_image):
        return self.outp_projector(nn.functional.relu(self.hidden(self.inp_projector(inp_image.flatten(start_dim=1)))))

class ReLU_CNN(nn.Module):
    def __init__(self, imdims, channels, filter_size, output="sigmoid", layernorm=False):
        '''
        A generic ReLU CNN network.

        Arguments:
            - imdims: a length-2 tuple of integers, or a list of these tuples. Each is image dimensions in HW format.
                If input is a list, the list must have one fewer item than the length of channels. The output of each
                layer is resized to the given dimensions.
            - channels: a list [c1, c2, ..., cn] where c1 is the number of input channels and c2, ..., cn
                    is the number of output channels of each intermediate layer.

                    The final layer does not resize the image so len(channels) = len(imdims) + 1 is required.
            - filter_size: size of convolutional filters in each layer.
            - output: output activation function, either "sigmoid" or "tanh".
            - layernorm: if True, apply layer normalization to the input of each layer.
        '''

        super(ReLU_CNN, self).__init__()
        layers = []

        assert all([type(x) == int for x in channels]), "Channels must be a list of integers"
        def istwotuple(x):
            return (type(x) == tuple) and (len(x) == 2) and (type(x[0]) == int) and (type(x[1]) == int)

        if(istwotuple(imdims)):
            imdims = [imdims for _ in range(len(channels) + 1)]
        elif(all([istwotuple(x) for x in imdims])):
            assert len(imdims)+1 == len(channels), "The length of channels must be one greater than the length of imdims."
        else:
            raise ValueError("Input image dimensions are not correctly formatted.")

        self.imdims = imdims

        padding = int((filter_size - 1) / 2)
        for i in range(1, len(channels) - 1):
            layers.append(nn.Conv2d(channels[i - 1], channels[i], kernel_size=filter_size, padding=padding))
            if(layernorm):
                layers.append(nn.BatchNorm2d(num_features=channels[i]))
            if(imdims[i-1] != imdims[i]):
                layers.append(torch.nn.Upsample(imdims[i], mode='bilinear', align_corners=True))
            layers.append(nn.ReLU(channels[i - 1]))
        if (output == "sigmoid"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
            layers.append(nn.Sigmoid())
        elif (output == "tanh"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
            layers.append(nn.Tanh())
        elif (output == "none"):
            layers.append(nn.Conv2d(channels[-2], channels[-1], kernel_size=filter_size, padding=padding))
        else:
            raise ValueError("Unrecognized output function.")
        self.layers = layers
        self.out = nn.Sequential(*layers)

    def forward(self, inp):
        return self.out(inp)

    def clip_weights(self, c):
        for layer in self.layers:
            if(isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d)):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)


class ImageCritic(nn.Module):
    def __init__(self, input_im_size, layers, channels, layernorm=False):
        '''
        The critic accepts an image or batch of images and outputs a scalar or batch of scalars.
        Given the input image size and a desired number of layers, the input image is downsampled at each layer until
            is a set of 4x4 feature maps. The scalar output is a regression over the channel dimension.

        Arguments:
            - input_im_size (int): size of input images, which are assumed to be square
            - layers (int): number of layers
            - channels (int):
        '''
        super(ImageCritic, self).__init__()


        scale = (input_im_size/4)**(-1/ (layers-1) )

        imdims = [input_im_size] + \
                 [int(input_im_size * (scale**k)) for k in range(1, layers-1)]  + \
                 [4]

        imdims = [(x, x) for x in imdims]

        self.net = ReLU_CNN(imdims, channels, filter_size=3, output="none", layernorm=layernorm)
        self.linear = torch.nn.Linear(4 * 4 * channels[-1], 1)
    def forward(self, image, *args):
        # the *args is a hack to allow extra arguments, for ex, if one wants to pass two input images.
        # TODO: nicely implement functionality for networks which take in two or more arguments.
        if(type(image) == tuple):
            args = image[1:]
            image = image[0]
        if(len(args) > 0):
            image = torch.cat([image] + list(args), dim=1)
        return self.linear(torch.flatten(self.net(image), start_dim=1))
    def clip_weights(self, c):
        for p in self.parameters():
            p.data.clamp_(-c, c)


class Compatibility(nn.Module):
    def __init__(self, inp_density_param, outp_density_param, regularization, reg_strength, transport_cost):
        '''
        Learn a density over the data by training an ImageCritic to solve a regularized OT problem.
        The ImageCritic induces a data density via a regularization specific function.

        Arguments:
            - inp_density_parameter (ImageCritic): an instantiated image critic which represents input density paramater
            - outp_density_parameter (ImageCritic):  an instantiated image critic which represents output density parameter
            - regularization (str): 'entropy' or 'L2'
            - reg_strength (float): weight of the regularization term
            - transport_cost (func: Image_Batch x Image_Batch -> vector): pairwise transport cost of images in two batches
        '''
        super(Compatibility, self).__init__()
        self.transport_cost = transport_cost

        # based on regularization: need a method to output a density
        # and a method to output a regularization value, to be used in a dual objective
        r = reg_strength
        if(regularization == "entropy"):
            self.penalty_fn = lambda x, y: r * torch.exp((1/r)*self._violation(x, y) - 1)
            self.compatibility_fn = lambda x, y: torch.exp((1/r)*self._violation(x, y) - 1)
        elif(regularization == "l2"):
            self.penalty_fn = lambda x, y: (1/(4*r)) * torch.relu(self._violation(x, y))**2
            self.compatibility_fn = lambda x, y: (1/(2*r)) * torch.relu(self._violation(x, y))
        else:
            raise ValueError("Invalid choice of regularization")

        self.inp_density_param_net = inp_density_param
        self.outp_density_param_net = outp_density_param

    def _violation(self, x, y):
        if(type(x) == tuple and type(y) == tuple):
            t_cost = sum([self.transport_cost(ex, why) for ex, why in zip(x, y)])
        else:
            t_cost = self.transport_cost(x, y)
        return self.inp_density_param_net(x) + self.outp_density_param_net(y) - t_cost

    def penalty(self, x, y):
        return self.penalty_fn(x, y)

    def forward(self, x, y):
        return self.compatibility_fn(x, y)

    def inp_density_param(self, x, *args):
        return self.inp_density_param_net(x, *args)

    def outp_density_param(self, y, *args):
        return self.outp_density_param_net(y, *args)


class CNN_Sampler(nn.Module):
    def __init__(self, inp_dim, outp_wc, layers, channels):
        '''
        The list of channels should be of the form [inp_dim, h1, h2, h3, ..., hk, outp_wc[1]]
        ie. the first and last channels are the number of input dimensions and the number of outp channels resp.

        Args:
            inp_dim:
            outp_wc:
            layers:
            channels:
        '''
        super(CNN_Sampler, self).__init__()
        assert inp_dim == channels[0], "Channels[0] should match the input channels."
        assert outp_wc[1] == channels[-1], "Channels[-1] should match the output channels."
        self.inp_dim = inp_dim
        self.outp_wc = outp_wc
        self.layers = layers
        self.channels = channels

        output_im_size, output_channels = outp_wc

        upsampler_scale = (output_im_size/8)**(1/(layers-1) )

        f = [ (upsampler_scale**k) for k in range(1, layers-1)]
        sampler_imdims = [8] + \
                           [math.ceil(8 * (upsampler_scale**k)) for k in range(1, layers-1)] + \
                           [output_im_size]
        sampler_imdims = [(x, x) for x in sampler_imdims]

        self.sampler_imdims = sampler_imdims
        self.uplinproj = nn.Linear(in_features=channels[0], out_features=8*8*channels[0])
        self.upsampler = ReLU_CNN(imdims=sampler_imdims, channels=channels, filter_size=3, output="tanh")

    def forward(self, latent):
        return self.upsampler(self.uplinproj(latent).view([-1, self.inp_dim, 8, 8]))


class FCN_Sampler(nn.Module):
    def __init__(self, inp_dim, outp_wc, hidden_layer_dims):
        super(FCN_Sampler, self).__init__()
        self.hidden_layer_dims = hidden_layer_dims
        self.outp_wc = outp_wc
        self.outp_dim = outp_wc[0] * outp_wc[0] * outp_wc[1]
        self.latent_dim = inp_dim

        self.net = ReLU_MLP(layer_dims=[inp_dim] + hidden_layer_dims + [self.outp_dim])
    def forward(self, latent):
        return torch.tanh(self.net(latent)).view([-1, self.outp_wc[1], self.outp_wc[0], self.outp_wc[0]])


class NetEnsemble():
    '''
    Base class for net ensemble classes which group together multiple neural networks.
    '''
    def save_net(self, net, path):
        torch.save(net.state_dict(), path)
    def load_net(self, net, path):
        try:
            net.load_state_dict(torch.load(path))
        except FileNotFoundError:
            print(f"Could not load {path}. Starting from random initialization.")

