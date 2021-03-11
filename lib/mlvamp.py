import torch
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

'''
Top level class: ML_VAMP_Solver
    - Args: measurement matrix, generator
    - Function:
        - View the 'sampler' as the composition of latent -> generator -> measurement matrix -> measurements
        - Seek to recover MAP/MMSE estimate of latent from measurements
    - Init:
        1. Index intermediate activations as (input latent) <-> 0, first layer output <-> 1, ..., final layer output <-> L
        2. Index measurements as L+1 viewing measurement matrix as an additional layer
        3. For indices l = 0...L+1, instantiate LayerEstimator l corresp to a particular layer
        4. The LayerEstimator is responsible for storing all quantities related to index l including input and output beliefs.
            - Different LayerEstimators for each of: Conv2D layer, Upsample layer, ReLU layer, Tanh layer
            - Each LayerEstimator has weights that are updated iteratively
'''

class LayerEstimator():
    def __init__(self, mode, in_dim, out_dim,
                 measurements=None, measurement_prec=None, prior_mean=None, prior_prec=None,
                 alpha_min=1e-5, alpha_max=1 - 1e-5,
                 gm_min=0, gm_max=10,
                 ):
        self.params = {
            "r-": None,  # residuals
            "r+": None,
            "z-": None,  # latent estimates
            "z+": None,
            "gm+": None,  # gamma
            "gm-": None,
            "e+": None,  # eta
            "e-": None,
            "t+": None, # theta
            "t-": None
        }
        assert mode in ["map", "mmse"], f"{mode} is not a recognized estimation mode. Choose 'map' or 'mmse'."
        self.incoming = EmptyNeighbor()
        self.outgoing = EmptyNeighbor()
        self.measurements = measurements
        self.measurements_prec = measurement_prec
        self.prior_mean = prior_mean
        self.prior_prec = prior_prec
        self.mode = mode
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.gm_min = gm_min
        self.gm_max = gm_max

    def register_neighbors(self, incoming, outgoing):
        self.incoming = incoming
        self.outgoing = outgoing

    def parameter_init(self, in_dim, out_dim, gm_init=1):
        if(isinstance(self.outgoing, EmptyNeighbor)):
            self.params["r-"] = self.measurements
            self.params["gm-"] = self.measurements_prec
        else:
            self.params["r-"] = np.zeros((in_dim,))
            self.params["gm-"] = gm_init

    def step(self, dir):
        '''
        Iterate timestep from ( (k-1)+ , k- ) to ( k+, (k+1)- ).

        During a forward pass at time k, compute
        - z+, theta+, alpha+, r+
        using the quantities
        - r-, alpha-
        which were computed at time k-1 in advance of time k.

        during backward pass at time k, compute
        - z- and theta- for time k, using *future info* from *previous layer* to compute theta-
        - alpha-, r- for time k+1
        using quantities computed at time k or in previous layers
        '''
        if(dir == "fwd"):

            fwd_theta = [self.params["gm-"], self.incoming["gm+"]]
            fwd_z, fwd_a = self._g(self.params["r-"], self.incoming["r+"],
                                   fwd_theta, dir="fwd", with_onsager=True, mode=self.mode)

            fwd_a = np.clip(fwd_a, self.alpha_min, self.alpha_max)

            fwd_r = (fwd_z - fwd_a * self.params["r-"]) / (1 - fwd_a)
            fwd_eta = self.params["gm-"] / fwd_a
            fwd_gamma = fwd_eta - self.params["gm-"]

            fwd_gamma = np.clip(fwd_gamma, self.gm_min, self.gm_max)

            self.params["z+"] = fwd_z
            self.params["r+"] = fwd_r
            self.params["e+"] = fwd_eta
            self.params["gm+"] = fwd_gamma
            print(f"FWD: gm+: {self.params['gm+']}, e+: {self.params['e+']}")

            #self.params["t+"] = fwd_theta
        elif(dir == "bck"):
            # update the bck pass to overwrite previous layers residuals. Ie. identify a backward pass
            #   with the calling of a certain node's proximal function.
            bck_theta = [self.outgoing["gm-"], self.params["gm+"]]
            bck_z, bck_a = self.outgoing._g(self.outgoing["r-"], self.params["r+"],
                                   bck_theta, dir="bck", with_onsager=True, mode=self.mode)

            bck_a = np.clip(bck_a, self.alpha_min, self.alpha_max)

            '''
            Q
            1. Why does thresholding alpha break bck pass but not forward pass
                > gm- is initialized at a very small value, balancing the min clipping on alpha.
            2. Is the indexing correct for the bck pass given that it is 1 offset?
                > signs point to yes. The layer indices all match in updates to eta so it should be straightforward.
            3. What is the issue?
                > Algorithm doesn't converge. Why? Error is blowing up. Also, gamma blows up. 
                >  Upper bound gamma? Note, VAMP bounds the inverse of gamma between 0 and big. 
                > I'm not sure what eta does so I don't know how to bound gamma relative to it. 
            '''


            bck_r = (bck_z - bck_a * self.params["r+"]) / (1 - bck_a)
            bck_eta = self.params['gm+'] / bck_a
            bck_gamma = bck_eta - self.params["gm+"]

            bck_gamma = np.clip(bck_gamma, self.gm_min, self.gm_max)

            self.params["z-"] = bck_z
            self.params["r-"] = bck_r
            self.params["e-"] = bck_eta
            self.params["gm-"] = bck_gamma
            print(f"BCK: gm-: {self.params['gm-']}, e+: {self.params['e-']}")
        else:
            raise ValueError(f"dir '{dir}' not recognized. Must be one of 'fwd' or 'bck'.")

    def _g(self, in_res, out_res, theta, dir, with_onsager=False, mode="map"):
        '''
        Using input and output residuals, compute a denoised estimate of this layer's activation.
        Return the denoised estimate.

        Args:
            in_res (torch.Tensor or None): the input residual.
            out_res (torch.Tensor or None): the output residual.
            theta (torch.Tensor): noise estimate.
            dir (string): either "fwd" or "bck"
            with_onsager: if True, returns a tuple (z, a) where z is the denoised estimate and a is the
                corresponding Onsager term.
            mode: either "map" for maximum a posteriori estimation or "mmse" for minimum mean squared error
                estimation.

        Returns (torch.Tensor or (torch.Tensor, torch.Tensor)): either z or (z, a) where z is the
            denoised estimate and a is the corresponding Onsager term.
        '''
        raise NotImplementedError("Subclasses implement this method.")

    def _val_g_inputs(self, out_res, in_res, theta, dir, mode):
        assert mode in ["map", "mmse"], f"Mode {mode} not recognized. Choose 'map' or 'mmse'."
        assert dir in ["fwd", "bck"], f"Direction {dir} not recognized. Choose 'fwd' or 'bck'."

        def given(x):
            return (x is not None)

        if(isinstance(self.incoming, EmptyNeighbor) and not isinstance(self.outgoing, EmptyNeighbor)):
            assert dir == "fwd", "Cannot run backward iteration at first layer."
            in_res = np.zeros_like(out_res)
            theta[1] = self.prior_prec
            if(not (given(in_res) or given(theta[1])) ):
                raise ValueError("Attempt to use input layer without passing input prior information.")
        elif(isinstance(self.outgoing, EmptyNeighbor) and not isinstance(self.incoming, EmptyNeighbor)):
            assert dir == "bck", "Cannot run forward iteration at last layer."
            out_res = self.measurements
            theta[0] = self.measurements_prec
            if(not (given(out_res) or given(theta[0])) ):
                raise ValueError("Attempt to use output layer without passing output prior information.")
        elif(isinstance(self.outgoing, EmptyNeighbor) and isinstance(self.incoming, EmptyNeighbor)):
            raise ValueError("Attempt to forward or backward pass on isolated node.")

        return out_res, in_res, theta

    def __delitem__(self, key):
        self.params.__delattr__(key)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

class EmptyNeighbor():
    '''This class is a signifier that a certain layer has either no left neighbor or right neighbor.
    The empty neighbor takes the place of the left and right neighbor and declares explicit
    'None' values to fulfull parameter requests from neighbors.'''
    def __init__(self):
        self.params = {
            "r-": None,  # residuals
            "r+": None,
            "z-": None,  # latent estimates
            "z+": None,
            "gm+": None,  # gamma
            "gm-": None,
            "e+": None,  # eta
            "e-": None
            }
    def step(self):
        pass
    def register_neighbors(self, incoming, outgoing):
        pass
    def __delitem__(self, key):
        self.params.__delattr__(key)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

class GaussianPrior(LayerEstimator):
    def __init__(self, mode, dim, mean, stddev):
        super().__init__(mode, in_dim = dim, out_dim = dim, prior_mean=mean, prior_prec = 1/(stddev**2))

    def _g(self, out_res, in_res, theta, dir, with_onsager=False, mode="map"):
        out_res, in_res, theta = self._val_g_inputs(out_res, in_res, theta, dir, mode)

        if (mode == "map"):
            gm_out, gm_in = theta

            if(dir == "bck"):
                raise ValueError("Cannot run backward pass on a prior distribution.")
            z_map_est = (gm_out * out_res + gm_in * in_res) / (gm_out + gm_in)

            if (with_onsager):
                # onsager is average of grad wrt out residual
                onsager = gm_in / (gm_out + gm_in)
                return z_map_est, onsager
            else:
                return z_map_est
        elif (mode == "mmse"):
            raise NotImplementedError("mmse is not implemented for ReLU.")


class ReLU(LayerEstimator):
    def _g(self, out_res, in_res, theta, dir, with_onsager=False, mode="map"):
        out_res, in_res, theta = self._val_g_inputs(out_res, in_res, theta, dir, mode)

        if (mode == "map"):
            gm_out, gm_in = theta
            zero = torch.zeros_like(in_res)

            cand_min_p = torch.maximum(zero, (gm_out*out_res+gm_in*in_res)/(gm_out+gm_in))
            cand_min_p_cost = (gm_out/2) * (cand_min_p - out_res)**2 + (gm_in/2) * (cand_min_p - in_res)**2

            cand_min_n = torch.minimum(zero, in_res)
            cand_min_n_cost = (gm_out/2) * out_res**2 + (gm_in/2) * (cand_min_n - in_res)**2

            take_p_huh = (cand_min_p_cost < cand_min_n_cost)
            z_est_in = cand_min_p * take_p_huh + cand_min_n * (1-take_p_huh)
            z_est_out = torch.relu(z_est_in)

            if(with_onsager):
                if(dir == "fwd"):
                    # onsager is average of grad wrt out residual
                    onsager = torch.mean(gm_out * (out_res - z_est_out)) # todo: the reference code computes negative of this. Which is correct?
                    return z_est_out, onsager
                elif(dir == "bck"):
                    onsager = torch.mean(gm_in * (in_res - z_est_in))
                    return z_est_in, onsager
            else:
                if (dir == "fwd"):
                    return z_est_out
                elif (dir == "bck"):
                    return z_est_in
        elif(mode == "mmse"):
            raise NotImplementedError("mmse is not implemented for ReLU.")

class Linear(LayerEstimator):
    def __init__(self, mode, A, b, in_dim, out_dim, measurements=None, measurement_prec=None):
        super().__init__(mode, in_dim, out_dim, measurements, measurement_prec)
        # A is a matrix, b is a bias
        self.A = A
        self.b = b
        self.AtA = self.A.T @ self.A

    def _g(self, out_res, in_res, theta, dir, with_onsager=False, mode="map"):
        out_res, in_res, theta = self._val_g_inputs(out_res, in_res, theta, dir, mode)

        in_dim = len(in_res)

        # note that map and mmse formulations are equivalent, so mode doesn't matter.
        gm_out, gm_in = theta

        '''
        # the MAP estimate of z_in is M^{-1} d where M is like vI + W^T W, d is like a residual
        M = gm_in * torch.eye(in_dim) + self.AtA
        d = gm_in * in_res + gm_out * self.A.T @ (out_res - self.b)

        z_est_in = torch.linalg.solve(d, M)
        z_est_out = self.layer.forward(z_est_in)
        '''

        # todo: use one linear layer est class and subclass it with conv, bilinear ups, etc layers
        F = LinearLSQROp(self.A, self.b, out_res, in_res, gm_out, gm_in)
        r = F.target_vec()
        z = scipy.sparse.linalg.lsqr(F, r)[0]
        # error: on the second fwd pass, the solver increased the norm while trying to minimize dist from 0 and from a small norm thing. it's far from both. ?
        # probably an implementation error. I suspect the issue is, the linear layer should only be doing g- iters. And a gaussian prior should be doing g+ iters.
        z_est_in, z_est_out = z, self.A @ z + self.b

        if (with_onsager):
            if (dir == "fwd"):
                # onsager is average of grad wrt out residual
                # todo: does this expression agree with the divergence of M^{-1} b as a function of out_res?
                #       (this expr is derived via differentiating cost at an optimizer)
                onsager = np.mean(gm_out * (out_res - z_est_out))
                return z_est_out, onsager
            elif (dir == "bck"):
                onsager = np.mean(gm_in * (in_res - z_est_in))
                return z_est_in, onsager
        else:
            if (dir == "fwd"):
                return z_est_out
            elif (dir == "bck"):
                return z_est_in

class Conv2d(LayerEstimator):
    def __init__(self, mode, conv, measurements=None, measurement_prec=None, transpose=False):
        super().__init__(mode, conv, measurements, measurement_prec)
        self.transpose = transpose

    def _g(self, out_res, in_res, theta, dir, with_onsager=False, mode="map"):
        out_res, in_res, theta = self._val_g_inputs(out_res, in_res, theta, dir, mode)
        gm_out, gm_in = theta

        F = ConvLSQROp(self.conv, out_res, in_res, gm_out, gm_in, transpose=self.transpose)
        r = F.target_vec()
        z = scipy.sparse.linalg.lsqr(F, r)[0]
        z_est_in, z_est_out = F.split(z)

        if (with_onsager):
            if (dir == "fwd"):
                onsager = torch.mean(gm_out * (out_res - z_est_out))
                return z_est_out, onsager
            elif (dir == "bck"):
                onsager = torch.mean(gm_in * (in_res - z_est_in))
                return z_est_in, onsager
        else:
            if (dir == "fwd"):
                return z_est_out
            elif (dir == "bck"):
                return z_est_in

class BilinearUpsample(LayerEstimator):
    def __init__(self, mode, in_HW, out_HW, measurements=None, measurement_prec=None):
        super().__init__(mode, conv, measurements, measurement_prec)
        self.in_HW = in_HW
        self.out_HW = out_HW

    def _g(self, out_res, in_res, theta, dir, with_onsager=False, mode="map"):
        out_res, in_res, theta = self._val_g_inputs(out_res, in_res, theta, dir, mode)
        gm_out, gm_in = theta

        F = BilinearUpsLSQROp(self.in_HW, self.out_HW, out_res, in_res, gm_out, gm_in)
        r = F.target_vec()
        z = scipy.sparse.linalg.lsqr(F, r)[0]
        z_est_in, z_est_out = F.split(z)

        if (with_onsager):
            if (dir == "fwd"):
                onsager = torch.mean(gm_out * (out_res - z_est_out))
                return z_est_out, onsager
            elif (dir == "bck"):
                onsager = torch.mean(gm_in * (in_res - z_est_in))
                return z_est_in, onsager
        else:
            if (dir == "fwd"):
                return z_est_out
            elif (dir == "bck"):
                return z_est_in


class LinearLSQROp(scipy.sparse.linalg.LinearOperator):
    def __init__(self, weight, bias, out_res, in_res, gm_out, gm_in):
        def colvec(x):
            return x.reshape([-1, 1])

        self.weight = weight
        self.bias = colvec(bias)
        self.in_res = colvec(in_res)
        self.out_res = colvec(out_res)
        self.in_dim = len(in_res)
        self.out_dim = len(out_res)
        self.sgm_in = np.sqrt(gm_in)
        self.sgm_out = np.sqrt(gm_out)
        self.shape = (self.in_dim + self.out_dim, self.in_dim)

        '''
        Designed to solve the prox least squares problem v{l-1}/2 | z{l-1} - r{l-1} |^2 + vl/2 |zl - rl|^2 
        given r{l-1}, rl, v{l-1}, vl. This is equivalent minimizing the norm of the vector difference 
        
        | v'{l-1}I  | | z{l-1} | - | v'{l-1} r{l-1} |
        | v'lA      |              | v'l rl         |
        
        in which v' = sqrt(v)
        '''

    def split(self, y):
        first = y[:self.in_dim]
        second = y[self.in_dim:]
        return first, second

    def target_vec(self):
        c= np.concatenate([self.sgm_in * self.in_res, self.sgm_out * (self.out_res - self.bias)], axis=0)
        return c

    def _matvec(self, in_z):
        return np.concatenate([self.sgm_in * in_z, self.sgm_out * self.weight @ in_z], axis=0)
        # Ah! This is the issue. Rather than doing shrinkage, it's treating r0- as an output of the layer, and trying to find an input near zero whose output is like r0-.
        # the solution is to make a gaussian prior node which is what ought to be doing the g0+ iterations.
        # SO the flow should be, forward passes shrink the est of z0, then backward passes pull it towards LS sol of Az = b.

    def _rmatvec(self, y):
        y = y.reshape([-1, 1])
        in_res, out_res = self.split(y)
        assert len(out_res) == self.out_dim, "dimensionality mismatch in LSQR input vector."
        return self.sgm_in * in_res + self.sgm_out * self.weight.T @ out_res

class BilinearUpsLSQROp(scipy.sparse.linalg.LinearOperator):
    def __init__(self, in_HW, out_HW, out_res, in_res, gm_out, gm_in):
        def colvec(x):
            return x.reshape([-1, 1])

        self.layer = torch.nn.Upsample(size=out_HW, mode='bilinear', align_corners=True)
        # bilinear upsampling implements a sparse structured matrix multiplication
        # here we compute the transpose by computing columns by applying the matrix to a basis
        basis = torch.eye(in_HW[0] * in_HW[1]).reshape((1, -1, in_HW[0], in_HW[1])).to('cpu')
        img_of_basis = self.layer(basis).detach().cpu().numpy().reshape((in_HW[0] * in_HW[1], out_HW[0] * out_HW[1]))
        upsT_operator = torch.sparse_coo_tensor( np.argwhere(img_of_basis != 0).T, img_of_basis[img_of_basis != 0], img_of_basis.shape).to('cpu')

        in_C = len(in_res)
        out_C = len(out_res)
        self.upsT_operator = upsT_operator
        self.img_of_basis = img_of_basis.T
        self.img_of_basis = self.layer(basis)
        self.in_res = colvec(in_res)
        self.out_res = colvec(out_res)
        self.in_shape = (in_C, in_HW[0], in_HW[1])
        self.out_shape = (out_C, out_HW[0], out_HW[1])
        self.in_HW = in_HW
        self.out_HW = out_HW
        self.sgm_in = np.sqrt(gm_in)
        self.sgm_out = np.sqrt(gm_out)
        self.in_dim = np.prod(self.in_shape)
        self.out_dim = np.prod(self.out_shape)
        self.shape = (self.out_dim, self.in_dim)

    def stack(self, in_im, out_im):
        return np.concatenate([in_im.flatten(), out_im.flatten()], axis=0)

    def split(self, y):
        first = y[:self.in_dim]
        second = y[self.in_dim:]
        return first, second

    def target_vec(self):
        return np.concatenate([self.sgm_in * self.in_res, self.sgm_out * (self.out_res - self.bias)], axis=0)

    def _matvec(self, in_z):
        # given an image in flattened vector format of size in_dim
        def tensorize(z):
            return torch.FloatTensor(z.reshape(self.in_shape)[None, ...]).to('cpu')
        ups_in_z = self.layer.forward(tensorize(in_z))[0].detach().cpu().numpy()
        return self.stack(self.sgm_in * in_z, self.sgm_out * ups_in_z)

    def _rmatvec(self, y):
        # y is the stack of two input and output images in vector format.
        in_res, out_res = self.split(y)
        assert len(out_res) == self.out_dim, "dimension mismatch on channel dimension in the linear convolution operator"

        out_res = torch.FloatTensor(out_res.reshape((self.out_shape[0], -1))).to('cpu')

        t_ups_out_res = torch.sparse.mm(self.upsT_operator, out_res.T).T

        return self.stack(self.sgm_in * in_res, self.sgm_out * t_ups_out_res)

class ConvLSQROp(scipy.sparse.linalg.LinearOperator):
    def __init__(self, conv, out_res, in_res, gm_out, gm_in, transpose=False):
        # inputs in_res and out_res are assumed to be C x H x W images.
        # they will be concatenated on the channel dimension.
        # todo: watch for errors related to the fact that input and output vectors are not vector shaped.
        def colvec(x):
            return x.reshape([-1, 1])

        self.bias = conv.bias
        if(transpose):
            self.conv = torch.nn.ConvTranspose2d(in_channels=conv.in_channels,
                                        out_channels=conv.out_channels,
                                        kernel_size=conv.kernel_size,
                                        padding=conv.padding,
                                        bias=False).to('cpu')
            self.transpose_conv = torch.nn.Conv2d(in_channels=conv.out_channels,
                                                           out_channels=conv.in_channels,
                                                           kernel_size=conv.kernel_size,
                                                           padding=conv.padding,
                                                           bias=False).to('cpu')
            self.conv.weight.data = conv.weight
            self.transpose_conv.weight.data = conv.weight
        else:
            self.conv = torch.nn.Conv2d(in_channels=conv.in_channels,
                                        out_channels=conv.out_channels,
                                        kernel_size=conv.kernel_size,
                                        padding=conv.padding,
                                        bias=False).to('cpu')
            self.transpose_conv = torch.nn.ConvTranspose2d(in_channels=conv.out_channels,
                                                           out_channels=conv.in_channels,
                                                           kernel_size=conv.kernel_size,
                                                           padding=conv.padding,
                                                           bias=False).to('cpu')
            self.conv.weight.data = conv.weight
            self.transpose_conv.weight.data = conv.weight

        self.in_res = colvec(in_res)
        self.out_res = colvec(out_res)
        self.in_c = len(in_res)
        self.out_c = len(out_res)
        self.in_shape = np.prod(in_res.shape())
        self.out_shape = np.prod(out_res.shape())
        self.sgm_in = np.sqrt(gm_in)
        self.sgm_out = np.sqrt(gm_out)

        self.shape = (self.out_shape, self.in_shape) # !!!
    def split(self, y):
        first = y[:self.in_c]
        second = y[self.in_c:]
        return first, second

    def target_vec(self):
        return np.concatenate([self.sgm_in * self.in_res, self.sgm_out * (self.out_res - self.bias)], axis=0)

    def _matvec(self, in_z):
        def tensorize(z):
            return torch.FloatTensor(z[None, ...]).to('cpu')
        conv_at_in_z = self.conv.forward(tensorize(in_z))[0].detach().cpu().numpy()
        return np.concatenate([self.sgm_in * in_z, self.sgm_out * conv_at_in_z], axis=0)

    def _rmatvec(self, y):
        in_res, out_res = self.split(y)
        assert len(out_res) == self.out_dim, "dimension mismatch on channel dimension in the linear convolution operator"
        def tensorize(z):
            return torch.FloatTensor(z[None, ...]).to('cpu')
        t_conv_at_out_res = self.transpose_conv.forward(tensorize(out_res))[0].detach().cpu().numpy()
        return np.concatenate([self.sgm_in * in_res, self.sgm_out * t_conv_at_out_res], axis=0)

if __name__ == "__main__":
    out_res = np.zeros((3, 10, 10))
    in_res = np.zeros((6, 10, 10))
    out_res_2 = np.zeros((6, 20, 20))

    out_res[0, 0, 0] = 1
    out_res_2[1, 1, 1] = 1
    in_res[1, 1, 1] = 1

    conv_layer = torch.nn.Conv2d(in_channels=6, out_channels=3, kernel_size=(3, 3), padding=(1, 1))
    conv = ConvLSQROp(conv_layer, out_res, in_res, 1, 1)
    ups = BilinearUpsLSQROp((10, 10), (20, 20), out_res_2, in_res, 1, 1)

    join = np.concatenate([in_res, out_res], axis=0)
    _, conv_trns_out_res = conv.split(conv._rmatvec(join))
    _, conv_in_res = conv.split(conv._matvec(in_res))

    _, trns_out_res = ups.split(ups._rmatvec(ups.stack(in_res, out_res_2)))
    _, ups_in_res = ups.split(ups._matvec(in_res.flatten()))

    print("Conv2D")
    print(np.dot(in_res.flatten(), conv_trns_out_res.flatten()))
    print(np.dot(conv_in_res.flatten(), out_res.flatten()))
    print("Upsample")
    print(np.dot(in_res.flatten(), trns_out_res.flatten()))
    print(np.dot(ups_in_res.flatten(), out_res_2.flatten()))

