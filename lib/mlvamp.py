import torch
import numpy as np


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
    def __init__(self, mode, layer, measurements=None, measurement_prec=None, latent_prec=None):
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
        self.layer = layer
        self.measurements = measurements
        self.measurements_prec = measurement_prec
        self.latent_prec = latent_prec
        self.mode = mode

    def register_neighbors(self, incoming, outgoing):
        self.incoming = incoming
        self.outgoing = outgoing

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
            fwd_theta = (self.params["gm-"], self.incoming["gm+"])
            fwd_z, fwd_a = self._g(self.params["r-"], self.incoming["r+"],
                                   fwd_theta, dir="fwd", with_onsager=True, mode=self.mode)
            fwd_r = (fwd_z - fwd_a * self.params["r-"]) / (1 - fwd_a)

            self.params["z+"] = fwd_z
            self.params["r+"] = fwd_r
            self.params["e+"] = self.params["gm-"] / fwd_a
            self.params["gm+"] = self.params["e+"] - self.params["gm-"]
            self.params["t+"] = fwd_theta
        elif(dir == "bck"):
            bck_z, bck_a = self._g(self.outgoing["r-"], self.params["r+"],
                                   self.outgoing["t-"], dir="bck", with_onsager=True, mode=self.mode)
            bck_r = (bck_z - bck_a * self.params["r+"]) / (1 - bck_a)
            bck_eta = self.params['gm+'] / bck_a
            bck_gamma = bck_eta - self.params["gm+"]
            bck_theta = (bck_gamma, self.incoming["gm+"])

            self.params["z-"] = bck_z
            self.params["r-"] = bck_r
            self.params["e-"] = bck_eta
            self.params["gm-"] = bck_gamma
            self.params["t-"] = bck_theta
        else:
            raise ValueError(f"dir '{dir}' not recognized. Must be one of 'fwd' or 'bck'.")

    def _g(self, in_res, out_res, theta, dir, with_onsager=False, mode="map"):
        raise NotImplementedError("Subclasses implement this method.")

    def __delitem__(self, key):
        self.params.__delattr__(key)

    def __getitem__(self, key):
        return self.params.__getattribute__(key)

    def __setitem__(self, key, value):
        self.params.__setattr__(key, value)

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
        return self.params.__getattribute__(key)

    def __setitem__(self, key, value):
        self.params.__setattr__(key, value)

class ReLULayerEstimator(LayerEstimator):
    def _g(self, out_res, in_res, theta, dir, with_onsager=False, mode="map"):
        '''
        Using input and output residuals, compute a denoised estimate of this layers activation.
        Return the denoised estimate.

        Args:
            out_res (torch.Tensor or None): the output residual.
            in_res (torch.Tensor or None): the input residual.
            theta (torch.Tensor): noise estimate.
            dir (string): either "fwd" or "bck"
            with_onsager: if True, returns a tuple (z, a) where z is the denoised estimate and a is the
                corresponding Onsager term.
            mode: either "map" for maximum a posteriori estimation or "mmse" for minimum mean squared error
                estimation.

        Returns (torch.Tensor or (torch.Tensor, torch.Tensor)): either z or (z, a) where z is the
            denoised estimate and a is the corresponding Onsager term.
        '''

        # cases:
        # - in res, theta[1] is None
        # - out res, theta[0] is None
        # - neither are none
        assert mode in ["map", "mmse"], f"Mode {mode} not recognized. Choose 'map' or 'mmse'."
        assert dir in ["fwd", "back"], f"Direction {dir} not recognize. Choose 'fwd' or 'bck'."
        def given(x):
            return (x is not None)

        if(not (given(in_res) or given(theta[1])) and given(out_res) and given(theta[0])):
            assert dir == "fwd", "Cannot run backward iteration at first layer."
            in_res = torch.zeros_like(out_res)
            theta[1] = self.latent_prec
            if(not (given(in_res) or given(theta[1])) ):
                raise ValueError("Attempt to use input layer without passing input prior information.")
        elif(not (given(out_res) or given(theta[0])) and given(in_res) and given(theta[1])):
            assert dir == "bck", "Cannot run forward iteration at last layer."
            out_res = self.measurements
            theta[0] = self.measurements_prec
            if(not (given(out_res) or given(theta[0])) ):
                raise ValueError("Attempt to use output layer without passing output prior information.")
        else:
            raise ValueError("Invalid combination of residual, theta inputs to ReLU proximal denoiser.")

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
        
