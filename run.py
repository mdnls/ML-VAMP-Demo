import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import yaml
import os
import shutil
from lib.nets import ImageCritic, NetEnsemble, Compatibility, FCN_Sampler, CNN_Sampler
from lib.distributions import Gaussian, TruncatedGaussian, Uniform, ProductDistribution, ImageDataset, DataGaussianApprox
import numpy as np
import scipy.stats as stats
import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from lib.SamplerOptimizer import WSamplerOptimizer, DCGANSamplerOptimizer, WGPSamplerOptimizer

class MNISTNetEnsemble(NetEnsemble):
    def __init__(self, device, transport_cost, outp_wc=(28, 1), inp_dim=100, sampler="FCN", gan="WGP", regularization="entropy", reg_strength=0.1):
        outp_width, outp_channels = outp_wc

        def gimme_critic(layernorm=True):
            return ImageCritic(input_im_size=outp_width, layers=4, channels=[outp_wc[1], 64, 128, 256, 512], layernorm=layernorm).to(device)

        self.device = device
        if(sampler == "CNN"):
            self.sampler = CNN_Sampler(inp_dim=100, outp_wc=(28, 1), layers=4, channels=[100, 256, 256, 256, 1]).to(device)
        elif(sampler == "FCN"):
            self.sampler = FCN_Sampler(inp_dim, outp_wc, [2048, 4096, 8192, 8192]).to(device)
        else:
            raise ValueError(f"{sampler} is not a recognized GAN architecture.")

        if(gan == "WGP"):
           self.sampler_opt = WGPSamplerOptimizer(self.sampler, gimme_critic(layernorm=False), critic_steps=5)
        elif(gan == "W"):
            self.sampler_opt = WSamplerOptimizer(self.sampler,  gimme_critic(), critic_steps=5)
        elif(gan == "DCGAN"):
            self.sampler_opt = DCGANSamplerOptimizer(self.sampler, gimme_critic())
        else:
            raise ValueError(f"{gan} is not a recognized GAN training scheme.")

        self.inp_density_parameter = gimme_critic()
        self.outp_density_parameter = gimme_critic()
        self.cpat = Compatibility(self.inp_density_parameter, self.outp_density_parameter,
                                  regularization=regularization, transport_cost=transport_cost,
                                  reg_strength=reg_strength).to(device)

    def save(self, path):
        self.save_net(self.sampler, os.path.join(path, f"sampler.pt"))
        self.save_net(self.inp_density_parameter, os.path.join(path, f"inp_density_parameter.pt"))
        self.save_net(self.outp_density_parameter, os.path.join(path, f"outp_density_parameter.pt"))
        self.sampler_opt.save(path)

    def load(self, path):
        self.load_net(self.sampler, os.path.join(path, f"sampler.pt"))
        self.load_net(self.inp_density_parameter, os.path.join(path, f"inp_density_parameter.pt"))
        self.load_net(self.outp_density_parameter, os.path.join(path, f"outp_density_parameter.pt"))
        self.sampler_opt.load(path)

def run(outp_batch_iter, z_batch_iter, net_ensemble, opt_iter_schedule, artifacts_path, device):
    '''
    Arguments:
        - outp_batch_iter: an iterator which produces batches of data from the output distribution Q
        - z_batch_iter: an iterator which produces batches of data from the latent code distribution
        - net_ensemble: a NetEnsemble class which provides the sampler, critic, and discriminator networks.
        - opt_iter_schedule: a tuple of two integers (D, G). Run D steps of training the OT plan conditional density
            and then run G steps of training the generator.
        - reg_strength: the regularization parameter. As lambda increases the regularization strength decreases.
        - artifacts_path: path to a folder where experimental data is saved
    '''

    assert (len(opt_iter_schedule) == 2) and all([type(o) == int for o in opt_iter_schedule]),\
        "opt_iter_schedule must contain 2 integers"

    density_loops, sampler_loops = opt_iter_schedule

    sampler, cpat = net_ensemble.sampler, net_ensemble.cpat
    sampler_opt_manager = net_ensemble.sampler_opt

    cpat_opt = torch.optim.Adam(params = cpat.parameters(), lr=0.000001)
    cpat_opt_lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(cpat_opt, mode='max', factor=0.5, patience=3, threshold=0.01, verbose=True)
    size_of_epoch=100

    def cpat_closure(inp_sample, outp_sample):
        cpat_opt.zero_grad()
        density_real_inp = cpat.inp_density_param(inp_sample)
        density_real_outp = cpat.outp_density_param(outp_sample)
        density_reg = cpat.penalty(inp_sample, outp_sample)
        obj = torch.mean(density_real_inp + density_real_outp - density_reg)
        (-obj).backward() # for gradient ascent rather than descent
        return obj

    writer = SummaryWriter(log_dir=artifacts_path)

    t_ex_z_sample = torch.FloatTensor(next(z_batch_iter)[:50]).to(device)
    t_ex_outp_sample = torch.FloatTensor(next(outp_batch_iter)[:50]).to(device)

    ex__outp_grid = torchvision.utils.make_grid( (t_ex_outp_sample+1)/2)
    writer.add_image('Example Outputs', ex__outp_grid)

    def new_batch(outp_batch_iter, z_batch_iter, device):
        def _safe_sample(itr):
            try:
                return itr, torch.FloatTensor(next(itr)).to(device)
            except StopIteration:
                fresh_itr = iter(itr)
                return fresh_itr, torch.FloatTensor(next(fresh_itr)).to(device)
        outp_batch_iter, outp_sample = _safe_sample(outp_batch_iter)
        z_batch_iter, z_sample = _safe_sample(z_batch_iter)
        return outp_sample, z_sample


    sum_obj = torch.zeros(size=(1,)).to(device)
    for d_step in range(density_loops):
        outp_sample, z_sample = new_batch(outp_batch_iter, z_batch_iter, device)
        obj = cpat_opt.step(lambda: cpat_closure(z_sample, outp_sample))

        avg_density = torch.mean(cpat.forward(z_sample, outp_sample))

        obj_val = round(obj.item(), 5)
        avg_density_val = round(avg_density.item(), 5)
        print(f"\rO{d_step} - Density Loss: {obj_val} - Average Density: {avg_density_val}", end="")
        writer.add_scalars('Optimization', {
            'Objective': obj_val,
            'Average Density': avg_density_val
        }, d_step)

        sum_obj += obj
        if(d_step % size_of_epoch == size_of_epoch - 1):
            avg_obj = sum_obj / size_of_epoch
            sum_obj = torch.zeros(size=(1,)).to(device)
            cpat_opt_lr_sched.step(avg_obj)

        if(d_step % 100 == 0):
            single_inp_batch = np.tile( z_sample[0:1].detach().cpu().numpy(), (outp_sample.shape[0], 1, 1, 1))
            single_inp_batch = torch.FloatTensor(single_inp_batch).to(device)
            cpats = cpat(single_inp_batch, outp_sample)
            sorted_imgs = [(x[0]+1)/2 for x in sorted(zip(outp_sample, cpats), key=lambda x: x[1])]
            img_grid = torchvision.utils.make_grid(sorted_imgs)
            writer.add_image('Output images sorted by cpat to fixed latent', img_grid, d_step)

    for s_step in range(sampler_loops):
        outp_sample, z_sample = new_batch(outp_batch_iter, z_batch_iter, device)
        s = sampler_opt_manager.step(outp_sample, z_sample)
        s_val = round(s.item(), 5)
        print(f"\rO{s_step} - Sampler: {s_val}", end="")
        writer.add_scalars('Optimization', {
            'Sampler': s_val
        }, s_step)
        if(s_step % 500 == 0):
            with torch.no_grad():
                samples = sampler(t_ex_z_sample)
            net_ensemble.save(path)
            img_grid1 = torchvision.utils.make_grid((torch.clamp(samples, -1, 1) + 1 )/2)
            writer.add_image('Samples', img_grid1, s_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Help text.")
    parser.add_argument("-n", "--name", type=str, help="Name of this experiment.")
    parser.add_argument("-y", "--yaml", type=str, help="Path of a yaml configuration file to use. If provided, this config will overwrite any arguments.")
    parser.add_argument("--overwrite", action="store_true", help="If true, overwrite previous experimental results with the same name")
    parser.add_argument("-d", "--dataset", type=str, help="Choose a dataset: usps-mnist, svhn-mnist")
    parser.add_argument("--use_cpu", action="store_true", help="If true, train on CPU.")
    parser.add_argument("-b", "--batch_size", type=int, default=50, help="Batch size for training the density and generator.")
    parser.add_argument("-rs", "--reg_strength", type=float, default=0.1, help="Regularization strength.")
    parser.add_argument("--density_steps", type=int, default=500, help="Steps to train the density estimator.")
    parser.add_argument("--sampler_steps", type=int, default=500, help="Steps to train the sampler.")
    parser.add_argument("-r", "--regularization", type=str, default="entropy", help="Either l2 or entropy regularization.")
    parser.add_argument("--parallel", action="store_true", help="Parallelize over multiple GPUs.")
    parser.add_argument("--load_from", type=str, help="Path to load a learned model from.", default="")
    parser.add_argument('--sampler', type=str, help="Choose the sampler optimization strategy")
    parser.add_argument('--gan', type=str, help="Choose the GAN optimizer.")
    parser.add_argument('--dparam', type=str, help="Dparam architecture", default="imagecritic")
    args = parser.parse_args()

    args.device = 'cpu' if args.use_cpu else 'cuda'

    if args.yaml is not None:
        try:
            with open(args.yaml) as yaml:
                args = yaml.safe_load(yaml)
        except FileNotFoundError:
            raise FileNotFoundError("Could not load the provided yaml file.")

    path = os.path.join("artifacts", args.name)
    if(os.path.exists(path)):
        if(args.overwrite):
            shutil.rmtree(path)
        else:
            path += "_1"
            i = 1
            while os.path.exists(path):
                i = i + 1
                path = path.split("_")[0] + "_" + str(i)

    os.makedirs(path)

    args.yaml = os.path.join(path, "config.yml")

    with open(args.yaml, "w+") as f_out:
        yaml.dump(vars(args), f_out, default_flow_style=False, allow_unicode=True)

    bs = args.batch_size
    reg_strength = args.reg_strength
    if(args.dataset == "gaussian-mnist"):
        Z = Gaussian(mu=0, batch_size=bs, data_dim=100, sigma=1)
        Q = ImageDataset(path="data/mnist", batch_size=bs, im_size=28, channels=1)
        c = lambda x, y: torch.mean((x-y)**2, dim=(1, 2, 3))[:, None]
        net_ensemble = MNISTNetEnsemble(args.device, c, outp_wc=(28, 1), inp_dim=100,
                                        sampler=args.sampler,
                                        gan=args.gan,
                                        reg_strength=args.reg_strength,
                                        regularization=args.regularization)
    else:
        raise ValueError(f"'{args.dataset}' is an invalid choice of dataset.")

    net_ensemble.save(path)

    if(args.load_from != ""):
        net_ensemble.load(args.load_from)

    run(outp_batch_iter=Q,
        z_batch_iter=Z,
        net_ensemble=net_ensemble,
        opt_iter_schedule=(args.density_steps, args.sampler_steps),
        artifacts_path=path,
        device=args.device)
    net_ensemble.save(path)
