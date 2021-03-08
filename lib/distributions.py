import numpy as np
from scipy import stats
import torch
import torchvision
from sklearn import datasets
import scipy.stats


class Ellipse():
    def __init__(self, centroid, T, noise_std, batch_size):
        '''Sample points uniformly from the ellipse given by transorming the unit circle by the 2x2 matrix T
        and translating by adding T'''
        self.centroid = centroid.flatten()
        self.T = T
        self.batch_size = batch_size
        self.noise_std = noise_std
    def __iter__(self):
        return self
    def __next__(self):
        points = np.random.uniform(0, 2 * np.pi, size=self.batch_size)
        data = np.stack((np.cos(points), np.sin(points)), axis=1)
        return data @ self.T.T + self.centroid + np.random.normal(loc=0, scale=self.noise_std, size=(self.batch_size, 2))

class UnifEllipseMixture():
    def __init__(self, centroids, Ts, noise_std, batch_size):
        '''
        Sample points from a mixture of Gaussians each having uniform weights.
        Each batch is constructed by sampling an equal number of points from the Gaussians in the mixture.
        So, len(centroids)=len(Ts) must divide batch_size.
        '''
        self.batch_size = batch_size
        self.n_ellipses = len(centroids)
        self.ellipses = [Ellipse(centroids[i], Ts[i], noise_std, batch_size//self.n_ellipses) for i in range(self.n_ellipses)]
    def __iter__(self):
        return self
    def __next__(self):
        return np.random.permutation(np.concatenate([next(g) for g in self.ellipses], axis=0))

class RandomEllipseMixture():
    def __init__(self, centroid, n_ellipses, noise_std, batch_size):
        assert batch_size // n_ellipses == batch_size / n_ellipses
        Ts = [np.random.normal(loc=0, scale=1, size=(2, 2)) for _ in range(n_ellipses)]
        self.ellipses = UnifEllipseMixture(centroids=[centroid for _ in range(n_ellipses)], Ts=Ts, noise_std=noise_std, batch_size = batch_size)
    def __iter__(self):
        return self
    def __next__(self):
        return next(self.ellipses)

class DataGaussianApprox():

    def __init__(self, mu, cov, batch_size, im_WC=None):
        # TODO: merge to Gaussian
        data_dim = len(mu)
        self.data_dim = data_dim
        self.mu = mu
        self.cov = cov
        self.batch_size = batch_size
        self.im_WC = im_WC

        self.normal = scipy.stats.multivariate_normal(mean=mu, cov=cov, allow_singular=True)

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.normal.rvs(size=self.batch_size)
        if(self.im_WC is not None):
            return samples.reshape((self.batch_size, self.im_WC[1], self.im_WC[0], self.im_WC[0]))
        else:
            return samples

class Gaussian():
    def __init__(self, mu, sigma, batch_size, data_dim):
        '''
        Gaussian distribution.

        Arguments:
            - mu (np.ndarray): the mean of the distribution, same shape as data_dim
            - sigma (number): the standard deviation of each coordinate of the distribution
            - batch_size (int): number of samples per batch
            - data_dim (int) or list of ints: the dimensionality of output data
        '''
        if(type(data_dim) == int):
            data_dim = (data_dim,)
        self.mu = mu
        self.sigma = sigma
        self.batch_size = batch_size
        self.data_dim = data_dim

    def __iter__(self):
        return self

    def __next__(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size= (self.batch_size,) + self.data_dim)

class TruncatedGaussian(Gaussian):
    def __init__(self, mu, sigma, batch_size, data_dim, bound):
        '''
        Truncated Gaussian distribution.

        Arguments:
            - mu, sigma, batch_size, data_dim: see Gaussian.
            - bound: a scalar. Samples which deviate from the mean by a magnitude larger than this number are
                thrown out.
        '''
        super().__init__(mu, sigma, batch_size, data_dim)
        self.bound = bound
    def __iter__(self):
        return self
    def __next__(self):
        return stats.truncnorm.rvs(loc=self.mu, scale=self.sigma,
                                   size=(self.batch_size,) + self.data_dim,
                                   a=-self.bound, b=self.bound)

class LabeledDist():
    def next_labelled(self):
        '''
        Classes which subclass LabelledDist should implement next_labelled which should
            return a tuple (data, labels)
        '''
        raise NotImplementedError()

class TwoMoons(LabeledDist):
    def __init__(self, sigma, batch_size, scale=1):
        '''
        Sample data from a two moons dataset. The output data is 2 dimensional.

        Arguments:
            - sigma: noise added to points along the moon half circles.
            - batch_size: number of samples to output per batch.
        '''
        self.sigma = sigma
        self.batch_size = batch_size
        self.scale = scale
    def __iter__(self):
        return self

    def __next__(self):
        return self.scale * self.next_labelled()[0]

    def next_labelled(self):
        data, labels = datasets.make_moons(n_samples=self.batch_size, shuffle=True, noise=self.sigma)
        return (self.scale * data, labels)

    def from_moon(self, moon="top"):
        '''
        Sample a batch from a single moon.

        Arguments:
            - moon: either "top" or "bottom". Return samples from the specified moon.
        '''
        if(moon == "top"):
            return self.scale * datasets.make_moons(n_samples=(self.batch_size, 0), shuffle=True, noise=self.sigma)[0]
        elif(moon == "bottom"):
            return self.scale * datasets.make_moons(n_samples=(0, self.batch_size), shuffle=True, noise=self.sigma)[0]
        else:
            raise ValueError(f"{moon} is not a valid argument to TwoMoons().from_moon.")

class UnifGaussianMixture(LabeledDist):
    def __init__(self, mus, sigmas, batch_size, data_dim):
        '''
        Sample points from a mixture of Gaussians each having uniform weights.
        Each batch is constructed by sampling an equal number of points from the Gaussians in the mixture.
        So, len(mus)=len(sigmas) must divide batch_size.

        Arguments:
            - mus (np.ndarray or list of np.ndarray): means of gaussians in the mixture
            - sigmas (np.ndarray or list of np.ndarray): standard deviations of each coordinate gaussians in the mixture
            - batch_size (int): number of samples per batch
            - data_dim (int) or list of ints: the dimensionality of output data
        '''
        if(not (type(mus) in (list, tuple))):
            assert type(mus) is np.ndarray, "mus must be a numpy array or a list"
            assert type(sigmas) is np.ndarray, "sigmas must be a numpy array or a list"
            mus = [mus]
            sigmas = [sigmas]

        assert len(mus) == len(sigmas), "mus and sigmas must be the same length."
        assert batch_size // len(mus) == batch_size / len(mus), "The number of Gaussians must divide batch_size"
        self.batch_size = batch_size
        self.n_gaussians = len(mus)
        self.data_dim = data_dim
        self.gaussians = [Gaussian(mus[i], sigmas[i], batch_size//len(mus), data_dim) for i in range(self.n_gaussians)]

    def __iter__(self):
        return self

    def __next__(self):
        return np.random.permutation(np.concatenate([next(g) for g in self.gaussians], axis=0))

    def next_labelled(self):
        indices = np.random.permutation(np.arange(self.batch_size))
        labels = np.array(sum([self.batch_size * [i] for i in range(self.n_gaussians)], []))
        data = np.concatenate([next(g) for g in self.gaussians], axis=0)
        return (data[indices], labels[indices])

class MappedDataset(LabeledDist):
    def __init__(self, source_dist, sampler):
        '''
        Map one labelled dataset into another using sampler and preserving labels.

        Arguments:
            - source_dist (LabelledDist): a labelled dataset
            - sampler (nn.Module): a network whose forward function maps batches from source
                to batches from some output distribution.
        '''
        self.source_dist = source_dist
        self.sampler = sampler

    def __iter__(self):
        return self


    def __next__(self):
        return self.sampler(next(self.source_dist))

    def next_labelled(self):
        data, labels = self.source_dist.next_labelled
        return (self.sampler(data), labels)


class Uniform():
    def __init__(self, mu, batch_size, data_dim, bound):
        '''
        Uniform distribution.

        Arguments:
            - mu (number): the mean of the uniform distribution, same shape as data_dim
            - batch_size (int): number of samples per batch
            - data_dim (int) or list of ints: the dimensionality of output data
            - bound (number): each coordinate of each point is sampled from (mu - bound, mu + bound)
        '''
        if(type(data_dim) == int):
            data_dim = (data_dim,)
        self.mu = mu
        self.batch_size = batch_size
        self.data_dim = data_dim
        self.bound = bound
    def __iter__(self):
        return self
    def __next__(self):
        return stats.uniform.rvs(loc=self.mu - self.bound, scale=2 * self.bound,
                                   size=(self.batch_size,) + self.data_dim)

class ProductDistribution():
    def __init__(self, *args):
        '''
        Construct a product distribution whose output samples are tuples where each component
            is a sample from each input distribution.
        '''
        data_batch_sizes = [dist.batch_size for dist in args]
        assert all([dist.batch_size == args[0].batch_size for dist in args]),\
            "Product distribution requires input distributions whose batch size parameters are equal."
        self.dists = args
        self.batch_size = data_batch_sizes[0]

        if(all([type(dist.data_dim) == int for dist in self.dists])):
            # if all distributions in the product are over vectors, we can concatenate them to make a higher dim vector
            self.data_dim = sum([dist.data_dim for dist in self.dists])
        else:
            # if the distributions are not all vectors, they cannot necessarily be concatenated.
            self.data_dim = None
    def __iter__(self):
        return self
    def __next__(self):
        if(self.data_dim is None):
            raise ValueError("Cannot concatenate non-vector distributions. Use .next_tuple() instead.")
        return np.concatenate(self.next_tuple(), axis=-1)
    def next_tuple(self):
        return (next(dist) for dist in self.dists)


class ImageDataset():
    def __init__(self, path, batch_size, im_size, padding=0, channels=3):
        '''
        A distribution given by samples from a dataset of images. Each image is assumed to be square
        and normalized to [0, 1].

        Arguments:
            - path (string): path to a dataset of images
            - batch_size (int): number of images per batch
            - im_size (int): images will be square cropped and resized so their side length is im_size.
            - padding (int): images will be padded by this many 0 pixels on all sides. Padding is taken into account
                of im_size, ie. if im_size=32 and padding=3 the image is resized to 29px and padded by 3 afterwards.
            - channels (int): number of channels in output images
        '''
        assert type(im_size) == int, "Image size must be an integer"
        assert type(batch_size) == int, "Batch size must be an integer"
        assert type(path) == str, "Path must be a string"
        assert type(channels) == int and channels in [1, 3], "Input must be a greyscale or RGB image having 1 or 3 channels"

        self.path = path
        self.batch_size = batch_size
        self.im_size = im_size
        self.padding = padding
        self.data_dim = (3, im_size, im_size)
        self.channels = channels

        transforms = [
            torchvision.transforms.Resize(im_size - 2 * padding),
            torchvision.transforms.Pad(padding),
            torchvision.transforms.CenterCrop(im_size)
        ]

        if(self.channels == 1):
            transforms.append(torchvision.transforms.Grayscale(num_output_channels=1))

        # greyscale transformation must preceed the ToTensor() or pytorch will give an error.
        transforms.append(torchvision.transforms.ToTensor())

        transform = torchvision.transforms.Compose(transforms)
        dataset = torchvision.datasets.ImageFolder(root = self.path, transform = transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True, num_workers=0)

        self.dataset = dataset
        self.dataloader = dataloader
        self.dataloader_iter = iter(dataloader)

    def __iter__(self):
        self.dataloader_iter = iter(self.dataloader)
        return self
    def __next__(self):
        itr = 2 * next(self.dataloader_iter)[0] - 1
        if(itr.shape[0] < self.batch_size):
            raise StopIteration
        return itr


class FeatureTransform:
    def __init__(self, transf_net):
        '''
        A dataset transform to be used in a DataLoader. Computes output features for each input sample using transf_net.

        Arguments:
            - transf_net (nn.Module): a neural network.
        '''
        self.transf_net = transf_net

    def __call__(self, sample):
        data, labels = sample
        return self.transf_net(data), labels
