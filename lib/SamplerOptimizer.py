import torch
import os
import torch.nn.modules.module as torchmodules

class WSamplerOptimizer():
    def __init__(self, sampler, critic, critic_steps):
        self.sampler = sampler
        self.sampler_opt = torch.optim.RMSprop(sampler.parameters(), lr=0.0005) # 0.00005
        self.critic = critic
        self.critic_opt = torch.optim.RMSprop(critic.parameters(), lr=0.0005) # 0.0005
        self.critic_steps = critic_steps

    def step(self, outp_batch, z_batch):
        for _ in range(self.critic_steps):
            self.critic_opt.step(lambda: self._critic_closure(outp_batch, z_batch))
            self._clip_weights(self.critic, 0.01)
        return self.sampler_opt.step(lambda: self._sampler_closure(outp_batch, z_batch))

    def _sampler_closure(self, outp_batch, z_batch):
        self.sampler_opt.zero_grad()

        crit_fake = self.critic(self.sampler(z_batch))
        crit_real = self.critic(outp_batch)

        obj = torch.mean(crit_real) - torch.mean(crit_fake)
        obj.backward()
        return obj

    def _critic_closure(self, outp_batch, z_batch):
        self.critic_opt.zero_grad()

        crit_fake = self.critic(self.sampler(z_batch))
        crit_real = self.critic(outp_batch)

        obj = torch.mean(crit_real) - torch.mean(crit_fake)
        (-obj).backward() # for gradient ascent
        return obj

    def _clip_weights(self, net, tol):
        try:
            net.clip_weights(tol)
        except AttributeError:
            net.module.clip_weights(tol)
    def save(self, path):
        torch.save(self.critic.state_dict(), os.path.join(path, 'critic.pt'))

    def load(self, path):
        try:
            self.critic.load_state_dict(torch.load(os.path.join(path, f"critic.pt")))
        except FileNotFoundError:
            print("Starting from randomly initialized critic.")

class WGPSamplerOptimizer(WSamplerOptimizer):
    def __init__(self, sampler, critic, critic_steps):
        super().__init__(sampler, critic, critic_steps)
        self.gp_weight = 10
        self.lr = 0.0001
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0, 0.9)) # 0.0005
        self.sampler_opt = torch.optim.Adam(self.sampler.parameters(), lr=self.lr, betas=(0, 0.9)) # 0.00005

    def step(self, outp_batch, z_batch):
        self.critic_opt.step(lambda: self._critic_closure(outp_batch, z_batch))
        return self.sampler_opt.step(lambda: self._sampler_closure(outp_batch, z_batch))

    def _sampler_closure(self, outp_batch, z_batch):
        self.sampler_opt.zero_grad()

        crit_fake = self.critic(self.sampler(z_batch))
        crit_real = self.critic(outp_batch)

        obj = torch.mean(crit_real) - torch.mean(crit_fake)
        obj.backward()
        return obj

    def _critic_closure(self, outp_batch, z_batch):
        self.critic_opt.zero_grad()

        samples = self.sampler(z_batch)

        bs, outp_batch_order = outp_batch.shape[0], len(outp_batch.shape[1:])

        gp_mix_weights = torch.rand(size=[bs] + outp_batch_order*[1], device=outp_batch.device)
        gp_mix_outp = gp_mix_weights * outp_batch + (1-gp_mix_weights) * samples
        grad = torch.cat(torch.autograd.grad(outputs=list(self.critic(gp_mix_outp)), inputs=(gp_mix_outp),
                                   create_graph=True, retain_graph=True),
                         dim=1)

        grad_norms = torch.sqrt(torch.sum(grad**2, dim=list(range(1, outp_batch_order + 1))) + 1e-12)
        gp = self.gp_weight * torch.mean( (grad_norms - torch.ones((bs,), device=outp_batch.device) )**2)
        crit_fake = self.critic(samples)
        crit_real = self.critic(outp_batch)

        obj = torch.mean(crit_real) - torch.mean(crit_fake)
        (gp - obj).backward() # for gradient ascent
        return obj

class DCGANSamplerOptimizer():
    def __init__(self, sampler, discriminator):
        self.sampler = sampler
        self.sampler_opt = torch.optim.Adam(sampler.parameters(), lr=0.00002)
        self.discriminator = discriminator
        self.discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=0.00002)
        self.global_itr = 0

    def step(self, outp_batch, z_batch):
        self.discriminator_opt.step(lambda: self._disc_closure(outp_batch, z_batch))
        cur_loss = self.sampler_opt.step(lambda: self._sampler_closure(outp_batch, z_batch))
        return cur_loss

    def _sampler_closure(self, outp_batch, z_batch):
        self.sampler_opt.zero_grad()
        disc_fake = self.discriminator(self.sampler(z_batch))
        disc_real = self.discriminator(outp_batch)

        disc_real_prob = torch.mean( -torch.log1p(torch.exp(-disc_real)))
        disc_fake_prob = torch.mean(-disc_fake - torch.log1p(torch.exp(-disc_fake)))
        obj = disc_real_prob - disc_fake_prob
        obj.backward()
        return obj

    def _disc_closure(self, outp_batch, z_batch):
        self.sampler_opt.zero_grad()
        disc_fake = self.discriminator(self.sampler(z_batch))
        disc_real = self.discriminator(outp_batch)

        disc_real_prob = torch.mean( -torch.log1p(torch.exp(-disc_real)))
        disc_fake_prob = torch.mean(-disc_fake - torch.log1p(torch.exp(-disc_fake)))
        obj = disc_real_prob - disc_fake_prob
        (-obj).backward()
        return obj

    def save(self, path):
        torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pt'))

    def load(self, path):
        try:
            self.discriminator.load_state_dict(torch.load(os.path.join(path, f"discriminator.pt")))
        except FileNotFoundError:
            print("Starting from randomly initialized discriminator.")

