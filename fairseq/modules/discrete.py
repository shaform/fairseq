import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_token, n_embed, decay=0.99, eps=1e-8):
        super().__init__()

        self.n_token = n_token
        self.n_embed = n_embed
        self._decay = decay
        self._eps = eps

        self._embeds = nn.Embedding(self.n_token, self.n_embed)
        self._embeds.weight.requires_grad = False

        self.register_buffer('_ema_n', torch.ones(self.n_token))
        self.register_buffer('_ema_w', self._embeds.weight.clone())
        self._update_weights()

    def get_weights(self):
        return self._embeds.weight

    def get_distances(self, inputs):
        embeds = self.get_weights()
        dists = ((inputs**2).sum(dim=-1, keepdim=True) +
                 (embeds**2).sum(dim=-1)[None, :] -
                 2 * torch.matmul(inputs, embeds.t()))
        return dists

    def forward(self, inputs):

        size = inputs.size()
        inputs = inputs.view(-1, self.n_embed)

        # get quantized outputs
        dists = self.get_distances(inputs)
        indices = torch.argmin(dists, dim=-1)
        quantized_embs = self._embeds(indices)
        quantized = inputs + (quantized_embs - inputs).detach()

        rev_indices = torch.argmin(dists, dim=0)
        if self.training:
            self._update_ema(inputs, indices, rev_indices)

        # return original shape
        quantized = quantized.reshape(size)

        return quantized

    def _update_weights(self):
        embed_normed = self._ema_w / self._ema_n[:, None]
        self._embeds.weight.data = embed_normed.data

    def _update_ema(self, inputs, indices, rev_indices):
        # indices: [B]
        # rev_indices: [N]
        assert len(indices.size()) == 1
        assert len(rev_indices.size()) == 1
        bsz = indices.size()[0]

        n_sample = indices.size()[-1]
        ratio = 1.0 * self.n_token / n_sample

        # update counts
        one_hots = F.one_hot(indices, self.n_token).to(inputs.dtype) * ratio
        rev_one_hots = F.one_hot(rev_indices, bsz).to(inputs.dtype) / self.n_token

        batch_cnts = (one_hots + rev_one_hots.t())
        cnts = batch_cnts.sum(dim=0)
        self._ema_n.data = (self._ema_n.data * self._decay +
                            (1 - self._decay) * cnts)

        # update weights
        ws = batch_cnts.t().matmul(inputs)
        self._ema_w.data = (self._ema_w.data * self._decay +
                            (1 - self._decay) * ws.data)

        self._update_weights()


class VectorQuantizedDropout(nn.Module):
    def __init__(self, n_token, n_embed, decay=0.99, eps=1e-8):
        super().__init__()

        self.n_token = n_token
        self.n_embed = n_embed
        self._decay = decay
        self._eps = eps

        self._embeds = nn.Embedding(self.n_token, self.n_embed)
        self._embeds.weight.requires_grad = False

        self.register_buffer('_ema_n', torch.ones(self.n_token))
        self.register_buffer('_ema_w', self._embeds.weight.clone())
        self._update_weights()

    def get_weights(self):
        return self._embeds.weight

    def get_distances(self, inputs):
        embeds = self.get_weights()
        dists = ((inputs**2).sum(dim=-1, keepdim=True) +
                 (embeds**2).sum(dim=-1)[None, :] -
                 2 * torch.matmul(inputs, embeds.t()))
        return dists

    def forward(self, inputs, rho=1.):
        if not self.training:
            return inputs

        size = inputs.size()
        inputs = inputs.view(-1, self.n_embed)

        # get quantized outputs
        dists = self.get_distances(inputs)
        indices = torch.argmin(dists, dim=-1)
        quantized_embs = self._embeds(indices)
        # quantized = inputs + (quantized_embs - inputs).detach()
        quantized = quantized_embs

        # gating function
        gate_probs = rho * torch.ones(
            inputs.size()[0], dtype=inputs.dtype, device=inputs.device)
        gates = torch.bernoulli(gate_probs)[:, None]

        quantized = quantized * gates + (1. - gates) * inputs

        rev_indices = torch.argmin(dists, dim=0)
        self._update_ema(inputs, indices, rev_indices)

        # return original shape
        quantized = quantized.reshape(size)

        return quantized

    def _update_weights(self):
        embed_normed = self._ema_w / self._ema_n[:, None]
        self._embeds.weight.data = embed_normed.data

    def _update_ema(self, inputs, indices, rev_indices):
        # indices: [B]
        # rev_indices: [N]
        assert len(indices.size()) == 1
        assert len(rev_indices.size()) == 1
        bsz = indices.size()[0]

        n_sample = indices.size()[-1]
        ratio = 1.0 * self.n_token / n_sample

        # update counts
        one_hots = F.one_hot(indices, self.n_token).to(inputs.dtype) * ratio
        rev_one_hots = F.one_hot(rev_indices, bsz).to(inputs.dtype) / self.n_token

        batch_cnts = (one_hots + rev_one_hots.t())
        cnts = batch_cnts.sum(dim=0)
        self._ema_n.data = (self._ema_n.data * self._decay +
                            (1 - self._decay) * cnts)

        # update weights
        ws = batch_cnts.t().matmul(inputs)
        self._ema_w.data = (self._ema_w.data * self._decay +
                            (1 - self._decay) * ws.data)

        self._update_weights()


class SoftMemory(nn.Module):
    def __init__(self, n_token, n_embed):
        super().__init__()

        self.n_token = n_token
        self.n_embed = n_embed

        self.linear = nn.Linear(n_embed, n_token)
        self._value_embeds = nn.Embedding(self.n_token, self.n_embed)

    def get_value_weights(self):
        return self._value_embeds.weight

    def forward(self, inputs):
        size = inputs.size()
        inputs = inputs.view(-1, self.n_embed)

        # get quantized outputs
        probs = torch.softmax(self.linear(inputs), dim=-1)

        quantized_embs = probs.matmul(self.get_value_weights())
        quantized = quantized_embs + inputs

        quantized = quantized.reshape(size)
        return quantized


class VectorQuantizedMemory(nn.Module):
    def __init__(self, n_token, n_embed, decay=0.99, eps=1e-8):
        super().__init__()

        self.n_token = n_token
        self.n_embed = n_embed
        self._decay = decay
        self._eps = eps

        self._key_embeds = nn.Embedding(self.n_token, self.n_embed)
        self._key_embeds.weight.requires_grad = False

        self._value_embeds = nn.Embedding(self.n_token, self.n_embed)

        self.register_buffer('_ema_n', torch.ones(self.n_token))
        self.register_buffer('_ema_w', self._key_embeds.weight.clone())
        self._update_weights()

    def get_key_weights(self):
        return self._key_embeds.weight

    def get_value_weights(self):
        return self._value_embeds.weight

    def get_distances(self, inputs):
        embeds = self.get_key_weights()
        dists = ((inputs**2).sum(dim=-1, keepdim=True) +
                 (embeds**2).sum(dim=-1)[None, :] -
                 2 * torch.matmul(inputs, embeds.t()))
        return dists

    def forward(self, inputs):
        size = inputs.size()
        inputs = inputs.view(-1, self.n_embed)

        # get quantized outputs
        dists = self.get_distances(inputs)
        indices = torch.argmin(dists, dim=-1)
        quantized_embs = self._value_embeds(indices)
        quantized = inputs + quantized_embs

        # update ema
        if self.training:
            self._update_ema(inputs, indices)

        # return original shape
        quantized = quantized.reshape(size)
        return quantized

    def _update_weights(self):
        n = self._ema_n.sum()
        size = ((self._ema_n + self._eps) / (n + self._eps * self.n_token) * n)
        embed_normed = self._ema_w / size[:, None]
        self._key_embeds.weight.data = embed_normed.data

    def _update_ema(self, inputs, indices):
        bsz = indices.size()[0]
        ratio = 1.0 * self.n_token / bsz

        # update counts
        batch_cnts = F.one_hot(indices, self.n_token).to(inputs.dtype) * ratio
        cnts = batch_cnts.sum(dim=0)
        self._ema_n.data = (self._ema_n.data * self._decay +
                            (1 - self._decay) * cnts)

        # update weights
        ws = batch_cnts.t().matmul(inputs)
        self._ema_w.data = (self._ema_w.data * self._decay +
                            (1 - self._decay) * ws.data)

        self._update_weights()


class VectorQuantizerMaxEnt(nn.Module):
    def __init__(self, n_token, n_embed, n_sample=10, decay=0.99, eps=1e-8):
        super().__init__()

        self.n_token = n_token
        self.n_embed = n_embed
        self._n_sample = n_sample
        self._decay = decay
        self._eps = eps

        self._embeds = nn.Embedding(self.n_token, self.n_embed)
        self._embeds.weight.requires_grad = False

        self.register_buffer('_ema_n', torch.ones(self.n_token))
        self.register_buffer('_ema_w', self._embeds.weight.clone())
        self._update_weights()

    def get_weights(self):
        return self._embeds.weight

    def get_distances(self, inputs):
        embeds = self.get_weights()
        dists = ((inputs**2).sum(dim=-1, keepdim=True) +
                 (embeds**2).sum(dim=-1)[None, :] -
                 2 * torch.matmul(inputs, embeds.t())) / self.n_token
        return dists

    def get_probs(self, inputs, tau=1.):
        dists = self.get_distances(inputs)
        probs = torch.softmax(-dists / tau, dim=-1)
        return probs

    def forward(self, inputs, rho=1., tau=1., return_mse=False):
        size = inputs.size()
        inputs = inputs.view(-1, self.n_embed)

        # get quantized outputs
        probs = self.get_probs(inputs, tau=tau)
        samples = torch.multinomial(probs,
                                    num_samples=self._n_sample,
                                    replacement=True)
        quantized_embs = self._embeds(samples).mean(dim=1)
        quantized = inputs + (quantized_embs - inputs).detach()

        # gating function
        gate_probs = rho * torch.ones(
            inputs.size()[0], dtype=inputs.dtype, device=inputs.device)
        gates = torch.bernoulli(gate_probs)[:, None]

        quantized = quantized * gates + (1. - gates) * inputs

        # update ema
        if self.training:
            self._update_ema(inputs, samples)

        # return original shape
        quantized = quantized.reshape(size)
        samples = samples.reshape(list(size[:-1]) + [self._n_sample])

        if return_mse:
            mse = (((inputs - quantized_embs.detach())**2).sum(dim=-1) /
                   self.n_embed)
            mse = mse.reshape(size[:-1])
            return quantized, samples, mse
        return quantized, samples

    def _update_weights(self):
        n = self._ema_n.sum()
        size = ((self._ema_n + self._eps) / (n + self._eps * self.n_token) * n)
        embed_normed = self._ema_w / size[:, None]
        self._embeds.weight.data = embed_normed.data

    def _update_ema(self, inputs, samples):
        bsz, n_sample = samples.size()
        ratio = 1.0 * self.n_token / n_sample / bsz

        # update counts
        one_hots = F.one_hot(samples, self.n_token).to(inputs.dtype)
        batch_cnts = one_hots.sum(dim=1) * ratio
        cnts = batch_cnts.sum(dim=0)
        self._ema_n.data = (self._ema_n.data * self._decay +
                            (1 - self._decay) * cnts)

        # update weights
        ws = batch_cnts.t().matmul(inputs)
        self._ema_w.data = (self._ema_w.data * self._decay +
                            (1 - self._decay) * ws.data)

        self._update_weights()
