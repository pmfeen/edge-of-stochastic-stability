import torch as T
import torch
import torch.nn as nn
from utils.measure import calculate_all_the_grads

import numpy as np


def param_length(net):
    params = list(net.parameters())
    return sum([p.numel() for p in params])

def add_gradients(net, grads):
    '''
    Adds the value provided in `grads` to the gradients to the network
    '''
    # assign the gradients to the network
    if len(grads) != param_length(net):
        raise ValueError(f"The length of the grads vector, {len(grads)} is not equal to the number of parameters in the network, {param_length(net)}")
    i = 0
    for p in net.parameters():
        p.grad += grads[i:i + p.numel()].reshape(p.shape)
        i += p.numel()



class GradStorage:
    # store the gradients for the network
    # but only recalculates them every n steps
    def __init__(self, net, recalculate_every=30, ):
        self.grads = None
        self.net = net
        self.device = net.parameters().__next__().device
        self.recalculate_every = recalculate_every

    def get_grads(self, X, Y, loss_fn, optimizer, step_number):
        if step_number % self.recalculate_every == 0 or self.grads is None:
            if self.grads is not None:
                del self.grads
            self.grads = calculate_all_the_grads(self.net, X, Y, loss_fn, optimizer)
        return self.grads


class GradStorageNew:
    # store the gradients for the network
    # but only recalculates them every n steps
    def __init__(self, net, recalculate_every=30, ):
        self.grads = None
        self.net = net
        self.device = net.parameters().__next__().device
        self.recalculate_every = recalculate_every
        self.stds = None

    def get_grads(self, X, Y, loss_fn, optimizer, step_number):
        if step_number % self.recalculate_every == 0 or self.grads is None:
            self.grads = calculate_all_the_grads(self.net, X, Y, loss_fn, optimizer, storage_device='cpu')
        return self.grads
    
    def get_stds(self, X, Y, loss_fn, optimizer, step_number):
        if step_number % self.recalculate_every == 0 or self.stds is None:
            grads = self.get_grads(X, Y, loss_fn, optimizer, step_number)
            self.stds = grads.std(dim=0).to(self.device)
        return self.stds


def gd_with_noise(net, 
                  X, 
                  Y, 
                  loss_fn, 
                  noise_type, 
                  optimizer, 
                  batch_size,
                  step_number,
                  grad_storage,
                  noise_magnitude=None,
                  ):
    device = net.parameters().__next__().device
    # the function also updates the weights of the network

    if noise_type == 'sgd':
        # aka sampling noise

        # create gaussian vector of same size as # samples
        n = X.shape[0]
        noise = T.randn(n, device=device, requires_grad=False)

        # create the matrix to multiply the noise by to get the scorrect Cov
        cov_correction = (T.eye(n, device=device) - 1/n * T.ones((n, n), device=device)) / np.sqrt(n * batch_size)
        # multiply the noise by the matrix to get the noise term of sampling vector
        sampling_vector_noise = cov_correction @ noise
        # add the mean (the GD part, i.e. the full gradient)
        sampling_vector = T.ones(n, device=device) / n + sampling_vector_noise
        # run the forward pass
        preds = net(X).squeeze(dim=-1)
        # compute the sampled loss
        loss = loss_fn(preds, Y, sampling_vector=sampling_vector)
        # compute the gradients
        loss.backward()

        # step
        optimizer.step()

        return loss
    
    if noise_type == 'diag':
        # request the gradient storage for gradients
        grads = grad_storage.get_grads(X, Y, loss_fn, optimizer, step_number) # shape = (n_entries, n_params)

        # # compute the stds (aka the SQUARE ROOT of diagonal of the covariance matrix)
        stds = grads.std(dim=0)

        # stds = grad_storage.get_stds(X, Y, loss_fn, optimizer, step_number)

        # sample a noise vector
        noise = T.randn_like(stds, device=device)

        # multiply the noise by the stds and batch size (see the formula)
        cov_diag_sqrt = stds / np.sqrt(batch_size) 
        diag_noise = cov_diag_sqrt * noise

        print("Noise norm:", (diag_noise**2).sum().item())

        # compute full gradient for update
        # don't forget to zero the grad
        optimizer.zero_grad()
        preds = net(X).squeeze(dim=-1)
        loss = loss_fn(preds, Y)
        loss.backward()

        # store the loss to return

        # add the noise to the gradient 
        # modify the grad assignement function for this to add instead of assign
        add_gradients(net, diag_noise)

        # step
        optimizer.step()

        # return the loss
        return loss

    if noise_type == 'iso':
        # request the gradient storage for gradients
        grads = grad_storage.get_grads(X, Y, loss_fn, optimizer, step_number) # shape = (n_entries, n_params)

        # # compute the stds (aka the SQUARE ROOT of diagonal of the covariance matrix)
        vars = grads.var(dim=0)
        # vars = grad_storage.get_stds(X, Y, loss_fn, optimizer, step_number)**2
        trace = vars.sum()
        diag_magnitude = T.sqrt(trace / len(vars))

        print("Ave variance:", diag_magnitude.item())

        # sample a noise vector
        noise = T.randn_like(vars, device=device)

        # multiply the noise by the stds and batch size (see the formula)
        scaler =  diag_magnitude / np.sqrt(batch_size) 
        diag_noise = scaler * noise

        print("Noise norm:", (diag_noise**2).sum().item())

        # compute full gradient for update
        # don't forget to zero the grad
        optimizer.zero_grad()
        preds = net(X).squeeze(dim=-1)
        loss = loss_fn(preds, Y)
        loss.backward()

        # store the loss to return

        # add the noise to the gradient 
        # modify the grad assignement function for this to add instead of assign
        add_gradients(net, diag_noise)

        # step
        optimizer.step()

        # return the loss
        return loss

    if noise_type == 'const':
        noise = T.randn(param_length(net), device=device)
        scaler = noise_magnitude / np.sqrt(batch_size)

        grad_noise = scaler * noise

        print("Noise norm:", (grad_noise**2).sum().item())

        # compute full gradient for update
        # don't forget to zero the grad
        optimizer.zero_grad()
        preds = net(X).squeeze(dim=-1)
        loss = loss_fn(preds, Y)
        loss.backward()

        # store the loss to return

        # add the noise to the gradient 
        # modify the grad assignement function for this to add instead of assign
        add_gradients(net, grad_noise)

        # step
        optimizer.step()

        # return the loss
        return loss


    raise ValueError(f"Unknown noise type{noise_type}")