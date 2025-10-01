import torch as T
import torch
import torch.nn as nn
from utils.measure import calculate_all_the_grads

import numpy as np
import math


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


def euler_step(
    net: torch.nn.Module, 
    h: float, 
    X: torch.Tensor, 
    Y: torch.Tensor, 
    loss_fn: torch.nn.Module, 
    batch_size: int, 
    eta: float, 
    *, 
    rng: torch.Generator
) -> float:
    """
    One Euler–Maruyama step for the SDE that models SGD dynamics.
    
    This implements the SDE: dθ = -∇F(θ)dt + √η Σ(θ)^(1/2) dW
    where:
    - F(θ) is the population loss (full-batch loss)
    - Σ(θ) is the covariance matrix of per-sample gradients
    - η is the learning rate of the original SGD we're modeling
    - dW is Brownian motion
    
    The Euler-Maruyama discretization gives:
    θ_{n+1} = θ_n - h∇F(θ_n) + √η √h Σ(θ_n)^(1/2) ξ_n
    
    Parameters:
    -----------
    net : torch.nn.Module
        Neural network to be updated in-place
    h : float
        SDE integration time step (NOT the SGD learning rate)
    X, Y : torch.Tensor
        Full dataset tensors on correct device
    loss_fn : torch.nn.Module
        Loss function that supports per-sample losses with reduction='none'
    batch_size : int
        Batch size of the original SGD we're modeling (affects noise scaling)
    eta : float
        Learning rate of the original SGD algorithm we're modeling
    rng : torch.Generator
        Random number generator for reproducibility
        
    Returns:
    --------
    float
        Full-batch loss value
    """
    n = X.shape[0]  # Full dataset size
    device = next(net.parameters()).device
    
    # Parameter manipulation utilities
    params = list(net.parameters())
    param_shapes = [p.shape for p in params]
    
    def flatten_params() -> torch.Tensor:
        """Flatten all network parameters into a single vector."""
        return torch.cat([p.view(-1) for p in params])
    
    def unflatten_and_assign(flat_tensor: torch.Tensor) -> None:
        """Unflatten parameter vector and assign back to network."""
        idx = 0
        for p, shape in zip(params, param_shapes):
            numel = p.numel()
            p.data.copy_(flat_tensor[idx:idx+numel].view(shape))
            idx += numel
    
    # ============================================================================
    # STEP 1: Compute full-batch gradient (drift term of SDE)
    # ============================================================================
    # This computes ∇F(θ) where F is the population loss
    net.zero_grad(set_to_none=True)

    preds = net(X).squeeze(-1)
    full_loss = loss_fn(preds, Y)  # Population loss F(θ)
    full_loss.backward()
    
    # Extract full gradient: ∇F(θ)
    full_grad = torch.cat([p.grad.view(-1) for p in params])
    
    # ============================================================================
    # STEP 2: Generate sampling noise for SGD covariance matrix
    # ============================================================================
    # Generate ξ ~ N(0, I_n) and project to create sampling noise
    xi = torch.randn(n, generator=rng, device=device)
    Pxi = xi - xi.mean()  # Project: (I - 11ᵀ/n)ξ. This is for it to have the (unscaled) correct covariance structure of (I - 11ᵀ/n)
    
    # Scale by SGD batch size: this creates the noise vector that, when applied
    # to per-sample gradients, gives the correct SGD noise covariance
    V = Pxi / math.sqrt(n * batch_size)   # to have the correct covariance of 1/bn *(I - 11ᵀ/n)

    # The resulting increment has

    # $$
    # \operatorname{E}[\Delta\theta] = -h\nabla F(\theta),\qquad
    # \operatorname{Cov}[\Delta\theta] = \eta\,h\,\bigl(\tfrac1{n\,b}GPG^\top\bigr)
    # \;=\;\eta\,h\,\Sigma(\theta),
    # $$

    # exactly matching the SDE **when the mini‑batch is sampled with replacement** (the simplest and most common modelling assumption).  
    # If you want the *finite‑population* (without‑replacement) covariance you would multiply by the finite‑population correction $\tfrac{n-b}{n-1}$;

    
    # ============================================================================
    # STEP 3: Compute noise term using Jacobian-Vector Product (JVP)
    # ============================================================================
    # We need to compute G @ V where G is the n×d gradient matrix
    # (G[i,:] = gradient of loss_i w.r.t. parameters)
    # Using JVP avoids materializing the full gradient matrix G
    
    # for p in params:
    #     if p.grad is not None:
    #         p.grad.zero_()
    net.zero_grad(set_to_none=True)
    
    # Compute per-sample losses: [ℓ₁(θ), ℓ₂(θ), ..., ℓₙ(θ)]
    preds = net(X).squeeze(-1)
    per_sample_loss = loss_fn(preds, Y, reduction='none')  # shape (n,)
    
    # JVP trick: ∇_θ[(per_sample_loss ⊙ V)ᵀ1] = Gᵀ V
    # where G[i,:] = ∇_θ ℓᵢ(θ) and ⊙ is element-wise product
    weighted_loss = (per_sample_loss * V).sum()
    weighted_loss.backward()
    
    # Extract JVP result: G @ V (the stochastic gradient direction)
    gv_product = torch.cat([p.grad.view(-1) for p in params])
    
    # ============================================================================
    # STEP 4: Euler-Maruyama parameter update
    # ============================================================================
    # The SDE is: dθ = -∇F(θ)dt + √η Σ(θ)^(1/2) dW
    # 
    # Euler-Maruyama discretization:
    # θ_{n+1} = θ_n - h∇F(θ_n) + √η √h Σ(θ_n)^(1/2) ξ_n
    #
    # Where:
    # - h∇F(θ_n) is the drift term (deterministic part)
    # - √η √h Σ(θ_n)^(1/2) ξ_n is the diffusion term (stochastic part)
    # - √η comes from the original SGD learning rate
    # - √h comes from the Euler-Maruyama scheme
    # - Σ(θ_n)^(1/2) ξ_n ≈ G @ V (computed via JVP above)
    
    theta_flat = flatten_params()
    
    # Construct the parameter update
    drift_term = h * full_grad                    # h∇F(θ)
    diffusion_term = math.sqrt(eta * h) * gv_product  # √η √h G@V
    
    theta_new = theta_flat - drift_term + diffusion_term
    
    # Update network parameters in-place
    unflatten_and_assign(theta_new)
    
    return full_loss.item()


def sde_integration(
    net: torch.nn.Module,
    X: torch.Tensor, 
    Y: torch.Tensor, 
    loss_fn: torch.nn.Module,  
    batch_size: int,
    h: float,
    eta: float,
    rng: torch.Generator = None
) -> float:
    """
    Integrate the SDE for one step using Euler-Maruyama method to simulate SGD.
    
    This is a wrapper around euler_step that handles default RNG setup.
    
    Parameters:
    -----------
    net : torch.nn.Module
        Neural network to be updated
    X, Y : torch.Tensor
        Full dataset tensors
    loss_fn : torch.nn.Module
        Loss function (must support reduction='none')
    batch_size : int
        Batch size of the SGD we're modeling
    h : float
        SDE integration time step size
    eta : float
        Learning rate of the original SGD
    rng : torch.Generator, optional
        Random number generator (creates default if None)
        
    Returns:
    --------
    float
        Full-batch loss value after the update
    """

    assert h <= eta, "Step size h must be less than or equal to learning rate eta"
    if h > 0.5*eta:
        print(f"WARNING!! Are you sure you want to use h={h} that is so close to eta={eta}? h is the step size of integration. It should be much smaller than eta to ensure stability of the Euler-Maruyama method. Consider using a smaller h, e.g. h=0.1*eta or even smaller. But feel free to proceed if you know what you are doing.")
        # raise ValueError(f"Are you sure you want to use h={h} that is so close to eta={eta}? h is the step size of integration. It should be much smaller than eta to ensure stability of the Euler-Maruyama method. Consider using a smaller h, e.g. h=0.1*eta or even smaller. But feel free to proceed if you know what you are doing.")

    # Create default RNG if not provided
    if rng is None:
        rng = torch.Generator(device=next(net.parameters()).device)
    
    # Perform eta/h Euler-Maruyama steps
    for i in range(int(eta/h)):
        # Each step is a single Euler-Maruyama update
        # This will update the network parameters in-place
        
        loss = euler_step(net, h, X, Y, loss_fn, batch_size, eta, rng=rng)
    
    return loss

