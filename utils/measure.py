import torch as T
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np

__all__ = ['param_vector', 'param_length', 'flatt', 'grads_vector', 
           'calculate_all_the_grads', 'compute_lambdamax', 'compute_grad_H_grad', 
           'compute_eigenvalues', 'calculate_averaged_lambdamax', 'create_ntk', 
           'compute_fisher_eigenvalues', 'calculate_averaged_gHg_old', 'calculate_all_net_grads',
           'calculate_averaged_grad_H_grad', 'calculate_averaged_gni', 'compute_grad_H_grad_new',
           'calculate_accuracy', 'calculate_param_distance']


def param_vector(net):
    '''
    Returns a vector of all the parameters of the network
    '''
    # params = list(net.parameters())
    param_vector = T.cat([p.flatten() for p in net.parameters()])
    return param_vector

def param_length(net):
    '''
    Returns the number of parameters in the network
    '''
    params = list(net.parameters())
    return sum([p.numel() for p in params])

def flatt(vectors):
    '''
    Flattens a list of vectors into a single vector
    '''
    return T.cat([v.flatten() for v in vectors])


def grads_vector(net):  
    # pull out all the gradients from a network as one vector
    grads = []
    for p in net.parameters():
        grads.append(p.grad.flatten().detach().clone())
    return T.cat(grads)


def calculate_param_distance(net, reference_params, p=2):
    """
    Calculate the distance between current network parameters and reference parameters.
    
    Args:
        net (nn.Module): Neural network model
        reference_params (Tensor): Flattened reference parameters (from param_vector())
        p (int, optional): The norm degree. Default: 2 for Euclidean distance
    
    Returns:
        Tensor: The p-norm distance between current and reference parameters
    """
    with torch.no_grad():
        current_params = param_vector(net)
        return T.linalg.vector_norm(current_params - reference_params, ord=p)


def calculate_all_the_grads(net, X, Y, loss_fn, optimizer, storage_device=None):
    # device = net.parameters().__next__().device

    grads = [] # datapoint, parameter
    for x, y in zip(X, Y):
        optimizer.zero_grad()
        y_pred = net(x.unsqueeze(0)).squeeze(dim=-1)
        loss = loss_fn(y_pred, y.unsqueeze(0))
        loss.backward()
        detached_grads = grads_vector(net).detach()
        if storage_device:
            detached_grads = detached_grads.to(storage_device)
        grads.append(detached_grads)
    
    return T.stack(grads)


def calculate_accuracy(predictions, targets):
    """
    Calculate the accuracy given the model predictions and target labels.
    
    Args:
        predictions: tensor of shape (num_samples, num_classes) with model predictions
        targets: tensor of shape (num_samples, num_classes) with one-hot encoded labels
                or tensor of shape (num_samples,) with class indices
    
    Returns:
        accuracy: float representing the accuracy (0.0 to 1.0)
    """
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get the predicted class (highest value in each row)
        # this is if we have all the classes
        pred_classes = torch.argmax(predictions, dim=1)
    else:
        # Get the predicted class (sign of the prediction)
        # this is if we have only two classes
        pred_classes = torch.sign(predictions).long()

    
    
    # Check if targets are one-hot encoded or class indices
    if len(targets.shape) > 1 and targets.shape[1] > 1:
        # One-hot encoded targets
        true_classes = torch.argmax(targets, dim=1)
    else:
        # Class indices (1D tensor)
        if len(targets.shape) == 1:
            true_classes = torch.round(targets).long()
        else:
            true_classes = targets.long()
    
    # Compare and compute accuracy
    correct = (pred_classes == true_classes).sum().item()
    total = targets.size(0)
    
    return correct / total


def compute_lambdamax(loss, net, max_iterations = 100, 
                    
                      epsilon = 1e-3,
                      init_vector = None,
                      batched = None,
                      compute_gHg: bool = False,
                      return_ghg_gg_separately: bool = False):
    """
    Computes the sharpness (maximum eigenvalue of the Hessian) of the loss function at the current point using power iteration method.

    Args:
        loss (Tensor): The loss value at the current point
        net (nn.Module): The neural network model
        max_iterations (int, optional): Maximum number of power iterations. Defaults to 100.
        epsilon (float, optional): Convergence threshold for eigenvalue computation. Defaults to 1e-3.
        init_vector (Tensor, optional): Initial vector for power iteration. If None, uses gradient. Defaults to None.
        batched (Any, optional): Unused parameter. Defaults to None.
        compute_gHg (bool, optional): Whether to compute gHg (gradient-Hessian-gradient) value. Defaults to False.
        return_ghg_gg_separately (bool, optional): Whether to return gHg and g^2 separately. Defaults to False.

    Returns:
        Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, Tuple[Tensor, Tensor]]]:
            - If compute_gHg is False: Returns the maximum eigenvalue of the Hessian
            - If compute_gHg is True and return_ghg_gg_separately is False: Returns tuple of (maximum eigenvalue, gHg value)
            - If compute_gHg is True and return_ghg_gg_separately is True: Returns tuple of (maximum eigenvalue, (gHg, g^2))

    Note:
        The function uses power iteration method to compute the maximum eigenvalue
        of the Hessian matrix without explicitly forming the matrix.
        The computation stops when the relative change in eigenvalue is less than epsilon
        or when max_iterations is reached.
    """
    device = next(net.parameters()).device

    if return_ghg_gg_separately and not compute_gHg:
        raise ValueError("return_ghg_gg_separately can only be True if compute_gHg is True")

    # random init vector
    # TODO: start with the gradient as the initial vector
    size = param_length(net)
    if init_vector is None:
        v = T.rand(size, device=device)*2 - 1
    else:
        v = init_vector
    v = v / T.linalg.vector_norm(v)


    # compute gradient and keep it
    params = list(net.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grads_vector = flatt(grads)

    size = param_length(net)
    if init_vector is None:
        # v = T.rand(size, device=device)*2 - 1
        v = grads_vector.detach()
    else:
        v = init_vector
    v = v / T.linalg.vector_norm(v)

    # compute the gHg
    if compute_gHg:
        step_vector = grads_vector.detach()
        grad_step = T.dot(grads_vector, step_vector)
        Hv = T.autograd.grad(grad_step, params, retain_graph=True)
        Hv = flatt(Hv).detach()
        
        if not return_ghg_gg_separately:
            gHg = T.dot(step_vector, Hv) / T.dot(step_vector, step_vector)
        else:
            gHg = T.dot(step_vector, Hv)
            norm_g = T.dot(step_vector, step_vector)


    # grad_vector as init_vector, since it is very close to the eigenvector
    # v = grads_vector.detach()

    # repetitive procedure:
    # compute Hv, normalize, repeat
    eigenval = -10
    eigenvals = [-100, -20]
    for i in range(max_iterations):
        grad_v = T.dot(grads_vector, v)
        Hv = flatt(T.autograd.grad(grad_v, params, retain_graph=True))
        with T.no_grad():
            v = Hv / T.linalg.vector_norm(Hv)
            v = v.detach()
            eigenval = T.dot(Hv, v) / T.dot(v, v)
        old_eigenval = eigenvals[-2]
        eigenvals.append(eigenval)

        if abs(eigenval - old_eigenval) / eigenval < epsilon:
            break

    if compute_gHg:
        if return_ghg_gg_separately:
            return eigenval, (gHg, norm_g)
        return eigenval, gHg

    return eigenval


def compute_grad_H_grad(loss, net, grad_already_there: bool = False,
                        return_ghg_gg_separately: bool = False):
    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=True)
    else:
        grads = [p.grad for p in params]
    grads_vector = flatt(grads)

    # compute Hessian vector product
    # grads_vector = T.cat([g.flatten() for g in grads])
    step_vector = grads_vector.detach()
    grad_step = T.dot(grads_vector, step_vector)
    Hv = T.autograd.grad(grad_step, params, retain_graph=False)
    Hv = flatt(Hv).detach()

    if return_ghg_gg_separately:
        return T.dot(step_vector, Hv), T.dot(step_vector, step_vector)
    return T.dot(step_vector, Hv) / T.dot(step_vector, step_vector)

def compute_grad_H_grad_new(loss, net, grad_already_there: bool = False):
    '''
    This one returns gHg and norm of g separately, so you can compute the expectation for both
    '''

    device = next(net.parameters()).device

    # compute gradient and keep it
    params = list(net.parameters())
    if not grad_already_there:
        grads = torch.autograd.grad(loss, params, create_graph=True)
    else:
        grads = [p.grad for p in params]
    grads_vector = flatt(grads)

    # compute Hessian vector product
    # grads_vector = T.cat([g.flatten() for g in grads])
    step_vector = grads_vector.detach()
    grad_step = T.dot(grads_vector, step_vector)
    Hv = T.autograd.grad(grad_step, params, retain_graph=False)
    Hv = flatt(Hv).detach()

    return T.dot(step_vector, Hv), T.dot(step_vector, step_vector)


def compute_eigenvalues(operator, size, device, iterations=100, epsilon=1e-3):
    # compute eigenvalues using power method
    # matrix is a function that computes matrix-vector product
    with T.no_grad():
        v = T.rand(size, device=device)
        v = v / T.linalg.norm(v)

        eigenval = -10
        for _ in range(iterations):
            Av = operator(v)
            v_new = Av / T.linalg.norm(Av)
            eigenval_new = T.dot(v, Av)

            if T.linalg.norm(eigenval_new - eigenval) < epsilon:
                break
            
            v = v_new
            eigenval = eigenval_new

        return eigenval.item()


def calculate_averaged_lambdamax(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                              max_hessian_iters = 1000,
                              hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False
                              ): 
    
    
    sharpnesses = []

    if compute_gHg:
        gHg_values = []
    
    if batch_size is None:
        batch_size = len(X)
    

    for i in range(n_estimates):
        shuffle = T.randperm(len(X))
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        sharpness = compute_lambdamax(loss, 
                        net,
                        max_iterations=max_hessian_iters,
                        compute_gHg=compute_gHg,
                        epsilon=hessian_tolerance,
                        )
        if compute_gHg:
            sharpness, gHg = sharpness
            gHg = gHg.item()
            gHg_values.append(gHg)
        
        sharpness = sharpness.item()
        
        sharpnesses.append(sharpness)

        if batch_size >= len(X):
            break

        if len(sharpnesses) > min_estimates:
            mean = np.mean(sharpnesses)
            sem = np.std(sharpnesses) / np.sqrt(len(sharpnesses))

            if sem / mean < tolerance:
                break
    
    if compute_gHg:
        return sharpnesses, gHg_values
    return sharpnesses


def calculate_averaged_gHg_old(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                              ): 
    gHg_values = []
    

    for i in range(n_estimates):
        shuffle = T.randperm(len(X))
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        
        gHg = compute_grad_H_grad(loss, net)
        gHg = gHg.item()
        
        
        gHg_values.append(gHg)

        if batch_size >= len(X):
            break

        if len(gHg_values) > min_estimates:
            mean = np.mean(gHg_values)
            sem = np.std(gHg_values) / np.sqrt(len(gHg_values))

            if sem / mean < tolerance:
                break
    

    return gHg_values



def calculate_averaged_grad_H_grad(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 100,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                              ): 
    '''
    Computes the fraction E[gHg] / E[g^2], rather than the previous E [gHg / g^2].
    Does a calculationg at a point
    '''
    gHg_values = []
    norm_g_values = []
    

    for i in range(n_estimates):
        shuffle = T.randperm(len(X))
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        
        gHg, norm_g = compute_grad_H_grad_new(loss, net)
        gHg = gHg.item()
        norm_g = norm_g.item()
        
        
        gHg_values.append(gHg)
        norm_g_values.append(norm_g)

        if batch_size >= len(X):
            break

        if len(gHg_values) > min_estimates:
            mean = np.mean(gHg_values)
            sem = np.std(gHg_values) / np.sqrt(len(gHg_values))

            if sem / mean < tolerance:
                break
    
    gHg_new = np.mean(gHg_values) / np.mean(norm_g_values)

    return gHg_new


def calculate_averaged_gni(net,
                              X,
                              Y,
                              loss_fn,
                              batch_size,
                              n_estimates = 500,
                              min_estimates = 10,
                              tolerance = 0.01, # st error of mean / mean
                            #   max_hessian_iters = 1000,
                            #   hessian_tolerance = 1e-3,
                              batched = None,
                              compute_gHg: bool = False
                              ): 
    sharpnesses = []

    params = list(net.parameters())


    total_loss = loss_fn(net(X).squeeze(dim=-1), Y)

    total_grad = flatt(torch.autograd.grad(total_loss, params, create_graph=True))

    total_grad_detach = total_grad.detach()

    normalizer = T.dot(total_grad_detach, total_grad_detach).item()

    gHg_list = []


    for i in range(n_estimates):
        shuffle = T.randperm(len(X))
        random_idx = shuffle[:batch_size]

        X_batch = X[random_idx]
        Y_batch = Y[random_idx]


        loss = loss_fn(net(X_batch).squeeze(dim=-1), Y_batch)

        grads_vector = flatt(torch.autograd.grad(loss, params))
        step_vector = grads_vector.detach()

        total_grad_dot_step = T.dot(total_grad, step_vector)

        Hg = flatt(torch.autograd.grad(total_grad_dot_step, params, retain_graph=True))

        Hg = Hg.detach()

        gHg = T.dot(step_vector, Hg)

        gHg_list.append(gHg.item())


    quantity = np.mean(gHg_list) / normalizer

    return quantity





##### All THE FISHER STUFF

def calculate_all_net_grads(net, X):

    gradients = []
    params = list(net.parameters())

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)
    del gradients
    return G



def create_ntk(net, X):
    params = list(net.parameters())

    gradients = []

    for x in X:
        y = net(x.unsqueeze(0))
        # compute gradient
        grads = torch.autograd.grad(y, params)
        grads_vector = flatt(grads).detach()
        gradients.append(grads_vector)
    
    G = T.stack(gradients)

    ntk = G @ G.T
    del G
    # f = lambda v: G.T @ (G @ v) / len(X)

    return ntk


def compute_fisher_eigenvalues(net, X):
    '''
    The trick here is that instead of computing the fisher information matrix, we compute the NTK
    They have the same eigenvalues, but NTK is size n_samples x n_samples, while FIM is size n_params x n_params
    '''

    ntk = create_ntk(net, X)
    # size = param_length(net)

    # device = next(net.parameters()).device
    # eigenval = compute_eigenvalues(operator, size, device, iterations=iterations, epsilon=epsilon)

    eigenval = T.lobpcg(ntk, k=1)
    eigenval = 2/len(X) * eigenval[0]

    return eigenval



