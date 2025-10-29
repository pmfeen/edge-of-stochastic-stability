import torch as T
import torch
import torch.nn as nn
import os
from einops import rearrange, repeat
from torch import linalg as LA
import numpy as np
from copy import deepcopy
from pathlib import Path
import math
import random
import argparse

import time

from utils.data import prepare_dataset, get_dataset_presets
from utils.nets import SquaredLoss, MLP, CNN, prepare_net, initialize_net, prepare_optimizer, get_model_presets
from utils.nets import ResNet
from utils.ddpm_wrapper import DDPMWrapper
from utils.storage import initialize_folders
from utils.wandb_utils import (
    init_wandb,
    log_metrics,
    save_checkpoint_wandb,
    find_closest_checkpoint_wandb,
    load_checkpoint_wandb,
    get_checkpoint_dir_for_run,
    is_wandb_available,
    generate_run_id,
)
from utils.noise import gd_with_noise, GradStorage, sde_integration
from utils.measure import *
from utils.frequency import frequency_calculator, MeasurementContext
from utils.quadratic import QuadraticApproximation, flatten_params, set_model_params, unflatten_params

from torch.autograd import grad
import json

if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
if 'RESULTS' not in os.environ:
    raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

DATASET_FOLDER = Path(os.environ.get('DATASETS'))
# export RESULTS=/scratch/gpfs/andreyev/eoss/results
RES_FOLDER = Path(os.environ.get('RESULTS'))


# -------------------------------------
# Section: Measurement Runner
# -------------------------------------

class MeasurementRunner:
    """Centralized measurement orchestration for the training loop."""
    _FIRST_FEW_FULL_LOSS_STEPS = 32

    def __init__(
        self,
        *,
        net,
        loss_fn,
        full_inputs,
        measurements,
        device,
        batch_size,
        save_dir,
        eigenvector_cache,
        num_eigenvalues,
        use_power_iteration,
        precise_plots,
        rare_measure,
        param_reference,
        step_to_start,
        sde_enabled,
        gd_noise,
        proj_switch_step,
        quad_approx,
    ):
        self.net = net
        self.loss_fn = loss_fn
        self.X, self.Y = full_inputs
        self.measurements = measurements
        self.device = device
        self.batch_size = batch_size
        self.eigenvector_cache = eigenvector_cache
        self.num_eigenvalues = num_eigenvalues
        self.use_power_iteration = use_power_iteration
        self.precise_plots = precise_plots
        self.rare_measure = rare_measure
        self.param_reference = param_reference
        self.step_to_start = step_to_start
        self.sde_enabled = sde_enabled
        self.gd_noise = gd_noise
        self.proj_switch_step = proj_switch_step
        self.quad_approx = quad_approx

        self.eigenvalues_log = []
        if 'lmax' in measurements and num_eigenvalues > 1:
            eigenvalues_path = save_dir / 'eigenvalues.json'
            self.eigenvalues_file = open(eigenvalues_path, 'w')
            self.eigenvalues_file.write('[\n')
        else:
            self.eigenvalues_file = None

    def close(self):
        if self.eigenvalues_file is not None:
            self.eigenvalues_file.write('\n]')
            self.eigenvalues_file.close()

    def collect(
        self,
        *,
        ctx,
        optimizer,
        X_batch,
        Y_batch,
        epoch,
        step_in_epoch,
        step_number,
    ):
        metrics = {
            'batch_lmax': np.nan,
            'step_sharpness': np.nan,
            'batch_sharpness': np.nan,
            'batch_sharpness_exp_inside': np.nan,
            'fisher_batch_eigenval': np.nan,
            'fisher_total_eigenval': np.nan,
            'full_accuracy': np.nan,
            'full_gHg': np.nan,
            'full_loss': np.nan,
            'gni': np.nan,
            'grad_projections': None,
            'one_step_loss_change': np.nan,
            'param_distance': np.nan,
            'gradient_norm_squared': np.nan,
            'lmax': np.nan,
            'all_eigenvalues': None,
            'quadratic_loss': None,
            'quadratic_loss_gn': None,
            'proj_grad_ratio': None,
            'hessian_trace': np.nan,
        }

        epoch_loss_update = None


        # ----- Batch sharpness (expected Rayleigh quotient) -----
        if 'batch_sharpness' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness', ctx):
                metrics['batch_sharpness'] = calculate_averaged_grad_H_grad_step(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=20,
                    eps=0.005,
                )
        # ----- Instantaneous step sharpness (current-batch Rayleigh quotient) -----
        if 'step_sharpness' in self.measurements:
            if frequency_calculator.should_measure('step_sharpness', ctx):
                self.net.zero_grad()
                preds = self.net(X_batch).squeeze(dim=-1)
                loss = self.loss_fn(preds, Y_batch)
                metrics['step_sharpness'] = compute_grad_H_grad(loss, self.net).item()

        # ----- Eigenvalues/Lambda max (full batch) -----
        lmax_now = False
        if 'lmax' in self.measurements:
            measurement_type = 'full_batch_lambda_max'
            lmax_now = frequency_calculator.should_measure(measurement_type, ctx)

        if lmax_now:
            if str(self.device).startswith('cuda'):
                torch.cuda.empty_cache()
            optimizer.zero_grad()

            lmax_max_size = 4096
            if str(self.device).startswith('cuda'):
                total_memory = torch.cuda.get_device_properties(0).total_memory
                if total_memory < 20 * 1024**3:
                    if isinstance(self.net, CNN):
                        lmax_max_size = 2048 + 512
                    if isinstance(self.net, ResNet):
                        lmax_max_size = 512

            if len(self.X) > lmax_max_size:
                print(f"Warning: Computing eigenvalues on subset of {lmax_max_size} samples instead of full dataset ({len(self.X)} samples) due to memory/time constraints. Most of the time it is fine, but should be corrected")
                idx = gimme_random_subset_idx(len(self.X), lmax_max_size)
                X_subset = self.X[idx]
                Y_subset = self.Y[idx]
            else:
                X_subset = self.X
                Y_subset = self.Y

            # Handle DDPM models that compute loss internally
            if isinstance(self.net, DDPMWrapper):

                loss = self.net(X_subset)
            else:
                preds = self.net(X_subset).squeeze(dim=-1)
                loss = self.loss_fn(preds, Y_subset)

            if self.eigenvector_cache is not None:
                max_iterations = 100 if not self.use_power_iteration else 1000
                tolerance = 0.005 if self.num_eigenvalues < 6 else 0.03
                if self.precise_plots:
                    max_iterations = 300 if not self.use_power_iteration else 3000
                    tolerance = 0.001 if self.num_eigenvalues < 6 else 0.01

                eigenvalues, eigenvectors = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=max_iterations,
                    reltol=tolerance,
                    eigenvector_cache=self.eigenvector_cache,
                    return_eigenvectors=True,
                    use_power_iteration=self.use_power_iteration,
                )

                if self.num_eigenvalues == 1:
                    self.eigenvector_cache.store_eigenvector(eigenvectors, eigenvalues.item())
                    lmax_value = eigenvalues
                else:
                    self.eigenvector_cache.store_eigenvectors(
                        [eigenvectors[:, i] for i in range(eigenvectors.shape[1])],
                        eigenvalues.tolist(),
                    )
                    lmax_value = eigenvalues[0]
            else:
                eigenvalues = compute_eigenvalues(
                    loss,
                    self.net,
                    k=self.num_eigenvalues,
                    max_iterations=200,
                    reltol=0.03,
                    use_power_iteration=self.use_power_iteration,
                )
                if self.num_eigenvalues == 1:
                    lmax_value = eigenvalues
                else:
                    lmax_value = eigenvalues[0]

            if self.num_eigenvalues > 1:
                metrics['all_eigenvalues'] = eigenvalues
                if self.eigenvalues_file is not None:
                    eigenvalues_data = {
                        'epoch': epoch,
                        'step': step_number,
                        'eigenvalues': eigenvalues.tolist()
                        if isinstance(eigenvalues, torch.Tensor)
                        else [eigenvalues],
                    }
                    self.eigenvalues_log.append(eigenvalues_data)
                    if len(self.eigenvalues_log) > 1:
                        self.eigenvalues_file.write(',\n')
                    json.dump(eigenvalues_data, self.eigenvalues_file)
                    self.eigenvalues_file.flush()

            metrics['lmax'] = lmax_value.item()
            metrics['full_loss'] = loss.item()
            if not isinstance(self.net, DDPMWrapper):
                metrics['full_accuracy'] = calculate_accuracy(preds, Y_subset)

            epoch_loss_update = metrics['full_loss']

            print(
                f"Epoch {epoch + 1}, Step {step_in_epoch}: Total lambda max = {metrics['lmax']}, "
                f"Loss = {metrics['full_loss']} !!!"
            )

        

        if 'hessian_trace' in self.measurements:
            if frequency_calculator.should_measure('hessian_trace', ctx):
                metrics['hessian_trace'] = estimate_hessian_trace(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    max_estimates=256,
                    min_estimates=20,
                    eps=0.01,
                )


        # ----- Gradient-noise interaction (GNI) -----
        gni_now = False
        if 'gni' in self.measurements:
            gni_now = frequency_calculator.should_measure('gni', ctx)

        if gni_now:
            metrics['gni'] = calculate_gni(
                net=self.net,
                X=self.X,
                Y=self.Y,
                loss_fn=self.loss_fn,
                batch_size=self.batch_size,
                n_estimates=500,
                tolerance=0.05,
            )


        
        # ----- Fisher eigenvalues (total and batch) -----
        if 'fisher' in self.measurements:
            if frequency_calculator.should_measure('fisher_total', ctx):
                metrics['fisher_total_eigenval'] = compute_fisher_eigenvalues(self.net, self.X).item()
            if frequency_calculator.should_measure('fisher_batch', ctx):
                metrics['fisher_batch_eigenval'] = compute_fisher_eigenvalues(self.net, X_batch).item()

        # ----- Parameter distance from reference -----
        if 'param_distance' in self.measurements:
            if self.param_reference is None:
                raise ValueError('Parameter reference must be provided for param_distance measurement')
            if frequency_calculator.should_measure('param_distance', ctx):
                metrics['param_distance'] = calculate_param_distance(self.net, self.param_reference).item()

        # ----- Gradient norm squared estimate -----
        if 'gradient_norm' in self.measurements:
            if frequency_calculator.should_measure('gradient_norm_squared', ctx):
                metrics['gradient_norm_squared'] = calculate_gradient_norm_squared_mc(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=200,
                    min_estimates=10,
                    eps=0.01,
                )

        # ----- Expected one-step loss change -----
        if 'one_step_loss_change' in self.measurements:
            if frequency_calculator.should_measure('one_step_loss_change', ctx):
                metrics['one_step_loss_change'] = calculate_expected_one_step_full_loss_change(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    optimizer,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=10,
                    eps=0.01,
                    use_subset_of_data=2048,
                )

        # ----- Quadratic approximation diagnostics -----
        if self.quad_approx is not None and self.quad_approx.is_active:
            metrics['quadratic_loss'] = self.quad_approx.compute_quadratic_loss_for_logging(self.X, self.Y)

        # ----- Gradient projection diagnostics -----
        grad_projection_now = False
        if 'grad_projection' in self.measurements:
            if self.sde_enabled or self.gd_noise:
                raise Exception('Gradient projection not implemented for SDE or GD with noise')
            grad_projection_now = frequency_calculator.should_measure('grad_projection', ctx)

        if self.proj_switch_step is not None and step_number >= self.proj_switch_step:
            grad_projection_now = True

        if grad_projection_now:
            if not (self.quad_approx is not None and self.quad_approx.is_active):
                if (
                    self.eigenvector_cache is not None
                    and hasattr(self.eigenvector_cache, 'eigenvectors')
                    and len(self.eigenvector_cache.eigenvectors) > 0
                ):
                    params = [p for p in self.net.parameters() if p.requires_grad]
                    full_preds = self.net(self.X).squeeze(dim=-1)
                    full_loss_for_grad = self.loss_fn(full_preds, self.Y)
                    grad_list = torch.autograd.grad(full_loss_for_grad, params, create_graph=False, retain_graph=False)
                    grad_flat = torch.cat([g.reshape(-1) for g in grad_list]).detach()

                    cached_vecs = torch.stack(self.eigenvector_cache.eigenvectors, dim=1).to(grad_flat.device)
                    cached_vals = getattr(self.eigenvector_cache, 'eigenvalues', None)

                    max_k = min(20, self.num_eigenvalues)
                    metrics['grad_projections'] = compute_gradient_projection_ratios(
                        grad_vector=grad_flat,
                        eigvecs=cached_vecs,
                        max_k=max_k,
                        eigenvalues=cached_vals,
                    )

        # ----- batch sharpness (but with the expectation inside) -----
        if 'batch_sharpness_exp_inside' in self.measurements:
            if frequency_calculator.should_measure('batch_sharpness_exp_inside', ctx):
                metrics['batch_sharpness_exp_inside'] = calculate_averaged_grad_H_grad(
                    self.net,
                    self.X,
                    self.Y,
                    self.loss_fn,
                    batch_size=self.batch_size,
                    n_estimates=1000,
                    min_estimates=20,
                    eps=0.005,
                )


        # ----- Batch lambda max -----
        batch_lmax_now = False
        if 'batch_lmax' in self.measurements:
            if self.gd_noise is None:
                batch_lmax_now = frequency_calculator.should_measure('batch_lambda_max', ctx)
            else:
                raise ValueError('Batch lambda max not implemented for GD noise')

        if batch_lmax_now:
            optimizer.zero_grad()
            preds = self.net(X_batch).squeeze(dim=-1)
            loss = self.loss_fn(preds, Y_batch)
            batch_lmax = compute_eigenvalues(loss, self.net, k=1, max_iterations=50, reltol=1e-3)
            metrics['batch_lmax'] = batch_lmax.item()
            print(
                f"Epoch {epoch + 1}, Step {step_in_epoch}: Batch Lambda Max = {metrics['batch_lmax']}, "
                f"Loss = {loss.item()}"
            )
    

        metrics['epoch_loss_update'] = epoch_loss_update
        return metrics
    
        


# -------------------------------------
# Section: Training Function
# -------------------------------------


def train(
            net,
            optimizer,
            data, # tuple of X_train, Y_train, X_test, Y_test
            max_epochs,
            max_steps,
            batch_size,
            save_to, #folder
            device,
            verbose=True,
            loss_fn=nn.MSELoss(),
            permute=True,
            stop_loss=None,
            epoch_to_start=0,
            step_to_start=0,
            gd_noise=False,
            noise_magnitude=None,
            results_rarely: bool = False,
            measurements: set = {},
            param_reference = None,  # reference weights to measure distance from during training
            cache_eigenvectors: bool = True,  # use eigenvector caching for warm starts
            sde_enabled: bool = False,  # enable SDE integration
            sde_h: float = 0.01,  # SDE integration time step
            sde_eta: float = None,  # SDE learning rate (uses optimizer lr if None)
            sde_seed: int = 888,  # SDE random seed
            use_power_iteration: bool = False,  # Use power iteration for eigenvalue computation
            num_eigenvalues: int = 1,  # Number of eigenvalues to compute
            checkpoint_every_n_steps: int = None,  # Checkpoint frequency 
            quad_switch_step: int = None,  # Step to switch to quadratic approximation
            use_gauss_newton: bool = False,  # Use Gauss-Newton instead of Hessian
            quad_switch_lr: float = None,  # lr to use after switching to quadratic approximation
            precise_plots: bool = False,  # Enable more frequent measurements for precise plotting
            rare_measure: bool = False,  # Make expensive measurements rarer
            # Gradient projection configuration
            proj_switch_step: int = None,  # Step to start projecting minibatch gradients
            proj_top_l: int = None,        # Number of top eigendirections to use for projection
            proj_to_residual: bool = False, # If True, project to orthogonal complement of top-l eigenspace
            wandb_run=None,
            wandb_enabled: bool = False,
            wandb_run_id: str = None,
            ):
    
    # -------------------------------------
    # Section: Setup
    # -------------------------------------
    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # ----- Checkpoint Frequency Defaults -----
    NET_SAVES_PER_TRAINING = 200

    assert max_epochs is not None or max_steps is not None
    if max_epochs is None:
        # Set very high epoch limit if only max_steps is used
        max_epochs = 100000

    # ----- Dataset Wiring -----
    X_train, Y_train, X_test, Y_test = data

    X, Y = X_train, Y_train

    # ----- Device Alignment -----
    net = net.to(device)
    net.train()
    net.float()
    
    X = X.to(device)
    Y = Y.to(device)

    # ----- Storage Preparation -----
    save_to.mkdir(parents=True, exist_ok=True)

    model_save_path = save_to / 'checkpoints'

    results_file = save_to / 'results.txt'
    if device == 'cpu':
        # No buffering on CPU to ensure writes happen immediately
        results_file = open(results_file, 'a', buffering=1)
        torch.set_num_threads(40)
    else:
        # Use buffering on GPU for better performance
        results_file = open(results_file, 'a', buffering=1_000)

    # ----- State Initialization -----
    step_number = -1 if step_to_start == 0 else step_to_start

    if gd_noise is not None:
        grad_storage = GradStorage(net, recalculate_every=30)
    
    # ----- Stochastic Dynamics Setup -----
    if sde_enabled:
        if sde_eta is None:
            sde_eta = optimizer.param_groups[0]['lr']  # Use optimizer's learning rate
        sde_rng = T.Generator(device=device)
        sde_rng.manual_seed(sde_seed)  # Use the provided SDE seed

    # ----- Checkpoint Interval Selection -----
    if checkpoint_every_n_steps is None:
        checkpoint_every_n_steps = max(max_steps // NET_SAVES_PER_TRAINING, 1)
    print(f"Will save checkpoints every {checkpoint_every_n_steps} steps")

    # ----- Quadratic Approximation Setup -----
    quad_approx = None
    if quad_switch_step is not None:
        quad_approx = QuadraticApproximation(net, loss_fn, device, quad_switch_step, use_gauss_newton)
        
        # Handle continuation case - if we're starting after the switch step,
        # we need to initialize the anchor immediately
        if step_to_start >= quad_switch_step:
            print(f"Initializing quadratic approximation at continuation step {step_to_start} (switch was at {quad_switch_step})")
            # Initialize with current model state as anchor
            quad_approx.anchor_params = flatten_params(net).detach().clone()
            quad_approx.anchor_loss = 0.0  # Will be computed on first batch
            quad_approx.delta = torch.zeros_like(quad_approx.anchor_params)
            quad_approx.is_active = True
            print("Quadratic approximation initialized as active for continuation")
        else:
            print(f"Quadratic approximation will switch at step {quad_switch_step}")
    # ----- Training State Trackers -----
    epoch_loss = float('+inf')
    stop_training = False

    # -------------------------------------
    # Section: Measurements
    # -------------------------------------
    # ----- Eigenvector Cache Setup -----
    eigenvector_cache = None
    # Also create cache if gradient projection is enabled, since it relies on cached eigenvectors
    if (('lmax' in measurements or 'grad_projection' in measurements) or (proj_switch_step is not None)) and cache_eigenvectors:
        max_cache = 5
        if num_eigenvalues is not None:
            max_cache = max(max_cache, num_eigenvalues)
        if proj_top_l is not None:
            max_cache = max(max_cache, proj_top_l)
        eigenvector_cache = EigenvectorCache(max_eigenvectors=max_cache)
    
    # ----- Measurement Runner Wiring -----
    measurement_runner = MeasurementRunner(
        net=net,
        loss_fn=loss_fn,
        full_inputs=(X, Y),
        measurements=measurements,
        device=device,
        batch_size=batch_size,
        save_dir=save_to,
        eigenvector_cache=eigenvector_cache,
        num_eigenvalues=num_eigenvalues,
        use_power_iteration=use_power_iteration,
        precise_plots=precise_plots,
        rare_measure=rare_measure,
        param_reference=param_reference,
        step_to_start=step_to_start,
        sde_enabled=sde_enabled,
        gd_noise=gd_noise,
        proj_switch_step=proj_switch_step,
        quad_approx=quad_approx,
    )
    # ----- Run Identification -----
    run_id = wandb_run_id or generate_run_id()

    # If resuming at/after projection switch step, precompute eigendirections immediately
    if proj_switch_step is not None and step_to_start >= proj_switch_step:
        raise ValueError("Start of grad projection has to be after restart step, not at or before it")

    # -------------------------------------
    # Section: Training Step
    # -------------------------------------
    for epoch in range(epoch_to_start, max_epochs):
        if step_number >= max_steps:
            print(f"Reached max steps {max_steps}, stopping the training")
            results_file.flush()
            results_file.close()
            if wandb_run is not None:
                wandb_run.finish()
            break

        # --- Epoch Data Preparation ---
        shuffle = T.randperm(len(X))
        if permute:
            X_shuffled = X[shuffle]
            Y_shuffled = Y[shuffle]
        else:
            X_shuffled = X
            Y_shuffled = Y

        # Checkpoint saving happens in the step loop now, based on step number

        losses_in_epoch = []
        if stop_training:
            break

        # --- Minibatch Iteration ---
        for i in range(0, len(X) // batch_size): # i runs over steps in a epoch
            step_number += 1

            msg = f"{epoch:03d}, {step_number:05d}, "
            # --- Measurement Context and Sampling ---
            ctx = MeasurementContext(
                step_number=step_number,
                batch_size=batch_size,
                epoch=epoch,
                device=str(device),
                lr=optimizer.param_groups[0]['lr'],
                precise_plots=precise_plots,
                rare_measure=rare_measure,
            )

            X_batch = X_shuffled[i*batch_size : (i+1)*batch_size]
            Y_batch = Y_shuffled[i*batch_size : (i+1)*batch_size]
            # Track batch indices for Gauss-Newton quadratic approximation
            if permute:
                batch_indices = shuffle[i*batch_size : (i+1)*batch_size]
            else:
                batch_indices = torch.arange(i*batch_size, (i+1)*batch_size, device=device)

            # -------------------------------------
            # Section: Measurements
            # -------------------------------------
            metrics = measurement_runner.collect(
                ctx=ctx,
                optimizer=optimizer,
                X_batch=X_batch,
                Y_batch=Y_batch,
                epoch=epoch,
                step_in_epoch=i,
                step_number=step_number,
            )

            # --- Epoch-Level Loss Tracking ---
            if metrics['epoch_loss_update'] is not None:
                if math.isnan(metrics['epoch_loss_update']):
                    print('Full loss is NaN, the network prolly diverged, stopping the training')
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None:
                        wandb_run.finish()
                    measurement_runner.close()
                    return
                epoch_loss = metrics['epoch_loss_update']

            if stop_loss is not None and epoch_loss < stop_loss:
                print(f"Loss {epoch_loss} is below the stop loss {stop_loss}, stopping the training")
                stop_training = True
                break

            # -------------------------------------
            # Section: Training Step (Update)
            # -------------------------------------
            optimizer.zero_grad()

            if sde_enabled:
                # SDE integration step - uses full dataset X, Y
                # integrates it for the time [0, eta]
                loss = sde_integration(net=net, X=X, Y=Y, loss_fn=loss_fn, 
                                     batch_size=batch_size, h=sde_h, eta=sde_eta, 
                                     rng=sde_rng)
                
                if math.isinf(loss) or math.isnan(loss):
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None:
                        wandb_run.finish()
                    raise ValueError("Loss is inf or NaN, stopping the training")
                    
                    
            elif gd_noise:
                # this is the GD with noise
                # the whole thing is done in the function, including updating the weights
                loss = gd_with_noise(net=net, X = X, Y=Y, loss_fn=loss_fn, noise_type=gd_noise, 
                                     optimizer=optimizer, batch_size=batch_size, step_number=step_number, 
                                     grad_storage=grad_storage, noise_magnitude=noise_magnitude)
            
            elif quad_approx is not None and quad_approx.is_active:
                # Quadratic approximation dynamics
                quad_gradient = quad_approx.compute_quadratic_gradient(X_batch, Y_batch, batch_indices)
                
                # Get current learning rate from optimizer
                current_lr = optimizer.param_groups[0]['lr']
                if quad_switch_lr is not None:
                    current_lr = quad_switch_lr

                # Update delta using quadratic gradient
                quad_approx.update_delta(current_lr, quad_gradient)
                
                # Set model parameters to current quadratic position
                current_params = quad_approx.get_current_params()
                set_model_params(net, current_params)
                
                # Compute loss for logging (using current model state)
                preds = net(X_batch).squeeze(dim=-1)
                loss = loss_fn(preds, Y_batch)
                
            elif proj_switch_step is not None and step_number >= proj_switch_step:
                # Gradient projection step (only for plain SGD, no momentum/Adam
                    
                # Recompute top-l eigendirections at the requested cadence (default: every step)
                if frequency_calculator.should_measure('proj_eigens_refresh', ctx):
                    ##### TEMP! 
                    full_preds = net(X).squeeze(dim=-1)
                    full_loss_for_eigs = loss_fn(full_preds, Y)
                    _eigvals, eigvecs_block = compute_eigenvalues(
                        full_loss_for_eigs, net,
                        k=proj_top_l if proj_top_l is not None else 1,
                        max_iterations=500,
                        reltol=0.005,
                        eigenvector_cache=eigenvector_cache,
                        return_eigenvectors=True,
                        use_power_iteration=False
                    )

                else:
                    # Use cached eigenvectors
                    if eigenvector_cache is not None and hasattr(eigenvector_cache, 'eigenvectors') and len(eigenvector_cache.eigenvectors) > 0:
                        eigvecs_block = torch.stack(eigenvector_cache.eigenvectors, dim=1).to(device)
                    else:
                        eigvecs_block = None

    

                from utils.lobpcg import _maybe_orthonormalize
                V = eigvecs_block.clone()
                if V.dim() == 1:
                    V = V.unsqueeze(1)
                V = _maybe_orthonormalize(V, assume_ortho=True)

                # --- Projected Step Solve ---
                optimizer.zero_grad()
                params_before_step = flatten_params(net).detach().clone()

                # calculate the step
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)

                if math.isinf(loss.item()) or math.isnan(loss.item()):
                    results_file.flush()
                    results_file.close()
                    if wandb_run is not None:
                        wandb_run.finish()
                    raise ValueError("Loss is inf or NaN, stopping the training")

                # Backward pass for minibatch gradient
                loss.backward()

                optimizer.step()
                
                params_after_step = flatten_params(net).clone()

                # --- Gradient Projection Adjustment ---
                with torch.no_grad():
                    update = params_after_step - params_before_step

                    coeffs = V.T @ update
                    update_in_top = V @ coeffs

                    if proj_to_residual:
                        update_proj = update - update_in_top
                    else:
                        update_proj = update_in_top

                    new_params = params_before_step + update_proj
                    set_model_params(net, new_params)

                    # # Flatten current gradient for logging
                    # params = [p for p in net.parameters() if p.requires_grad]
                    # with torch.no_grad():
                    #     grad_list = [p.grad.reshape(-1) if p.grad is not None else torch.zeros_like(p.reshape(-1)) for p in params]
                    #     g_flat = torch.cat(grad_list).detach().clone()

                    #     coeffs = V.T @ g_flat
                    #     g_in_top = V @ coeffs

                    #     if proj_to_residual:
                    #         g_proj = g_flat - g_in_top
                    #     else:
                    #         g_proj = g_in_top

                    #     denom = torch.linalg.vector_norm(g_flat)
                    #     numer = torch.linalg.vector_norm(g_proj)
                    #     proj_grad_ratio = (numer / (denom + 1e-12)).item()

                    #     projection_basis = V
                    #     params_before_step = flatten_params(net).detach().clone()

            else:
                # Standard SGD step
                if isinstance(net, DDPMWrapper):
                    # DDPM forward pass - loss is computed internally with computational graph
                    loss = net(X_batch)
                    
                    if math.isinf(loss.item()) or math.isnan(loss.item()):
                        results_file.flush()
                        results_file.close()
                        if wandb_run is not None:
                            wandb_run.finish()
                        raise ValueError("Loss is inf or NaN, stopping the training")
                    
                    # Check if we should initialize quadratic approximation
                    if quad_approx is not None:
                        full_dataset = (X, Y) if use_gauss_newton else None
                        quad_approx.initialize_anchor(step_number, loss.item(), full_dataset)
                    
                    # Backward pass for minibatch gradient
                    loss.backward()
                    
                    optimizer.step()
                else:
                    # Standard supervised learning
                    preds = net(X_batch).squeeze(dim=-1)

                    loss = loss_fn(preds, Y_batch)

                    if math.isinf(loss.item()) or math.isnan(loss.item()):
                        results_file.flush()
                        results_file.close()
                        if wandb_run is not None:
                            wandb_run.finish()
                        raise ValueError("Loss is inf or NaN, stopping the training")

                    # Check if we should initialize quadratic approximation
                    if quad_approx is not None:
                        full_dataset = (X, Y) if use_gauss_newton else None
                        quad_approx.initialize_anchor(step_number, loss.item(), full_dataset)

                    # Backward pass for minibatch gradient
                    loss.backward()

                    optimizer.step()


            # Handle loss value (SDE returns float, others return tensor)
            batch_loss = loss if isinstance(loss, float) else loss.item()
            losses_in_epoch.append(batch_loss)

            # --- Checkpoint Handling ---
            # Save checkpoint using wandb system
            checkpoint_path = save_checkpoint_wandb(
                model=net,
                optimizer=optimizer,
                step=step_number,
                epoch=epoch,
                loss=batch_loss,
                run_id=run_id,
                save_every_n_steps=checkpoint_every_n_steps
            )
            if checkpoint_path:
                print(f"Checkpoint saved at step {step_number}: {checkpoint_path}")

            # -------------------------------------
            # Section: Logging (Step)
            # -------------------------------------
            if True: # not results_rarely or (results_rarely and ghg_now):
                # (0) epoch, (1) step, (2) batch loss, (3) full loss, (4) lambda max, (5) step sharpness, (6) batch sharpness, (7) Gradient-Noise Interaction, (8) total accuracy"""
                # Log metrics   
                msg += (
                    f"{batch_loss:7.6f}, {metrics['full_loss']:7.6f}, {metrics['lmax']:6.2f}, "
                    f"{metrics['step_sharpness']:6.1f}, {metrics['batch_sharpness']:6.1f}, "
                    f"{metrics['gni']:6.2f}, {metrics['full_accuracy']:6.2f}"
                )
                results_file.write(msg + "\n")
                
                if wandb_enabled:
                    wandb_metrics = metrics.copy()
                    wandb_metrics.update({
                        "epoch": epoch,
                        "step": step_number,
                        "batch_loss": batch_loss,
                    })
                    rename_map = {
                        "batch_lmax": "batch_lambda_max",
                        "lmax": "lambda_max",
                        "batch_sharpness": "batch_sharpn",
                        "full_gHg": "grad_H_grad",
                        "fisher_batch_eigenval": "batch_fisher_eigenval",
                        "fisher_total_eigenval": "total_fisher_eigenval",
                        "gni": "GNI",
                        "full_accuracy": "accuracy",
                    }
                    for old_key, new_key in rename_map.items():
                        if old_key in wandb_metrics:
                            wandb_metrics[new_key] = wandb_metrics.pop(old_key)
                    wandb_metrics.pop("epoch_loss_update", None)
                    log_metrics(wandb_metrics)

        
        # --- Epoch Finalization ---
        epoch_loss = np.mean(losses_in_epoch)
        
        results_file.flush()

        
    # -------------------------------------
    # Section: Logging
    # -------------------------------------
    # ----- Final Checkpoint Save -----
    # Save final checkpoint
    final_checkpoint_path = save_checkpoint_wandb(
        model=net,
        optimizer=optimizer,
        step=step_number,
        epoch=epoch,
        loss=batch_loss,
        run_id=run_id,
        save_every_n_steps=1  # Always save final checkpoint
    )
    print(f"Final checkpoint saved: {final_checkpoint_path}")

    results_file.close()

    measurement_runner.close()

    # ----- WandB Teardown -----
    if wandb_run is not None:
        wandb_run.finish()

    # ----- Final Reporting -----
    end_time = time.time()
    print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training time: {end_time - start_time:.2f} seconds")



    # ----- Optional Final Measurements -----
    if 'final' in measurements:
        final_file = save_to / 'final.json'
        final_file = open(final_file, 'w') 

        # do the final measurements here - depending on what is needed
    




if __name__ == '__main__':
    # -------------------------------------
    # Section: Runtime Setup
    # -------------------------------------
    # ----- Reproducibility Seeds -----
    seed = 88881
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # -------------------------------------
    # Section: Argument Parser
    # -------------------------------------
    parser = argparse.ArgumentParser(description='Training script')
    # --- Training Parameters ---
    parser.add_argument('--batch', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train. Either epochs or steps should be provided')
    parser.add_argument('--cpu', action='store_true', help='Force training to run on CPU even if CUDA is available')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--stop-loss', '--stop_loss', type=float, default=None, help='Stop training if loss goes below this value')
    # --- Loss Configuration ---
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss function to use (mse or ce)')

    # --- Dataset Configuration ---
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--classes', type=int, nargs=2, default=[1, 9], help='Two class labels to use for training. Default is [1, 9], as being probably the most difficult classes to separate')
    parser.add_argument('--num-data', '--num_data', type=int, default=1024, help='Number of datapoints to train on')

    # --- Model Configuration ---
    parser.add_argument('--model', type=str, default='mlp', help='Network architecture to use for training')
    parser.add_argument('--init-scale', '--init_scale', type=float, default=0.2, help='Initialization scale for network weights')
    parser.add_argument('--no-init', '--no_init', action='store_true', help='If set, do not initialize network weights')

    # --- wandb Continuation Options ---
    parser.add_argument('--cont-run-id', '--cont_run_id', type=str, default=None, help='Wandb run ID to continue training from')
    parser.add_argument('--cont-step', '--cont_step', type=int, default=None, help='Step to continue training from (uses closest available checkpoint)')
    parser.add_argument('--checkpoint-every', '--checkpoint_every', type=int, default=None, help='Save checkpoint every N steps (default: auto-calculated based on total steps)')

    # --- Optimizer Variants ---
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD optimizer')
    parser.add_argument('--adam', action='store_true', help='If set, use Adam optimizer instead of SGD')
    
    # --- Measurement Flags (Primary) ---
    # parser.add_argument('--fullbs', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--lambdamax', '--lmax', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--batch-sharpness', '--batch-sharpness-step', '--bs', action='store_true', dest='batch_sharpness',
                        help='If set, compute the batch sharpness: E[gHg/g²] with the expectation taken across mini-batches. Use --batch-sharpness-step for backward compatibility.')
    parser.add_argument('--step-sharpness', action='store_true', dest='step_sharpness',
                        help='If set, compute the step sharpness: the current-mini-batch Rayleigh quotient g·Hg/g². Average across steps to recover the traditional batch sharpness.')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity.')

    # --- Measurement Flags (Secondary, aka still useful) ---
    parser.add_argument('--hessian-trace', action='store_true', help='Estimate the trace of the full-batch loss Hessian via a Hutchinson-style estimator')
    parser.add_argument('--grad-projection', action='store_true', help='Compute grad_projection_i: fraction of full-batch gradient lying in span of top-i cached Hessian eigenvectors (i up to 20); uses cached eigenvectors only; only for plain SGD')
    parser.add_argument('--one-step-loss-change', action='store_true', help='If set, compute the expected one-step change in loss using Monte Carlo estimation')
    parser.add_argument('--gradient-norm', action='store_true', help='If set, compute the Monte Carlo estimate of squared norm of mini-batch gradients')
    parser.add_argument('--final', action='store_true', help='If set, compute the lambda_max and step sharpness at the end')

    
    # --- Measurement Flags (Tertiary, aka almost completely useless) ---
    parser.add_argument('--batch-sharpness-exp-inside', action='store_true', help='If set, compute the batch sharpness using E[gHg]/E[g²], where the expectation is inside the ratio. Compare with step-sharpness, where the expectation stays outside the ratio.')
    parser.add_argument('--batch-lambdamax','--batchlmax', action='store_true', help='If set, compute the batch lambda_max(H_B), aka batch lambda max')
    parser.add_argument('--fisher', action='store_true', help='If set, compute Fisher information matrix eigenvalue. Currently only works with one-dim output')
    parser.add_argument('--param-distance', '--param_distance', action='store_true', help='If set, compute the distance from the reference weights')
    parser.add_argument('--param-file', '--param_file', type=str, default=None, help='Path to reference parameters for computing parameter distance')

    # --- Measurement Configuration ---
    parser.add_argument('--disable-cache-eigenvectors', '--disable_cache_eigenvectors', action='store_true', help='If set, disable eigenvector caching for warm starts to improve eigenvalue computation performance')
    parser.add_argument('--use-power-iteration', '--use_power_iteration', action='store_true', help='If set, use power iteration method instead of LOBPCG for eigenvalue computation')
    parser.add_argument('--num-eigenvalues', '--num_eigenvalues', '--k', type=int, default=1, help='Number of eigenvalues to compute when computing lambda_max (default: 1)')

    parser.add_argument('--results-rarely', '--results_rarely', action='store_true', help='If set, results will be recorded less frequently')
    parser.add_argument('--precise-plots', action='store_true', help='Enable more frequent measurements for precise plotting')
    parser.add_argument('--rare-measure', dest='rare_measure', action='store_true', help='Activate regime where expensive measurements are performed rarely')

    # --- Noise Configuration ---
    parser.add_argument('--gd-noise', '--gd_noise', type=str, default=None, help='Do noisy GD, to simulate SGD. Supported noises: sgd, diag, iso, const')
    parser.add_argument('--noise-mag', '--noise_mag', type=float, default=None, help='The noise magnitude for the constant noise')
    
    # --- SDE Configuration ---
    parser.add_argument('--sde', action='store_true', help='Simulate the SDE dynamics (the one that correspond to the SGD). It integrates the SDE using the Euler-Maruyama method')
    parser.add_argument('--sde-h', '--sde_h', type=float, default=0.01, help='SDE *integration* time step size (default: 0.01)')
    parser.add_argument('--sde-eta', '--sde_eta', type=float, default=None, help='Learning rate for SDE (uses --lr if not specified)')
    parser.add_argument('--sde-seed', '--sde_seed', type=int, default=888, help='Random seed for SDE noise generation (default: 888)')

    # --- Quadratic Approximation Configuration ---
    parser.add_argument('--quad-switch-step', '--quad_switch_step', type=int, default=None, help='Step at which to switch from true NN dynamics to quadratic Taylor approximation dynamics')
    parser.add_argument('--use-gauss-newton', '--use_gauss_newton', action='store_true', help='Use Gauss-Newton matrix instead of Hessian for quadratic approximation')
    parser.add_argument('--quad-switch-lr', '--quad_switch_lr', type=float, default=None, help='lr to use after switching, used to test explosion')

    # --- Gradient Projection Configuration ---
    parser.add_argument('--proj-switch-step', dest='proj_switch_step', type=int, default=None,
                        help='Step number to start projecting minibatch gradient onto top-l Hessian eigendirections (full batch)')
    parser.add_argument('--proj-top-l', dest='proj_top_l', type=int, default=None,
                        help='Number of top Hessian eigendirections to project against/onto after switch step')
    parser.add_argument('--proj-to-residual', dest='proj_to_residual', action='store_true',
                        help='After --proj-switch-step, apply gradient projected to orthogonal complement of top-l eigenspace')

    # --- Randomness Settings ---
    parser.add_argument('--dataset-seed', '--dataset_seed', type=int, default=888, help='Random seed for dataset preparation')
    parser.add_argument('--init-seed', '--init_seed', type=int, default=8888, help='Random seed for network initialization')

    # --- wandb Settings ---
    parser.add_argument('--wandb-tag', type=str, default=None, help='Tag to add to the wandb run')
    parser.add_argument('--wandb-name', type=str, default=None, help='Optional suffix appended to default wandb run name (sanitized)')
    parser.add_argument('--wandb-notes', type=str, default=None, help='Optional notes/description attached to the wandb run')
    parser.add_argument('--disable-wandb', action='store_true', help='Disable Weights & Biases logging for debugging/testing')

    # ----- Argument Parsing -----
    args = parser.parse_args()

    # ----- wandb Availability Check -----
    wandb_installed = is_wandb_available()
    if not wandb_installed and not args.disable_wandb:
        print("wandb is not installed; proceeding with logging disabled. Re-run with --disable-wandb to silence this message.")
        args.disable_wandb = True
    elif args.disable_wandb:
        print("wandb logging disabled by flag (--disable-wandb).")

    wandb_enabled = wandb_installed and not args.disable_wandb


    # -------------------------------------
    # Section: Experiment Setup
    # -------------------------------------
    # ----- Argument Post-processing -----
    # --- Parameter Extraction ---
    batch_size = args.batch
    dataset = args.dataset
    if args.cpu:
        if T.cuda.is_available():
            print('CUDA is available but running on CPU due to --cpu flag.')
        device = 'cpu'
    else:
        device = T.device('cuda') if T.cuda.is_available() else 'cpu'

    if args.momentum is not None and args.adam:
        raise ValueError("You should provide either momentum or adam, not both")

    if args.momentum is not None and args.momentum < 1e-4 and not args.adam:
        args.momentum = None  # if momentum is too small, just use SGD without momentum

    
    # --- Argument Validation ---
    if args.final:
        raise ValueError("--final needs to be re-implemented")

    if args.param_distance:
        raise NotImplementedError("--param-distance needs to be re-implemented")

    if args.steps is not None and args.epochs is not None:
        raise ValueError("You should provide either epochs or steps, not both")

    # Validate gradient projection feature flags and conflicts
    if (args.proj_switch_step is not None) or (args.proj_top_l is not None) or args.proj_to_residual:
        if args.proj_switch_step is None or args.proj_top_l is None:
            raise ValueError("Gradient projection requires both --proj-switch-step and --proj-top-l")
        if args.proj_top_l < 1:
            raise ValueError("--proj-top-l must be a positive integer")
        if args.adam or (args.momentum is not None and args.momentum != 0):
            raise ValueError("Gradient projection currently supports only plain SGD (no momentum/Adam)")
    
    # Validate wandb continuation arguments
    if (args.cont_run_id is not None) != (args.cont_step is not None):
        raise ValueError("Both --cont-run-id and --cont-step must be provided together for wandb continuation")

    # Check for mutually exclusive training modes
    exclusive_modes = []
    if args.proj_switch_step is not None:
        exclusive_modes.append("gradient projection (--proj-switch-step)")
    if args.sde:
        exclusive_modes.append("SDE dynamics (--sde)")
    if args.gd_noise is not None:
        exclusive_modes.append("GD with noise (--gd-noise)")
    if args.quad_switch_step is not None:
        exclusive_modes.append("quadratic approximation (--quad-switch-step)")
    
    if len(exclusive_modes) > 1:
        raise ValueError(f"Cannot use multiple training modes simultaneously: {', '.join(exclusive_modes)}. Please choose only one.")
    
    # ----- Measurement Selection -----
    measurements = {name for name, enabled in [
    ('lmax', args.lambdamax),
    ('batch_lmax', args.batch_lambdamax),
    ('step_sharpness', args.step_sharpness),
    ('batch_sharpness', args.batch_sharpness),
    ('batch_sharpness_exp_inside', args.batch_sharpness_exp_inside),
    ('grad_projection', args.grad_projection),
    ('gradient_norm', args.gradient_norm),
    ('one_step_loss_change', args.one_step_loss_change),
    ('gni', args.gni),
    ('fisher', args.fisher),
    ('final', args.final),
    ('param_distance', args.param_distance),
    ('hessian_trace', args.hessian_trace),
    ] if enabled}

    # ----- Result Storage Setup -----
    RES_FOLDER.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, RES_FOLDER)
    step_to_start = 0
    
    # ----- Loss Function Selection -----
    if args.loss == 'mse':
        loss_fn = SquaredLoss()
    elif args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()

    # ----- Dataset and Model Presets -----
    dataset_presets = get_dataset_presets()
    model_presets = get_model_presets()

    # --- Dataset Preparation ---
    data = prepare_dataset(dataset, DATASET_FOLDER, args.num_data, args.classes, args.dataset_seed, loss_type=args.loss)

    # --- Model Construction ---
    name = args.model
    params = model_presets[name]['params']
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']
    net = prepare_net(
        model_type=model_presets[name]['type'], 
        params=params
        )

    # --- Model Initialization ---
    if not args.no_init:
        initialize_net(net, scale=args.init_scale, seed=args.init_seed)

    # ----- Checkpoint Continuation Handling -----
    wandb_checkpoint_loaded = False
    epoch_to_start = 0
    if args.cont_run_id is not None and args.cont_step is not None:
        # Continue from wandb checkpoint
        checkpoint_dir = get_checkpoint_dir_for_run(args.cont_run_id)
        if checkpoint_dir is None:
            raise FileNotFoundError(f"Cannot find checkpoint directory for run ID: {args.cont_run_id}")
        
        checkpoint_info = find_closest_checkpoint_wandb(args.cont_step, checkpoint_dir=checkpoint_dir)
        if checkpoint_info is None:
            raise FileNotFoundError(f"No suitable checkpoint found for step {args.cont_step} in run {args.cont_run_id}")
        
        
        if args.adam:
            # loaded_data = load_checkpoint_wandb(checkpoint_info, net, optimizer)
            raise ValueError("Cannot continue from wandb checkpoint with Adam optimizer (only SGD is supported). With Adam need to also keep the state, which is not implemented yet")
        
        loaded_data = load_checkpoint_wandb(checkpoint_info, net)
        step_to_start = loaded_data['step']
        epoch_to_start = loaded_data['epoch']
        wandb_checkpoint_loaded = True
        
        print(f"Loaded checkpoint from step {loaded_data['step']} (epoch {loaded_data['epoch']}) from run {args.cont_run_id}")
        print(f"Closest checkpoint to requested step {args.cont_step}: actual step {loaded_data['step']}")
        
        # Handle quadratic approximation continuation
        if args.quad_switch_step is not None and loaded_data['step'] >= args.quad_switch_step:
            print(f"Warning: Continuing from step {loaded_data['step']} which is at or after quad_switch_step {args.quad_switch_step}")
            print("Quadratic approximation will be initialized as active from the start.")

    # ----- wandb Initialization -----
    wandb_run = None
    wandb_run_id = None
    if wandb_enabled:
        wandb_run = init_wandb(args, step_to_start)
        wandb_run_id = wandb_run.id
    else:
        wandb_run_id = generate_run_id()

    # ----- Reference Parameter Handling -----
    # Load the reference parameters to calculate the distance from (if provided)
    param_reference = None
    if args.param_distance:
        if args.param_file is None:
            # Create a zero vector as a reference if no param_file is provided
            print("No parameter file provided. Creating a zero vector as reference.")
            param_reference = []
            for param in net.parameters():
                param_reference.append(torch.zeros_like(param).flatten())
            param_reference = torch.cat(param_reference)
    if args.param_file is not None:
        param_reference = T.load(args.param_file, map_location=device)
        # param_reference = param_reference['model_state_dict']
        # param_reference = {k: v.to(device) for k, v in param_reference.items()}

    # ----- Optimizer Preparation -----
    optimizer = prepare_optimizer(net, args.lr, args.momentum, args.adam)

    # ----- Checkpoint Cadence Determination -----
    if args.checkpoint_every is not None:
        checkpoint_every_n_steps = args.checkpoint_every
    else:
        checkpoint_every_n_steps = max(args.steps // 200, 1) if args.steps else None
    
    # ----- Training Invocation -----
    train(
        net=net,
        optimizer=optimizer,
        data=data,
        max_epochs=args.epochs,
        max_steps=args.steps,
        batch_size=args.batch,
        save_to=run_folder,
        device=device,
        loss_fn=loss_fn,
        verbose=True,
        stop_loss = args.stop_loss,
        epoch_to_start=epoch_to_start,
        step_to_start=step_to_start,
        gd_noise=args.gd_noise,
        noise_magnitude=args.noise_mag,
        results_rarely=args.results_rarely,
        measurements=measurements,
        param_reference=param_reference,
        cache_eigenvectors = not args.disable_cache_eigenvectors,
        sde_enabled=args.sde,
        sde_h=args.sde_h,
        sde_eta=args.sde_eta,
        sde_seed=args.sde_seed,
        use_power_iteration=args.use_power_iteration,
        num_eigenvalues=args.num_eigenvalues,
        checkpoint_every_n_steps=checkpoint_every_n_steps,
        quad_switch_step=args.quad_switch_step,
        quad_switch_lr=args.quad_switch_lr,
        use_gauss_newton=args.use_gauss_newton,
        precise_plots=args.precise_plots,
        rare_measure=args.rare_measure,
        proj_switch_step=args.proj_switch_step,
        proj_top_l=args.proj_top_l,
        proj_to_residual=args.proj_to_residual,
        wandb_run=wandb_run,
        wandb_enabled=wandb_enabled,
        wandb_run_id=wandb_run_id,
    )
