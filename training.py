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
from utils.storage import initialize_folders
from utils.noise import gd_with_noise, GradStorage
from utils.measure import *

from torch.autograd import grad
import json


if 'DATASETS' not in os.environ:
    raise ValueError("Please set the environment variable 'DATASETS'. Use 'export DATASETS=/path/to/datasets'")
if 'RESULTS' not in os.environ:
    raise ValueError("Please set the environment variable 'RESULTS'. Use 'export RESULTS=/path/to/results'")

DATASET_FOLDER = Path(os.environ.get('DATASETS'))
RES_FOLDER = Path(os.environ.get('RESULTS'))



### ALL THE CONDITIIONS WHEN TO CALCULATE SHARPNESS ETC



def calculate_full_ghg_condition(i, step_number, batch_size, initial_sharpness=0, sharpness_every=None):
    if batch_size > 32:
        how_often = 256
    else:
        how_often = 512
    
    if step_number > 10000:
        how_often = how_often * 2

    return step_number % how_often == 0

def calculate_batch_ghg_condition(i, step_number, batch_size, initial_sharpness=0, sharpness_every=None):
    return step_number % 8 == 0


def calculate_fisher_total_condition(i, step_number, batch_size, initial_sharpness=0, sharpness_every=256):
    if batch_size > 32:
        how_often = 64 * 2
    else:
        how_often = 128 * 2
    
    if step_number > 10000:
        how_often = how_often * 2
    
    if step_number > 20000:
        how_often = how_often * 2

    return step_number % how_often == 0

def calculate_fisher_batch_condition(i, step_number, batch_size, initial_sharpness=0, sharpness_every=256):
    if batch_size > 32:
        how_often = 8
    else:
        how_often = 4

    if step_number > 10000:
        how_often = how_often * 2
    
    return step_number % how_often == 0


##### THE ACTUAL TRAINING

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
            initial_sharpness=0,
            epoch_to_start=0,
            step_to_start=0,
            sharpness_every=None,
            gd_noise=False,
            noise_magnitude=None,
            results_rarely: bool = False,
            measurements: set = {},
            param_reference = None,  # reference weights to measure distance from during training
            ):
    

    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")


    # COMPUTE_GHG = True

    NET_SAVES_PER_TRAINING = 200


    # if epochs is None:
    #     epochs = steps // (len(data[0]) // batch_size) + 1

    assert max_epochs is not None or max_steps is not None
    if max_epochs is None:
        max_epochs = 1000000

    X_train, Y_train, X_test, Y_test = data

    X, Y = X_train, Y_train

    net = net.to(device)
    net.train()
    net.float()

    RECORD_EVERY = 20

    # Create the save_to directory if it doesn't exist
    save_to.mkdir(parents=True, exist_ok=True)

    # paths
    model_save_path = save_to / 'checkpoints'

    results_file = save_to / 'results.txt'
    if device == 'cpu':
        # No buffering on CPU to ensure writes happen immediately
        results_file = open(results_file, 'a', buffering=1)
        torch.set_num_threads(40)
    else:
        # Use buffering on GPU for better performance
        # TO CHANGE
        results_file = open(results_file, 'a', buffering=1_00)

    final_file = save_to / 'final.json'
    final_file = open(final_file, 'w')  


    X = X.to(device)
    Y = Y.to(device)

    step_number = -1 if step_to_start == 0 else step_to_start

    if gd_noise is not None:
        grad_storage = GradStorage(net, recalculate_every=10)

    # calculate epochs to save - total we want at most 100 saves for the whole training
    
    steps_in_epoch = len(X) // batch_size
    epochs_expected = max_steps // steps_in_epoch
    save_net_every = max(epochs_expected // NET_SAVES_PER_TRAINING, 1)



    epoch_loss = float('+inf')
    stop_training = False


    # Training loop
    for epoch in range(epoch_to_start, max_epochs):
        if step_number >= max_steps:
            print(f"Reached max steps {max_steps}, stopping the training")
            results_file.flush()
            results_file.close()
            break

        shuffle = T.randperm(len(X))
        if permute:
            X_shuffled = X[shuffle]
            Y_shuffled = Y[shuffle]
        else:
            X_shuffled = X
            Y_shuffled = Y

        # save the model
        model_save_file = model_save_path / f'net_{epoch}.pt'
        if save_net_every == 0 or epoch % save_net_every == 0:
            T.save(net.state_dict(), model_save_file)


        losses_in_epoch = []
        if stop_training:
            break

        for i in range(0, len(X) // batch_size): # i runs over steps in a epoch
            step_number += 1

            msg = f"{epoch:03d}, {step_number:05d}, "

            X_batch = X_shuffled[i*batch_size : (i+1)*batch_size]
            Y_batch = Y_shuffled[i*batch_size : (i+1)*batch_size]

            ##### LMBDA MAX WORK ####

            ### condition to calculate the lambda max ###
            TEMP_OVERRIDE = False
            if 'gni' in measurements: TEMP_OVERRIDE = True 
            lmax_now = False
            
            if 'lmax' in measurements:
                if TEMP_OVERRIDE:
                    def do_lmax():
                        FIRST_FEW = 256
                        FIRST_SUPER_FEW = 128
                        if step_number < FIRST_SUPER_FEW:
                            return True
                        
                        if step_number < FIRST_FEW:
                            return step_number % 4 == 0

                        how_often = 256
                        if step_number > 10000:
                            how_often = how_often * 2
                        if step_number > 20000:
                            how_often = how_often * 2
                        if step_number > 100_000:
                            how_often = how_often * 2
                        fullbs_now = step_number % how_often == 0
                        return fullbs_now
                    
                    lmax_now = do_lmax()

                else:
                    def do_lmax():
                        if batch_size <= 32:
                            how_often = 256
                        else:
                            how_often = 128
                        
                        if step_number > 10000:
                            how_often = how_often * 2
                        if step_number > 20000:
                            how_often = how_often * 2
                        if step_number > 100_000:
                            how_often = how_often * 2
                        return step_number % how_often == 0
                    
                    lmax_now = do_lmax()

            if lmax_now:
                # Clear CUDA cache if we're using GPU
                if str(device).startswith('cuda'):
                    torch.cuda.empty_cache()
                optimizer.zero_grad()

                # # TODO
                # LMAX_MAX_SIZE = 8192
                # # Check available CUDA memory before full batch computation
                # if str(device).startswith('cuda'):
                #     total_memory = torch.cuda.get_device_properties(0).total_memory
                #     if total_memory < 20 * 1024**3:  # Less than 20GB
                #         if isinstance(net, CNN):
                #             LMAX_MAX_SIZE = 2048 + 512
                #         if isinstance(net, ResNet):
                #             LMAX_MAX_SIZE = 512
                LMAX_MAX_SIZE = 1_000_000 # a placeholder originally used not to compute the lambda max on the whole dataset

                    
                if len(X) > LMAX_MAX_SIZE:
                    # If the dataset is too large, take a random subset
                    subset_indices = np.random.choice(len(X), LMAX_MAX_SIZE, replace=False)
                    X_subset = X[subset_indices]
                    Y_subset = Y[subset_indices]
                else:
                    # Use the whole dataset
                    X_subset = X
                    Y_subset = Y

                preds = net(X_subset).squeeze(dim=-1)

                loss = loss_fn(preds, Y_subset)
                lmax = compute_lambdamax(loss, net, max_iterations=100, 
                                              epsilon=1e-4)
                
                # if COMPUTE_GHG:
                #     sharpness, gHg = sharpness
                #     gHg = gHg.item()
                # else:
                #     gHg = np.nan

                lmax = lmax.item()
                full_gHg = np.nan

                print(f"Epoch {epoch+1}, Step {i}: Total lambda max = {lmax}, Loss = {loss.item()} !!!")
                # total_lmax = lmax
                # total_gHg = gHg
                full_loss = loss.item()

                full_accuracy = calculate_accuracy(preds, Y_subset)



                if math.isnan(full_loss):
                    print("Full loss is NaN, the network prolly diverged, stopping the training")
                    results_file.flush()
                    results_file.close()
                    return
                
                epoch_loss = full_loss

            else:
                lmax = np.nan
                # total_lmax = np.nan
                full_loss = np.nan
                full_gHg = np.nan
                # total_gHg = np.nan
                full_accuracy = np.nan
            
            if stop_loss is not None and epoch_loss < stop_loss:
                print(f"Loss {epoch_loss} is below the stop loss {stop_loss}, stopping the training")
                stop_training = True
                break


            ###### BATCH LAMBDAMAX WORK  ######
            batch_lmax_now = False
            if 'batch_lmax' in measurements:
                if gd_noise is None:
                    how_often = 16
                    if step_number > 50_000:
                        how_often = 32
                    if step_number > 100_000:
                        how_often = 64
                    
                    batch_lmax_now = step_number % how_often == 0
                    
                else:
                    raise ValueError("There should be some value here, but it is not implemented yet")

            if batch_lmax_now:
                optimizer.zero_grad()
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)
                batch_lmax = compute_lambdamax(loss, net, max_iterations=50, 
                                                    epsilon=1e-3)
                batch_lmax = batch_lmax.item()

                #     gHg = gHg.item()
                #     batch_gHg = gHg
                # else:
                # batch_gHg = np.nan
                # batch_sharpness = batch_sharpness.item()

                if i % RECORD_EVERY == 0:
                    print(f"Epoch {epoch+1}, Step {i}: Batch Lambda Max = {batch_lmax}, Loss = {loss.item()}")
                
            else:
                batch_lmax = np.nan
                # batch_gHg = np.nan

            ###### BATCH SHARPNESS WORK ######
            batch_sharpness_now = False
            if 'batch_sharpness' in measurements:
                how_often = 8
                batch_sharpness_now = step_number % how_often == 0
            
            if batch_sharpness_now:
                net.zero_grad()
                preds = net(X_batch).squeeze(dim=-1)
                loss = loss_fn(preds, Y_batch)
                batch_sharpness = compute_grad_H_grad(loss,
                                    net)
                batch_sharpness = batch_sharpness.item()
            
            else:
                batch_sharpness = np.nan
            
            ####### STATIC BATCH SHARPNESS WORK #######
            # frequency to calculate it
            batch_sharpness_static_now = False
            if 'batch_sharpness_static' in measurements:
                if batch_size < 32:
                    how_often = 128
                else:
                    how_often = 64
                if step_number > 5000:
                    how_often = how_often * 2
                if step_number > 50000:
                    how_often = how_often * 2
                
                batch_sharpness_static_now = step_number % how_often == 0


            batch_sharpness_static = np.nan
            if batch_sharpness_static_now:
                batch_sharpness_static = calculate_averaged_grad_H_grad(net,
                                                  X,
                                                  Y,
                                                  loss_fn,
                                                  batch_size=batch_size,
                                                  n_estimates=600,
                                                  tolerance = 0.01
                )
            

            ##### FISHER WORK #####

            fisher_total_eigenval = np.nan
            fisher_batch_eigenval = np.nan

            if 'fisher' in measurements:
                if calculate_fisher_total_condition(i, step_number, batch_size, initial_sharpness, sharpness_every):
                    fisher_total_eigenval = compute_fisher_eigenvalues(net, X).item()
                
                if calculate_fisher_batch_condition(i, step_number, batch_size, initial_sharpness, sharpness_every):
                    fisher_batch_eigenval = compute_fisher_eigenvalues(net, X_batch).item()
            

            ##### GNI work ####
            # frequency to calculate it
            gni_now = False
            if 'gni' in measurements:
                if batch_size < 32:
                    how_often = 256
                else:
                    how_often = 64
                
                if step_number > 5000:
                    pass
                    # how_often = how_often * 2
                
                gni_now = step_number % how_often == 0

                if step_number - step_to_start < 8:
                    gni_now = True
            
            gni = np.nan
            if gni_now:
                gni = calculate_averaged_gni(
                    net=net,
                    X=X,
                    Y=Y,
                    loss_fn=loss_fn,
                    batch_size=batch_size,
                    n_estimates=500,
                    tolerance=0.01
                )
            
            ##### WEIGHT DISTANCE WORK ####

            param_distance = np.nan

            param_distance_now = False
            if 'param_distance' in measurements:
                if param_reference is None:
                    raise ValueError("You should provide a reference weights to measure distance from")
                if batch_size < 32:
                    how_often = 1
                else:
                    how_often = 1
                
                param_distance_now = step_number % how_often == 0

            if param_distance_now:
                # calculate the distance from the reference weights
                param_distance = calculate_param_distance(net, param_reference)
                param_distance = param_distance.item()



            
            # now calculate the total loss for GNI
            FIRST_FEW = 32
            full_loss_now = False
            if 'gni' in measurements:
                if step_number - step_to_start < FIRST_FEW:
                    full_loss_now = True
                
                how_often = 32
                if step_number % how_often == 0:
                    full_loss_now = True


                if full_loss_now:
                    X_subset = X
                    Y_subset = Y
                    preds = net(X_subset).squeeze(dim=-1)

                    loss = loss_fn(preds, Y_subset)
                    full_loss = loss.item()


            ######## SGD STEP #######
            optimizer.zero_grad()

            if not gd_noise:
                preds = net(X_batch).squeeze(dim=-1)

                loss = loss_fn(preds, Y_batch)

                if math.isinf(loss.item()) or math.isnan(loss.item()):
                    results_file.flush()
                    results_file.close()
                    raise ValueError("Loss is inf or NaN, stopping the training")

                # Backward pass
                loss.backward()
                
                optimizer.step()

            else:
                # this is the GD with noise
                # the whole thing is done in the function, including updating the weights
                loss = gd_with_noise(net=net, X = X, Y=Y, loss_fn=loss_fn, noise_type=gd_noise, 
                                     optimizer=optimizer, batch_size=batch_size, step_number=step_number, 
                                     grad_storage=grad_storage, noise_magnitude=noise_magnitude)
            

            batch_loss = loss.item()
            losses_in_epoch.append(batch_loss)

            ############## RECORD THE RESULTS ##############
            if True: # not results_rarely or (results_rarely and ghg_now):
                msg += f"{batch_loss:7.6f}, {full_loss:7.6f}, {batch_lmax:6.2f}, {lmax:6.2f}, {batch_sharpness:6.1f}, {full_gHg:6.1f}, {fisher_batch_eigenval:6.2f}, {fisher_total_eigenval:6.1f}, {batch_sharpness_static:6.2f}, {gni:6.2f}, {full_accuracy:6.2f}, {param_distance:.7e}"
                results_file.write(msg + "\n")

        
        # end of epoch
        epoch_loss = np.mean(losses_in_epoch)
        
        results_file.flush()

        
    
    T.save(net.state_dict(), model_save_path / f'net_final.pt')

    results_file.close()

    end_time = time.time()
    print(f"Training finished at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total training time: {end_time - start_time:.2f} seconds")

    if 'final' in measurements:
        final_sharpness = {}
        # first, minibs
        minibs_start_time = time.time()

        # minibs = calculate_averaged_lambdamax(net, 
        #                              X, 
        #                              Y, 
        #                              loss_fn, 
        #                              batch_size, 
        #                              n_estimates=500,
        #                             #  min_estimates=max((len(X) // max(batch_size, 2)) // 10, 20),
        #                             min_estimates=20,
        #                              tolerance=0.005,
        #                              max_hessian_iters=100,
        #                              hessian_tolerance=1e-3,
        #                              compute_gHg=True
        #                              )
        
        
        # minibs_end_time = time.time()
        # print(f"Time taken to calculate minibs: {minibs_end_time - minibs_start_time:.2f} seconds")



        STEPS_TO_DO = 1000 if batch_size < 32 else 1000

        batch_sharpnesses = []

        for i in range(STEPS_TO_DO):
            shuffle = T.randperm(len(X))
            X_batch = X[shuffle][:batch_size]
            Y_batch = Y[shuffle][:batch_size]

            if i % 2 == 0:
                preds = net(X_batch).squeeze(dim=-1)
                loss = loss_fn(preds, Y_batch)
                batch_sharpness = compute_grad_H_grad(loss, net)
                batch_sharpness = batch_sharpness.item()

                batch_sharpnesses.append(batch_sharpness)

        
            optimizer.zero_grad()
            preds = net(X_batch).squeeze(dim=-1)
            loss = loss_fn(preds, Y_batch)
            loss.backward()
            optimizer.step()
        
        final_sharpness['batch_sharpness'] = batch_sharpnesses

        json.dump(final_sharpness, final_file, indent=4)

        

        # then, full sharpness
        if True:
            fullbs_start_time = time.time()
            fullbs = calculate_averaged_lambdamax(net,
                                                  X, 
                                                  Y, 
                                                  loss_fn, 
                                                  batch_size=len(X), 
                                                  n_estimates=1, 
                                                    min_estimates=1, 
                                                    max_hessian_iters=1000,
                                                    hessian_tolerance=1e-4,
                                                    )

            fullbs_end_time = time.time()
            print(f"Time taken to calculate fullbs: {fullbs_end_time - fullbs_start_time:.2f} seconds")
        
            final_sharpness['lambda_max'] = fullbs

        # this is so not to write the data twice
        final_file.seek(0)
        json.dump(final_sharpness, final_file, indent=4)

    final_file.flush()
    final_file.close()



if __name__ == '__main__':
    seed = 888
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

    # Command line arguments
    parser = argparse.ArgumentParser(description='Training script')
    # Training parameters
    parser.add_argument('--batch', type=int, default=64, help='Input batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train')
    parser.add_argument('--steps', type=int, default=10000, help='Number of steps to train. Either epochs or steps should be provided')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
    parser.add_argument('--stop_loss', type=float, default=None, help='Stop training if loss goes below this value')
    # Loss function
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'ce'], help='Loss function to use (mse or ce)')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset to use for training')
    parser.add_argument('--classes', type=int, nargs=2, default=[1, 9], help='Two class labels to use for training. Default is [1, 9], as being probably the most difficult classes to separate')
    parser.add_argument('--num_data', type=int, default=1024, help='Number of datapoints to train on')

    # Model configuration
    parser.add_argument('--model', type=str, default='mlp', help='Network architecture to use for training')
    parser.add_argument('--init_scale', type=float, default=None, help='Initialization scale for network weights')
    parser.add_argument('--no_init', action='store_true', help='If set, do not initialize network weights')

    # Continuation options
    parser.add_argument('--cont_folder', type=str, default=None, help='Folder to continue training from')
    parser.add_argument('--cont_epoch', type=int, default=0, help='Epoch to continue training from')
    parser.add_argument('--cont_last', action='store_true', help='If set, continue training from the last run that fits the parameters')

    # Momentum and adam
    parser.add_argument('--momentum', type=float, default=None, help='Momentum for SGD optimizer')
    parser.add_argument('--adam', action='store_true', help='If set, use Adam optimizer instead of SGD')

    # Measurement settings
    parser.add_argument('--sharp_every', type=int, help='Frequency of sharpness computation')
    parser.add_argument('--init_sharp', type=int, default=0, help='Compute total sharpness for the first n steps')
    
    # Measurement settings (main)
    # parser.add_argument('--fullbs', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--lambdamax', '--lmax', action='store_true', help='If set, compute the lambda_max, aka FullBS')
    parser.add_argument('--batch-sharpness', '--bs',  action='store_true', 
                        help='If set, compute the batch sharpness in one step (aka not averaged, aka gradient-Hessian-gradient without the expectation). Average over multiple steps (and thus batches to get the batch sharpness). Compare with batch-sharpness-static, where the average is done over multiple batches within one step.')
    parser.add_argument('--batch-lambdamax','--batchlmax', action='store_true', help='If set, compute the batch lambda_max(H_B), aka batch lambda max')
    parser.add_argument('--batch-sharpness-static', action='store_true', help='If set, compute the ')
    parser.add_argument('--gni', action='store_true', help='If set, compute the Gradient-Noise Interaction quantity')

    # Measurement settings (secondary)
    parser.add_argument('--fisher', action='store_true', help='If set, compute Fisher information matrix eigenvalue. Currently only works with one-dim output')
    parser.add_argument('--param_distance', action='store_true', help='If set, compute the distance from the reference weights')
    parser.add_argument('--param_file', type=str, default=None, help='Path to reference parameters for computing parameter distance')
    parser.add_argument('--final', action='store_true', help='If set, compute the lambda_max and batch sharpness at the end')


    parser.add_argument('--results_rarely', action='store_true', help='If set, results will be recorded less frequently')

    # Noise configuration
    parser.add_argument('--gd_noise', type=str, default=None, help='Do GD, but with gaussian noise to simulate SGD. Supoprted noises: sgd, diag, iso, const')
    parser.add_argument('--noise_mag', type=float, default=None, help='The noise magnitude for the constant noise')

    # Randomness settings
    parser.add_argument('--dataset_seed', type=int, default=888, help='Random seed for dataset preparation')
    parser.add_argument('--init_seed', type=int, default=8888, help='Random seed for network initialization')

    args = parser.parse_args()


    #### deal with all the arguments
    # set the parameters
    batch_size = args.batch
    dataset = args.dataset
    device = (T.device('cuda') if T.cuda.is_available() else 'cpu')
    
    
    if args.steps is not None and args.epochs is not None:
        raise ValueError("You should provide either epochs or steps, not both")
    
    ### set which values to compute ####
    measurements = {name for name, enabled in [
    ('lmax', args.lambdamax),
    ('batch_lmax', args.batch_lambdamax),
    ('batch_sharpness', args.batch_sharpness),
    ('batch_sharpness_static', args.batch_sharpness_static),
    ('gni', args.gni),
    ('fisher', args.fisher),
    ('final', args.final),
    ('param_distance', args.param_distance),
    ] if enabled}

    #### result storage ####
    RES_FOLDER.mkdir(parents=True, exist_ok=True)
    run_folder = initialize_folders(args, RES_FOLDER)

    if args.cont_folder is not None:
        run_folder, step_to_start = run_folder
    else:
        step_to_start = 0

    #### prepare loss #####
    if args.loss == 'mse':
        loss_fn = SquaredLoss()
    elif args.loss == 'ce':
        loss_fn = nn.CrossEntropyLoss()

    ##### Prepare dataset and model ####
    dataset_presets = get_dataset_presets()
    model_presets = get_model_presets()

    ### Prepare dataset ###
    data = prepare_dataset(dataset, DATASET_FOLDER, args.num_data, args.classes, args.dataset_seed, loss_type=args.loss)

    ### Prepare model ###
    name = args.model
    params = model_presets[name]['params']
    params['input_dim'] = dataset_presets[dataset]['input_dim']
    params['output_dim'] = dataset_presets[dataset]['output_dim']
    net = prepare_net(
        model_type=model_presets[name]['type'], 
        params=params
        )

    #### Initialize net #####
    if not args.no_init:
        initialize_net(net, scale=args.init_scale, seed=args.init_seed)

    #### Load the model if continuing ####
    if args.cont_folder is not None:
        cont_folder = Path(RES_FOLDER / args.cont_folder)
        state_file = cont_folder / 'checkpoints' / f'net_{args.cont_epoch}.pt'
        net.load_state_dict(T.load(state_file, map_location=device))

    
    #### Distance from the reference weights ####
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

    ### Prepare optimizer ###
    optimizer = prepare_optimizer(net, args.lr, args.momentum, args.adam)


    # Train
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
        initial_sharpness = args.init_sharp,
        epoch_to_start=args.cont_epoch,
        step_to_start = step_to_start,
        sharpness_every=args.sharp_every,
        gd_noise=args.gd_noise,
        noise_magnitude=args.noise_mag,
        results_rarely=args.results_rarely,
        measurements=measurements,
    )






