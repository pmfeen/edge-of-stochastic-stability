# testing SDE
python training.py --dataset cifar10 --model mlp --sde --sde_h 0.001 --sde_eta 0.01 --sde_seed 111 --batch 4 --momentum 0 --lr 0.01 --loss mse  --stop_loss 0.00001 --steps 10000  --num_data 16 --init_scale 0.2 --dataset_seed 1011 --init_seed 1312


# testing noisy GD
python training.py --dataset cifar10 --model mlp --batch 16  --stop_loss 0.00001 --steps 100000 --lr 0.015 --num_data 64  --init_scale 0.3  --gd_noise diag
