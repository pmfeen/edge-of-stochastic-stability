import time
import shutil
from datetime import datetime
from pathlib import Path


def get_welcome_string(args, continue_msg=None):
    """
    Returns the header of the log file.

    Parameters:
    -----------
    args : argparse object with the parameters given to the program
    """

    msg = f"""# Edge of Stochastic Stability. {continue_msg}
# Dataset: {args.dataset}, model {args.model}, lr {args.lr}, batch size {args.batch}, gd_noise {args.gd_noise}
# Arguments: {str(args)}
# (0) epoch, (1) step, (2) batch loss, (3) full loss, (4) batch lambda max, (5) lambda max, (6) batch sharpness (unaveraged=step sharpness), (7) total grad H grad,  (8) batch fisher eigenval, (9) total fisher eigenval, (10) batch sharpness static (averaged), (11) Gradient-Noise Interaction, (12) total accuracy"""
    return msg


def clone_results(origin_folder, dest_folder, epoch):
    origin_res = origin_folder / 'results.txt'
    dest_res = dest_folder / 'results.txt'

    with open(origin_res, 'r') as f, open(dest_res, 'a') as dest_res:
        lines = f.readlines()
        # 000, 00001, 0.993737,     nan,   36.5,    nan,   17.9
        for i, line in enumerate(lines):
            if line.startswith("#"):
                continue
            dest_res.write(line)
            if int(line.split(',')[0]) >= epoch:
                step_to_start = int(line.split(',')[1])
                # >= since we want to start from that epoch
                return step_to_start
        step_to_start = int(line.split(',')[1])
        # >= since we want to start from that epoch
        return step_to_start


def initialize_folders(args, results_folder):
    run_folder_name = f'{args.dataset}_{args.model}'

    def generate_folder_name(args):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M_%S')
        config_name = f'{timestamp}_lr{args.lr:.5f}_b{args.batch}' 
        return config_name
    
    while True:
        config_name = generate_folder_name(args)
        runs_folder = results_folder / run_folder_name / config_name
        if not runs_folder.exists():
            try:
                runs_folder.mkdir(parents=True, exist_ok=False)
            except:
                time.sleep(2)
                continue
            break
        else:
            time.sleep(2)
            continue
    
    

    model_save_path = runs_folder / 'checkpoints'
    model_save_path.mkdir(parents=True, exist_ok=True)


    results_file = runs_folder / 'results.txt'

    if args.cont_last:
        if Path(results_folder / args.cont_folder).exists():
            checkpoint_files = list((results_folder / args.cont_folder / 'checkpoints').glob('net_*.pt'))
            if checkpoint_files:
                epochs = []
                for f in checkpoint_files:
                    epoch = f.stem.split('_')[1]
                    try:
                        epoch = int(epoch)
                    except:
                        continue
                    epochs.append(epoch)
                # epochs = [int(f.stem.split('_')[1]) for f in checkpoint_files]
                latest_epoch = sorted(epochs)[-2] if len(epochs) > 1 else max(epochs)
                args.cont_epoch = latest_epoch
                args.cont_folder = args.cont_folder

    if args.cont_folder is not None:
        continue_msg = f"Continuing from {args.cont_folder} from epoch {args.cont_epoch}"
    else:
        continue_msg = None
    welcome_string = get_welcome_string(args, continue_msg=continue_msg)

    with open(results_file, 'w') as f:
        f.write(welcome_string + "\n")


    
    if args.cont_folder is not None:
        cont_folder = Path(results_folder / args.cont_folder)

        step_to_start = clone_results(cont_folder, runs_folder, args.cont_epoch)
        
        # clone the checkpoint to start from
        state_file = cont_folder / 'checkpoints' / f'net_{args.cont_epoch}.pt'
        dest_state_file = runs_folder / 'checkpoints' / f'net_{args.cont_epoch}.pt'
        shutil.copy(state_file, dest_state_file)

        return runs_folder, step_to_start

    return runs_folder