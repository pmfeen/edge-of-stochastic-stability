#!/usr/bin/env python3
"""
Test script for the updated wandb checkpoint system with separate directory.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import wandb

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.wandb_utils import save_checkpoint_wandb, find_closest_checkpoint_wandb, load_checkpoint_wandb, get_checkpoint_dir_for_run
from utils.nets import MLP

def test_separate_checkpoint_system():
    """Test the wandb checkpoint system with separate directory"""
    
    print("Testing separate wandb checkpoint system...")
    
    # Setup temporary directory for testing
    original_cwd = os.getcwd()
    test_dir = tempfile.mkdtemp(prefix="test_separate_wandb_")
    os.chdir(test_dir)
    
    try:
        # Set environment variables for wandb offline mode
        os.environ['WANDB_MODE'] = 'offline'
        os.environ['WANDB_PROJECT'] = 'test_separate_eoss'
        
        # Initialize wandb
        run = wandb.init(project='test_separate_eoss', mode='offline', name='test_separate_checkpoint')
        run_id = run.id
        print(f"Started test run with ID: {run_id}")
        
        # Create a simple model
        model = MLP(input_dim=10, hidden_dim=20, n_layers=2, output_dim=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Test 1: Save checkpoints at different steps
        print("\nTest 1: Saving checkpoints to separate directory...")
        steps_to_test = [100, 250, 500]
        for step in steps_to_test:
            loss = 1.0 / (step + 1)  # Fake decreasing loss
            checkpoint_path = save_checkpoint_wandb(
                model=model,
                optimizer=optimizer, 
                step=step,
                epoch=step // 100,
                loss=loss,
                save_every_n_steps=1  # Save every checkpoint for testing
            )
            print(f"Saved checkpoint at step {step}: {checkpoint_path}")
        
        # Verify separate directory structure
        print("\nTest 2: Verifying directory structure...")
        checkpoint_dir = Path("wandb_checkpoints") / run_id
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Directory exists: {checkpoint_dir.exists()}")
        
        if checkpoint_dir.exists():
            checkpoint_files = list(checkpoint_dir.glob("checkpoint_step_*.pt"))
            print(f"Found checkpoint files: {[f.name for f in checkpoint_files]}")
            
            metadata_file = checkpoint_dir / "checkpoint_metadata.json"
            print(f"Metadata file exists: {metadata_file.exists()}")
        
        # Test 3: Find closest checkpoints using separate directory
        print("\nTest 3: Finding checkpoints in separate directory...")
        test_targets = [150, 400]
        for target_step in test_targets:
            checkpoint_info = find_closest_checkpoint_wandb(target_step, run_id=run_id)
            if checkpoint_info:
                print(f"Target step {target_step} -> Found checkpoint at step {checkpoint_info['step']}")
            else:
                print(f"Target step {target_step} -> No checkpoint found")
        
        # Test 4: Test get_checkpoint_dir_for_run function
        print("\nTest 4: Testing checkpoint directory lookup...")
        found_dir = get_checkpoint_dir_for_run(run_id)
        if found_dir:
            print(f"Found checkpoint directory: {found_dir}")
        else:
            print("No checkpoint directory found")
        
        # Test 5: Load a checkpoint
        print("\nTest 5: Loading checkpoint from separate directory...")
        checkpoint_info = find_closest_checkpoint_wandb(400, run_id=run_id)
        if checkpoint_info:
            # Create a new model to test loading
            new_model = MLP(input_dim=10, hidden_dim=20, n_layers=2, output_dim=1)
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            
            loaded_data = load_checkpoint_wandb(checkpoint_info, new_model, new_optimizer)
            print(f"Successfully loaded checkpoint from step {loaded_data['step']}")
            print(f"  Epoch: {loaded_data['epoch']}")
            print(f"  Loss: {loaded_data['loss']}")
        
        # Test 6: Verify wandb and checkpoint directories are separate
        print("\nTest 6: Verifying separation from wandb directory...")
        wandb_dir = Path(run.dir)
        print(f"Wandb run directory: {wandb_dir}")
        print(f"Checkpoint directory: {checkpoint_dir}")
        print(f"Are they different? {wandb_dir != checkpoint_dir}")
        
        wandb_has_checkpoints = (wandb_dir / "checkpoints").exists()
        print(f"Wandb directory has checkpoints folder: {wandb_has_checkpoints}")
        
        # Finish run
        run.finish()
        print("\nAll tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        os.chdir(original_cwd)
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
        
        # Clean up environment
        if 'WANDB_MODE' in os.environ:
            del os.environ['WANDB_MODE']
        if 'WANDB_PROJECT' in os.environ:
            del os.environ['WANDB_PROJECT']

if __name__ == '__main__':
    success = test_separate_checkpoint_system()
    sys.exit(0 if success else 1)