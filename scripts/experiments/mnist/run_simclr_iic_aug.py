#!/usr/bin/env python3

"""
Lambda sweep script for Barlow Twins training
This script runs train_barlowtwin.py with different lambda values
"""

import os
import subprocess
import sys
from pathlib import Path


def run_simclr():
    """Run Barlow Twins training with different lambda values."""
    
    # List of temperature values for contrastive learning
    #temperatures = [0.2 , 0.23, 0.26, 0.3 , 0.35, 0.4 , 0.46, 0.53, 0.61, 0.7,
    temperatures = [0.81,0.93]
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_iic_aug.py"
    
    print("Starting temperature sweep for SimCLR Representations with IIC clustering")
    print(f"Temperature values to test: {temperatures}")
    print(f"Script location: {script_dir}")
    print(f"Training script: {train_script}")
    print("=" * 60)
    
    # Track results
    successful_runs = []
    failed_runs = []
    
    # Loop through each lambda value
    for temp in temperatures:
        
        temp = f"{temp*1000:.2f}"
        representation_model_id = f'simclr_r18_{temp}'
        representation_model_path = f'weights/mnist/representations/simclr/simclr_resnet18_{temp}'
        
        
        try:
            # Run the training script with the current lambda
            # Using Hydra's override syntax to change the lambda_param
            result = subprocess.run([
                sys.executable, 
                str(train_script), 
                f"representation_model_id={representation_model_id}",
                f"representation_model_dir={representation_model_path}",
                f'clustering.num_aug_copies=5',
            ], check=True, capture_output=False)
            
            print(f"Training completed successfully for lambda: {temp}")
            successful_runs.append(temp)
            
        except subprocess.CalledProcessError as e:
            print(f"Training failed for lambda: {temp}")
            print(f"Error: {e}")
            print("Continuing with next lambda...")
            failed_runs.append(temp)
        
        except KeyboardInterrupt:
            print(f"\nInterrupted during lambda: {temp}")
            print("Stopping lambda sweep...")
            break
            
        print("=" * 60)
    
    # Print summary
    print(f"\nLambda sweep completed!")
    print(f"Successful runs ({len(successful_runs)}): {successful_runs}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}): {failed_runs}")
    print("Results saved in: weights/mnist/representations/barlowtwin/")
    print("Each run saved in subdirectory: barlowtwin_resnet18_<lambda*1000>")


if __name__ == "__main__":
    run_simclr()
