#!/usr/bin/env python3

"""
Lambda sweep script for Barlow Twins training on CIFAR-10
This script runs train_barlowtwin.py with different lambda values
"""

import os
import subprocess
import sys
from pathlib import Path


def run_lambda_sweep():
    """Run Barlow Twins training with different lambda values."""
    
    # List of lambda values to test
   # List of lambda values to test, divided by 1000 from your input
    lambda_values = [0.0024, 0.0029, 0.0035, 0.0042, 0.005 , 0.006 , 0.0072, 0.0086,
       0.0104]
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_barlowtwin.py"
    
    print("Starting lambda sweep for Barlow Twins training on CIFAR-10")
    print(f"Lambda values to test: {lambda_values}")
    print(f"Script location: {script_dir}")
    print(f"Training script: {train_script}")
    print("=" * 60)
    
    # Track results
    successful_runs = []
    failed_runs = []
    
    # Loop through each lambda value
    for lmbda in lambda_values:
        print(f"\nStarting training with lambda: {lmbda}")
        print("=" * 60)
        
        try:
            # Run the training script with the current lambda
            # Using Hydra's override syntax to change the lambda_param
            result = subprocess.run([
                sys.executable,
                str(train_script),
                f"representations.lambda_param={lmbda}"
            ], check=True, capture_output=False)
            
            print(f"Training completed successfully for lambda: {lmbda}")
            successful_runs.append(lmbda)
            
        except subprocess.CalledProcessError as e:
            print(f"Training failed for lambda: {lmbda}")
            print(f"Error: {e}")
            print("Continuing with next lambda...")
            failed_runs.append(lmbda)
        
        except KeyboardInterrupt:
            print(f"\nInterrupted during lambda: {lmbda}")
            print("Stopping lambda sweep...")
            break
            
        print("=" * 60)
    
    # Print summary
    print(f"\nLambda sweep completed!")
    print(f"Successful runs ({len(successful_runs)}): {successful_runs}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}): {failed_runs}")
    print("Results saved in: weights/cifar10/representations/barlowtwin/")
    print("Each run saved in subdirectory: barlowtwin_resnet18_<lambda*1000>")


if __name__ == "__main__":
    run_lambda_sweep()
