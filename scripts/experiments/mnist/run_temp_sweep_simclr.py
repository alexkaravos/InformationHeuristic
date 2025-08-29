#!/usr/bin/env python3

"""
Temperature sweep script for SimCLR training
This script runs train_simclr.py with different temperature values
"""

import os
import subprocess
import sys
from pathlib import Path


def run_temperature_sweep():
    """Run SimCLR training with different temperature values."""
    
    # List of temperatures to test
    #temperatures = [0.2 , 0.23, 0.26, 0.3 , 0.35, 0.4 , 0.46, 0.53, 0.61, 0.7]
    temperatures = [0.81,0.93]

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    train_script = script_dir / "train_simclr.py"
    
    print("Starting temperature sweep for SimCLR training")
    print(f"Temperatures to test: {temperatures}")
    print(f"Script location: {script_dir}")
    print(f"Training script: {train_script}")
    print("=" * 60)
    
    # Track results
    successful_runs = []
    failed_runs = []
    
    # Loop through each temperature
    for temp in temperatures:
        print(f"\nStarting training with temperature: {temp}")
        print("=" * 60)
        
        try:
            # Run the training script with the current temperature
            # Using Hydra's override syntax to change the temperature
            result = subprocess.run([
                sys.executable, 
                str(train_script), 
                f"representations.temperature={temp}"
            ], check=True, capture_output=False)
            
            print(f"Training completed successfully for temperature: {temp}")
            successful_runs.append(temp)
            
        except subprocess.CalledProcessError as e:
            print(f"Training failed for temperature: {temp}")
            print(f"Error: {e}")
            print("Continuing with next temperature...")
            failed_runs.append(temp)
        
        except KeyboardInterrupt:
            print(f"\nInterrupted during temperature: {temp}")
            print("Stopping temperature sweep...")
            break
            
        print("=" * 60)
    
    # Print summary
    print(f"\nTemperature sweep completed!")
    print(f"Successful runs ({len(successful_runs)}): {successful_runs}")
    if failed_runs:
        print(f"Failed runs ({len(failed_runs)}): {failed_runs}")
    print("Results saved in: weights/mnist/representations/simclr/")
    print("Each run saved in subdirectory: simclr_resnet18_<temperature>")


if __name__ == "__main__":
    run_temperature_sweep()
