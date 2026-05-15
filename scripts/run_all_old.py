#!/usr/bin/env python3
"""
Master script to run all thesis experiments in sequence.
Run this after installing dependencies.

Usage:
    python run_all.py

This script will:
    1. Check if dependencies are installed
    2. Run all scripts in the correct order (01 through 07)
    3. Stop if any script fails
    4. Print progress and summary at the end
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(message):
    """Print a formatted header message."""
    print("\n" + "=" * 70)
    print(f"{BOLD}{BLUE}{message}{RESET}")
    print("=" * 70 + "\n")

def print_success(message):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{RESET}")

def print_error(message):
    """Print an error message."""
    print(f"{RED}✗ {message}{RESET}")

def print_info(message):
    """Print an info message."""
    print(f"{YELLOW}ℹ {message}{RESET}")

def run_script(script_path, script_name):
    """
    Run a single Python script and return success status.
    
    Args:
        script_path: Full path to the script
        script_name: Display name of the script
    
    Returns:
        True if script succeeded, False otherwise
    """
    print_header(f"Running {script_name}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True,
            check=False
        )
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print_success(f"{script_name} completed in {elapsed_time:.2f} seconds")
            return True
        else:
            print_error(f"{script_name} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print_error(f"{script_name} encountered an error: {e}")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print_header("Checking Dependencies")
    
    required_packages = [
        'numpy', 'pandas', 'sklearn', 'matplotlib', 
        'seaborn', 'tensorflow', 'kagglehub', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print_success(f"{package} is installed")
        except ImportError:
            print_error(f"{package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print_info(f"\nMissing packages: {', '.join(missing_packages)}")
        print_info("Run: pip install -r requirements.txt")
        return False
    
    print_success("\nAll dependencies are installed")
    return True

def check_directories():
    """Ensure all required directories exist."""
    print_header("Checking Directories")
    
    required_dirs = ['data', 'models', 'results', 'results/figures', 'results/tables', 'src', 'scripts', 'mitigation', 'notebooks']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print_success(f"{dir_name}/ exists")
        else:
            print_info(f"{dir_name}/ does not exist, creating...")
            os.makedirs(dir_name, exist_ok=True)
            print_success(f"{dir_name}/ created")
    
    return True

def main():
    """Main execution function."""
    print_header("THESIS EXPERIMENT RUNNER")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Change to script directory if needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir:
        os.chdir(script_dir)
        print_info(f"Changed to directory: {os.getcwd()}")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print_error("Please install missing dependencies and try again.")
        sys.exit(1)
    
    # Step 2: Check directories
    check_directories()
    
    # Step 3: List of scripts to run in order
    scripts = [
        ("01_download_dataset.py", "Downloading CIC-DDoS2019 Dataset"),
        ("02_preprocess_data.py", "Preprocessing Data and Feature Selection"),
        ("03_train_baselines.py", "Training Baseline Models (SVM & Random Forest)"),
        ("04_train_cnn_lstm.py", "Training CNN-LSTM Hybrid Model"),
        ("05_evaluate_models.py", "Evaluating All Models"),
        ("06_test_mitigation.py", "Testing Real-Time Mitigation"),
        ("07_generate_chapter2_graphs.py", "Generating Chapter 2 Graphs")
    ]
    
    # Track results
    results = []
    total_start_time = time.time()
    
    # Run each script
    for script_file, script_description in scripts:
        script_path = os.path.join("scripts", script_file)
        
        if not os.path.exists(script_path):
            print_error(f"Script not found: {script_path}")
            results.append((script_description, False, "Script not found"))
            continue
        
        success = run_script(script_path, script_description)
        results.append((script_description, success, ""))
        
        if not success:
            print_error(f"\nStopping execution due to failure in {script_description}")
            break
    
    # Print summary
    total_elapsed = time.time() - total_start_time
    print_header("EXECUTION SUMMARY")
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    for description, success, error in results:
        if success:
            print_success(f"{description}")
        else:
            print_error(f"{description} - FAILED")
    
    print("\n" + "-" * 50)
    print(f"Total time: {total_elapsed:.2f} seconds")
    print(f"Successful: {success_count}/{total_count}")
    
    if success_count == total_count:
        print_success("\nAll scripts completed successfully!")
        print_info("\nYour results are located in:")
        print_info("  - models/        (trained models)")
        print_info("  - results/figures/   (thesis graphs)")
        print_info("  - results/tables/    (performance tables)")
        print_info("  - data/              (dataset sample)")
    else:
        print_error("\nSome scripts failed. Please check the errors above.")
    
    print_header("THESIS RUNNER COMPLETED")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print(f"{YELLOW}Execution interrupted by user{RESET}")
        print("=" * 70)
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)