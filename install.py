import os
import sys
import subprocess
import platform
from typing import List, Tuple
import urllib.request
import json

def run_command(command: List[str], description: str = None) -> Tuple[int, str]:
    """Run a command and return its exit code and output"""
    if description:
        print(f"\n{description}...")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    output = []
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            output.append(line.strip())
    
    return process.returncode, '\n'.join(output)

def check_python_version():
    """Check if Python version is 3.10"""
    if sys.version_info[:2] != (3, 10):
        print("Error: Python 3.10 is required")
        sys.exit(1)

def check_cuda():
    """Check CUDA availability and version"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. GPU acceleration will not be possible.")
            return False
        
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        return True
    except ImportError:
        print("PyTorch is not installed yet.")
        return False

def check_cuda():
    """Check CUDA availability and version"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. GPU acceleration will not be possible.")
            return False
        
        cuda_version = torch.version.cuda
        torch_version = torch.__version__  # Get the PyTorch version
        print(f"CUDA version: {cuda_version}")
        print(f"PyTorch version: {torch_version}")  # Print the PyTorch version
        return True
    except ImportError:
        print("PyTorch is not installed yet.")
        return False
    
def install_cuda():
    """Install CUDA if not present"""
    
    system = platform.system()
    if system == "Windows":
        print("Please install CUDA manually from: https://developer.nvidia.com/cuda-downloads")
        input("Press Enter once CUDA is installed to continue...")
    elif system == "Linux":
        # Add NVIDIA package repositories
        wget_command = ["wget", "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"]
        run_command(wget_command, "Downloading CUDA repository configuration")
        
        mv_command = ["sudo", "mv", "cuda-ubuntu2004.pin", "/etc/apt/preferences.d/cuda-repository-pin-600"]
        run_command(mv_command)
        
        # Install CUDA
        run_command(["sudo", "apt-get", "update"], "Updating package list")
        run_command(["sudo", "apt-get", "install", "-y", "cuda"], "Installing CUDA")
    else:
        print(f"Unsupported operating system: {system}")
        sys.exit(1)

def install_pytorch():
    """Install PyTorch with CUDA support"""
    cuda_available = check_cuda()
    if cuda_available:
        run_command([
            "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], "Installing PyTorch with CUDA support")
    else:
        run_command([
            "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ], "Installing PyTorch (CPU only)")

def install_requirements():
    """Install project requirements"""
    run_command(["pip", "install", "-r", "requirements.txt"], "Installing project requirements")

def install_project():
    """Install the project in development mode"""
    run_command(["pip", "install", "-e", "."], "Installing project in development mode")

def main():
    print("Starting installation process...")
    
    check_cuda()
    # Check Python version
    check_python_version()
    
    # Check and install CUDA if needed
    if not check_cuda():
        install_cuda()
    
    # Install PyTorch
    install_pytorch()
    
    # Install project requirements
    install_requirements()
    
    # Install project
    install_project()
    
    print("\nInstallation completed successfully!")
    print("\nTo verify the installation, try running:")
    print("python -c 'import torch; print(torch.cuda.is_available())'")

if __name__ == "__main__":
    main() 