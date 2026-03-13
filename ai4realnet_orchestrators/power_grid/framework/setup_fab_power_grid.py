#!/usr/bin/env python3
"""
=============================================================================
FAB Power Grid Orchestrator - Complete Environment Setup
=============================================================================

REQUIREMENTS:
- Python 3.10.18 (EXACT version required)
- Ubuntu 22.04/24.04 or similar Linux
- ~15GB disk space for all dependencies
- Internet connection

USAGE:
    python setup_fab_power_grid.py

WHAT THIS INSTALLS:
    1. Core ML: PyTorch 2.8.0, TensorFlow 2.12.1, NumPy, Pandas, SciPy
    2. RL Frameworks: Stable-Baselines3, Ray 2.5.1, Gymnasium
    3. Grid2Op 1.10.5: Power grid simulation environment
    4. LightSim2Grid, PandaPower: Power flow solvers
    5. Attack Framework dependencies: NNI, torch-geometric
    6. FAB Orchestrator: pika (RabbitMQ client)

FRAMEWORK STRUCTURE AFTER SETUP:
    /path/to/power_grid/
    ├── power_grid_test_runner.py    # Main test runner
    ├── framework/                    # Robustness/Resilience framework
    │   ├── attack_models/           # Attacker implementations
    │   ├── evaluation_framework/    # Metrics computation
    │   ├── modified_curriculum_classes/  # Agent loading
    │   ├── perturbation_agents/     # Perturbation logic
    │   └── trained_models/          # Pre-trained attacker models
    └── requirements.txt             # This requirements file

Author: AI4RealNet Project
Version: 1.0.0
Date: December 2024
=============================================================================
"""

import subprocess
import sys
import os
import platform
import shutil
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

REQUIRED_PYTHON_VERSION = "3.10.18"
PIP_OPTIONS = ["--break-system-packages"]  # Required for Ubuntu 24.04+

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")


def print_step(step_num: int, total: int, description: str):
    print(f"\n{Colors.BLUE}[{step_num}/{total}]{Colors.RESET} {Colors.BOLD}{description}{Colors.RESET}")
    print("-" * 60)


def print_success(text: str):
    print(f"{Colors.GREEN}✅ {text}{Colors.RESET}")


def print_error(text: str):
    print(f"{Colors.RED}❌ {text}{Colors.RESET}")


def print_warning(text: str):
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.RESET}")


def print_info(text: str):
    print(f"{Colors.CYAN}ℹ️  {text}{Colors.RESET}")


def run_command(cmd: list, description: str = "", allow_fail: bool = False) -> bool:
    """Run command with real-time output."""
    print(f"📦 {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    exit_code = subprocess.call(cmd)
    
    if exit_code == 0:
        print_success(f"{description}")
        return True
    else:
        if allow_fail:
            print_warning(f"{description} - FAILED (continuing anyway)")
            return True
        else:
            print_error(f"{description} - FAILED (Exit code: {exit_code})")
            return False


def check_python_version() -> bool:
    """Check Python version is exactly 3.10.18."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print_info(f"Python version: {version_str}")
    
    if version_str != REQUIRED_PYTHON_VERSION:
        print_error(f"Python {REQUIRED_PYTHON_VERSION} required, found {version_str}")
        print_info("Install with: pyenv install 3.10.18 && pyenv global 3.10.18")
        return False
    
    print_success(f"Python version OK: {version_str}")
    return True


def check_system_requirements() -> bool:
    """Check system requirements."""
    print_header("Checking System Requirements")
    
    # Check OS
    system = platform.system()
    print_info(f"Operating System: {system}")
    
    if system != "Linux":
        print_warning(f"Designed for Linux. Running on {system} may have issues.")
    
    # Check Python
    if not check_python_version():
        return False
    
    # Check disk space
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (1024**3)
    print_info(f"Free disk space: {free_gb} GB")
    
    if free_gb < 15:
        print_warning(f"Low disk space ({free_gb} GB). Recommend at least 15 GB.")
    
    # Check NVIDIA GPU
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        if result.returncode == 0:
            print_success("NVIDIA GPU detected - CUDA support available")
        else:
            print_info("No NVIDIA GPU detected - using CPU")
    except FileNotFoundError:
        print_info("nvidia-smi not found - using CPU")
    
    return True


def handle_distutils_packages():
    """Handle problematic distutils packages before main installation."""
    print_info("Handling problematic packages (llvmlite, numba)...")
    
    for package in ["llvmlite", "numba"]:
        try:
            __import__(package)
            print_info(f"Force reinstalling {package}...")
            cmd = [sys.executable, "-m", "pip", "install", "--force-reinstall", 
                   "--no-deps", package] + PIP_OPTIONS
            subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except ImportError:
            pass


def main():
    """Main installation routine."""
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║     FAB Power Grid Orchestrator - Environment Setup                  ║
║                                                                      ║
║     AI4RealNet Project - Robustness & Resilience Evaluation          ║
║     Python 3.10.18 Required                                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.RESET}
""")
    
    # Check system requirements
    if not check_system_requirements():
        print_error("System requirements not met. Aborting.")
        sys.exit(1)
    
    # Handle problematic packages
    handle_distutils_packages()
    
    # Define installation steps
    steps = [
        {
            "cmd": [sys.executable, "-m", "pip", "install", 
                    "torch==2.8.0", "torch-geometric==2.6.1", "numpy", "pandas"] + PIP_OPTIONS,
            "desc": "Step 1: PyTorch 2.8.0, torch-geometric, numpy, pandas"
        },
        {
            "cmd": [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"] + PIP_OPTIONS,
            "desc": "Step 2: Installing requirements.txt (Grid2Op, LightSim2Grid, etc.)"
        },
        {
            "cmd": [sys.executable, "-m", "pip", "install", "tensorflow==2.12.1"] + PIP_OPTIONS,
            "desc": "Step 3: TensorFlow 2.12.1"
        },
        {
            "cmd": [sys.executable, "-m", "pip", "install", 
                    "ray==2.5.1", "gymnasium", "stable-baselines3"] + PIP_OPTIONS,
            "desc": "Step 4: Ray 2.5.1, Gymnasium, Stable-Baselines3"
        },
        {
            "cmd": [sys.executable, "-m", "pip", "install", "--upgrade", "typing-extensions"] + PIP_OPTIONS,
            "desc": "Step 5: Upgrade typing-extensions (fixes conflicts)"
        },
        {
            "cmd": [sys.executable, "-m", "pip", "install", "nni==2.10.1"] + PIP_OPTIONS,
            "desc": "Step 6: NNI 2.10.1 (Neural Network Intelligence)"
        },
        {
            "cmd": [sys.executable, "-m", "pip", "install", "pika>=1.3.0"] + PIP_OPTIONS,
            "desc": "Step 7: Pika (RabbitMQ client for FAB)"
        },
    ]
    
    total_steps = len(steps)
    failed_steps = []
    
    for i, step in enumerate(steps, 1):
        print_step(i, total_steps, step['desc'])
        
        if not run_command(step['cmd'], step['desc']):
            failed_steps.append(step['desc'])
            response = input(f"\n{Colors.YELLOW}Continue despite failure? (y/n): {Colors.RESET}").lower().strip()
            if response not in ['y', 'yes']:
                print_error("Installation aborted by user")
                sys.exit(1)
    
    # Verification
    print_header("Verifying Installation")
    
    verify_imports = """
import sys
errors = []

# Core packages
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
except ImportError as e:
    errors.append(f'❌ PyTorch: {e}')

try:
    import tensorflow as tf
    print(f'✅ TensorFlow: {tf.__version__}')
except ImportError as e:
    errors.append(f'❌ TensorFlow: {e}')

try:
    import numpy as np
    print(f'✅ NumPy: {np.__version__}')
except ImportError as e:
    errors.append(f'❌ NumPy: {e}')

try:
    import pandas as pd
    print(f'✅ Pandas: {pd.__version__}')
except ImportError as e:
    errors.append(f'❌ Pandas: {e}')

# RL packages
try:
    import gymnasium as gym
    print(f'✅ Gymnasium: {gym.__version__}')
except ImportError as e:
    errors.append(f'❌ Gymnasium: {e}')

try:
    import stable_baselines3 as sb3
    print(f'✅ Stable-Baselines3: {sb3.__version__}')
except ImportError as e:
    errors.append(f'❌ Stable-Baselines3: {e}')

try:
    import ray
    print(f'✅ Ray: {ray.__version__}')
except ImportError as e:
    errors.append(f'❌ Ray: {e}')

# Grid2Op packages
try:
    import grid2op
    print(f'✅ Grid2Op: {grid2op.__version__}')
except ImportError as e:
    errors.append(f'❌ Grid2Op: {e}')

try:
    import lightsim2grid
    print(f'✅ LightSim2Grid: {lightsim2grid.__version__}')
except ImportError as e:
    errors.append(f'❌ LightSim2Grid: {e}')

try:
    import pandapower as pp
    print(f'✅ PandaPower: {pp.__version__}')
except ImportError as e:
    errors.append(f'❌ PandaPower: {e}')

# Additional packages
try:
    import pika
    print(f'✅ Pika: {pika.__version__}')
except ImportError as e:
    errors.append(f'❌ Pika: {e}')

try:
    import torch_geometric
    print(f'✅ torch-geometric: {torch_geometric.__version__}')
except ImportError as e:
    errors.append(f'❌ torch-geometric: {e}')

# Summary
print()
if errors:
    print('FAILURES:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('🎉 All packages verified successfully!')
"""
    
    subprocess.call([sys.executable, "-c", verify_imports])
    
    # Test Grid2Op environment
    print_header("Testing Grid2Op Environments")
    
    test_grid2op = """
import grid2op
import warnings
warnings.filterwarnings('ignore')

envs = ['l2rpn_case14_sandbox', 'l2rpn_icaps_2021']
for env_name in envs:
    try:
        env = grid2op.make(env_name)
        print(f'✅ {env_name}: OK (obs_dim={env.observation_space.n})')
        env.close()
    except Exception as e:
        print(f'⚠️  {env_name}: {e}')
"""
    subprocess.call([sys.executable, "-c", test_grid2op])
    
    # Summary
    print_header("Installation Summary")
    
    if failed_steps:
        print_warning(f"Completed with {len(failed_steps)} failed step(s):")
        for step in failed_steps:
            print(f"   ❌ {step}")
    else:
        print_success("All installation steps completed successfully!")
    
    # Print deployment instructions
    print_header("Next Steps for Manuel")
    
    print(f"""
{Colors.BOLD}1. FRAMEWORK DEPLOYMENT:{Colors.RESET}
   
   The power_grid folder should have this structure:
   
   power_grid/
   ├── power_grid_test_runner.py     # Main test runner
   ├── orchestrator.py               # FAB orchestrator
   ├── test_local.py                 # Local testing script
   ├── requirements.txt              # Dependencies (this file)
   └── framework/                    # Robustness framework
       ├── attack_models/            # Attacker implementations
       │   ├── SACAttacker.py
       │   ├── PPOAttacker.py
       │   ├── RLPerturbAttacker.py
       │   ├── GEPerturbAttacker.py
       │   ├── RPerturbAttacker.py
       │   ├── LambdaPIRAttacker.py
       │   └── Environment.py
       ├── evaluation_framework/     # Metrics computation
       │   ├── result_getter.py
       │   └── metrics.py
       ├── modified_curriculum_classes/  # Agent loading
       │   ├── baseline.py
       │   ├── my_agent.py
       │   ├── obs_converter.py
       │   └── utilities.py
       ├── perturbation_agents/
       └── trained_models/           # Pre-trained models
           ├── SAC.zip
           ├── PPO.zip
           └── RLPerturbAgent/
               ├── trained_rlpa_0.pth
               └── trained_rlpa_target_net_0.pth

{Colors.BOLD}2. RABBITMQ CONFIGURATION:{Colors.RESET}
   
   Set environment variables:
   
   export FAB_RABBITMQ_HOST=ai4realnet-rabbitmq.flatland.cloud
   export FAB_RABBITMQ_PORT=5672
   export FAB_RABBITMQ_USER=<username>
   export FAB_RABBITMQ_PASS=<password>

{Colors.BOLD}3. LOCAL TESTING:{Colors.RESET}
   
   cd power_grid
   python test_local.py

{Colors.BOLD}4. PRODUCTION DEPLOYMENT:{Colors.RESET}
   
   cd ai4realnet-orchestrators
   python -m ai4realnet_orchestrators.power_grid.orchestrator

{Colors.BOLD}5. ATTACKERS USED:{Colors.RESET}
   
   - SAC_5:     SAC attacker with factor=5 (moderate)
   - SAC_10:    SAC attacker with factor=10 (aggressive)
   - PPO:       PPO-based attacker
   - RLPerturb: RL perturbation agent
   - Random:    Random perturbation (prob=0.6)
   - LambdaPIR: Lambda-PIR hybrid attacker
   
   Note: GEPerturb removed due to TensorFlow model compatibility issues

{Colors.BOLD}6. EVALUATION CONFIGURATION:{Colors.RESET}
   
   In power_grid_test_runner.py:
   - NUM_EPISODES = 30    (recommended for production)
   - ENV_NAME = "l2rpn_case14_sandbox"
   - ATTACKER_TYPES = ["SAC_5", "SAC_10", "PPO", "RLPerturb", "Random", "LambdaPIR"]
""")
    
    print(f"\n{Colors.GREEN}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}Setup Complete!{Colors.RESET}")
    print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")


if __name__ == "__main__":
    main()
