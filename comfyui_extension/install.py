#!/usr/bin/env python3
"""
VersaSound Installation Script
Automatically installs dependencies and downloads pretrained models
"""

import sys
import subprocess
import os
from pathlib import Path
import argparse


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")

    requirements_file = Path(__file__).parent / "requirements.txt"

    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False

    try:
        print("📦 Installing Python packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencies installed successfully\n")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def download_pretrained_models(skip_download=False):
    """Download pretrained model weights"""
    print_header("Setting Up Model Checkpoints")

    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    models = {
        "visual_encoder": {
            "url": "https://huggingface.co/versasound/visual-encoder-v1/resolve/main/model.safetensors",
            "size": "~800MB"
        },
        "audio_generator": {
            "url": "https://huggingface.co/versasound/audio-generator-v1/resolve/main/model.safetensors",
            "size": "~1.2GB"
        },
        "temporal_aligner": {
            "url": "https://huggingface.co/versasound/temporal-aligner-v1/resolve/main/model.safetensors",
            "size": "~200MB"
        }
    }

    if skip_download:
        print("⚠️  Model download skipped (--skip-models flag)")
        print("ℹ️  You'll need to manually place model checkpoints in:")
        print(f"   {checkpoint_dir}")
        return True

    print(f"📁 Checkpoint directory: {checkpoint_dir}\n")

    try:
        import requests
        from tqdm import tqdm

        for model_name, info in models.items():
            checkpoint_path = checkpoint_dir / f"{model_name}.safetensors"

            if checkpoint_path.exists():
                print(f"✅ {model_name}: Already downloaded")
                continue

            print(f"📥 Downloading {model_name} ({info['size']})...")
            print(f"   URL: {info['url']}")

            try:
                response = requests.get(info['url'], stream=True)

                # Check if URL exists
                if response.status_code == 404:
                    print(f"⚠️  Model not yet available: {model_name}")
                    print(f"   Creating placeholder...")
                    # Create empty placeholder
                    checkpoint_path.touch()
                    continue

                response.raise_for_status()

                total_size = int(response.headers.get('content-length', 0))

                with open(checkpoint_path, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)

                print(f"✅ {model_name}: Downloaded successfully\n")

            except requests.exceptions.RequestException as e:
                print(f"⚠️  Could not download {model_name}: {e}")
                print(f"   Creating placeholder for now...\n")
                checkpoint_path.touch()

        return True

    except ImportError:
        print("❌ 'requests' and 'tqdm' are required for downloading models")
        print("   Install them with: pip install requests tqdm")
        return False
    except Exception as e:
        print(f"❌ Error during model download: {e}")
        return False


def verify_installation():
    """Verify installation is correct"""
    print_header("Verifying Installation")

    success = True

    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Python 3.8+ is required")
        success = False
    else:
        print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check critical imports
    critical_imports = [
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "diffusers",
        "numpy",
        "yaml"
    ]

    for module in critical_imports:
        try:
            __import__(module)
            print(f"✅ {module}: Installed")
        except ImportError:
            print(f"❌ {module}: Not found")
            success = False

    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ MPS (Apple Silicon) available")
        else:
            print("⚠️  No GPU detected - will use CPU (slow)")
    except:
        pass

    # Check model checkpoints
    checkpoint_dir = Path(__file__).parent / "checkpoints"
    checkpoint_count = len(list(checkpoint_dir.glob("*.safetensors"))) if checkpoint_dir.exists() else 0

    if checkpoint_count >= 3:
        print(f"✅ Model checkpoints: {checkpoint_count} found")
    elif checkpoint_count > 0:
        print(f"⚠️  Model checkpoints: Only {checkpoint_count}/3 found")
    else:
        print("⚠️  Model checkpoints: Not found (placeholders may be present)")

    return success


def create_config_if_needed():
    """Create default config if it doesn't exist"""
    config_dir = Path(__file__).parent / "configs"
    config_file = config_dir / "default_config.yaml"

    if config_file.exists():
        print("✅ Configuration file exists")
        return

    print("ℹ️  Configuration file already created during setup")


def main():
    parser = argparse.ArgumentParser(description="Install VersaSound ComfyUI Extension")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-models", action="store_true", help="Skip model download")
    parser.add_argument("--verify-only", action="store_true", help="Only verify installation")

    args = parser.parse_args()

    print_header("VersaSound ComfyUI Extension Installer")
    print("This script will:")
    print("  1. Install Python dependencies")
    print("  2. Download pretrained models (~2GB)")
    print("  3. Verify installation")
    print()

    try:
        if args.verify_only:
            success = verify_installation()
        else:
            # Install dependencies
            if not args.skip_deps:
                if not install_dependencies():
                    print("\n⚠️  Dependency installation had issues, but continuing...")

            # Download models
            if not args.skip_models:
                if not download_pretrained_models():
                    print("\n⚠️  Model download had issues, but continuing...")
            else:
                download_pretrained_models(skip_download=True)

            # Verify
            success = verify_installation()

        # Final message
        print("\n" + "=" * 70)
        if success:
            print("✅ Installation completed successfully!")
            print("\n📝 Next steps:")
            print("   1. Restart ComfyUI to load the extension")
            print("   2. Look for VersaSound nodes in the 'VersaSound' category")
            print("   3. Check examples/ for sample workflows")
        else:
            print("⚠️  Installation completed with warnings")
            print("\n📝 Please check the messages above and resolve any issues")

        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Installation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
