"""
Setup script for VersaSound
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "comfyui_extension" / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="versasound",
    version="1.0.0",
    author="VersaSound Team",
    author_email="your.email@example.com",
    description="Universal video-to-audio generation for ComfyUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/VersaSound",
    packages=find_packages(include=["comfyui_extension", "comfyui_extension.*",
                                   "training_system", "training_system.*",
                                   "fine_tuning", "fine_tuning.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "versasound-train=training_system.train:main",
            "versasound-finetune=fine_tuning.finetune:main",
        ],
    },
    include_package_data=True,
    package_data={
        "comfyui_extension": ["configs/*.yaml", "checkpoints/.gitkeep"],
        "training_system": ["configs/*.yaml"],
        "fine_tuning": ["configs/*.yaml"],
    },
    zip_safe=False,
)
