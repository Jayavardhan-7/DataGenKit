from setuptools import setup, find_packages
import os

with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="Datagenkit",
    version="0.2.0",
    description="A local Image Dataset Generator using smart augmentations and MobileNetV2 filtering.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jayavardhan",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "albumentations",
        "opencv-python-headless",
        "scikit-learn",
        "Pillow",
        "tqdm",
        "numpy<2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "datagenkit=datagenkit.cli:main",
        ]
    }
)
