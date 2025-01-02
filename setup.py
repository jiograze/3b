from setuptools import setup, find_packages

setup(
    name="otuken3d",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pillow>=8.3.1",
        "trimesh>=3.9.29",
        "matplotlib>=3.4.3",
        "wandb>=0.12.0",
        "tqdm>=4.62.2",
        "transformers>=4.11.3",
        "fvcore",
        "iopath",
        "torch>=2.0.0",
        "torchvision>=0.15.2",
        "pytorch3d-cpu"
    ],
    python_requires=">=3.8",
    package_data={
        "otuken3d": ["*.yaml", "*.json"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "otuken3d-train=modules.training.train:main",
            "otuken3d-demo=modules.inference.demo:main",
        ],
    },
) 