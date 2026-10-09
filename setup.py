from setuptools import setup, find_packages


setup(
    name="unid",
    version="0.1",
    packages=find_packages(where="Unid"),
    package_dir={"": "Unid"},
    install_requires=[
        "accelerate==0.27.2",
        "diffusers==0.30.2",
        "numpy==1.26.4",
        "tokenizers==0.15.2",
        "transformers==4.37.2",
        "xformers==0.0.24",
        "Pillow",
        "safetensors",
        "tqdm",
        "matplotlib",
    ],
    extras_require={
        "crf": ["pydensecrf"],
        "research": [
            "h5py",
            "opencv-python",
            "scipy",
            "omegaconf",
            "tabulate",
            "pandas",
            "pyarrow",
            "wandb",
            "datasets",
            "peft<0.14",
            "tensorboard",
            "trimesh",
            "scikit-image",
            "moviepy",
            "pycocotools",
        ],
    },
    python_requires=">=3.8",
)
