from setuptools import setup, find_packages

setup(
    name="torch-continuum",
    version="0.2.0",
    description="Accelerate any PyTorch workload in one line.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="torch-continuum",
    url="https://github.com/torch-continuum/torch-continuum",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "liger": ["liger-kernel>=0.5.0"],
        "all": ["liger-kernel>=0.5.0", "triton"],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
