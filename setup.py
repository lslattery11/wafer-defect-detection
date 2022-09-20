from setuptools import find_namespace_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="wafer-defect-detection",
    version=1.0,
    python_requires=">= 3.9",
    packages=find_namespace_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchmetrics",
        "pytorch-lightning",
        "pandas",
        "matplotlib",
        "ipykernel",
        "wandb",
        "scikit-learn",
        "imgaug",
        "pylint",
        "flake8",
        "huggingface_hub",
        "hyperopt"
    ],
    include_package_data=True,
    url="https://github.com/lslattery11/wafer-defect-detection/",
    description=(
        "A package for defect detection in silicon wafers."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
)