from setuptools import setup, find_packages

setup(
    name="mlp_bc",
    version="0.1.0",
    packages=find_packages(exclude=['test*', 'dependencies*', 'calibration*']),
    author="zyx",
    author_email="blackxunmeng@gmail.com",  # 
    description="MLP BC policy HIROL robots.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/mlp_bc",  # Replace with your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
