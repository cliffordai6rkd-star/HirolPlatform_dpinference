from setuptools import setup, find_packages

setup(
    name="HIROLRobotPlatform",
    version="0.1.0",
    packages=find_packages(),
    author="HIROL",
    author_email="kjustdoitno1@gmail.com",  # 
    description="A simulation and control platform for HIROL robots.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/HIROLRobotPlatform",  # Replace with your project's URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
