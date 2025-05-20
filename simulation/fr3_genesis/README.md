In tty1, start genesis sim server
```
run-franka-sim-server -v
```

In tty2,
```
conda create -n fr3-genesis-py12 python=3.12 # python版本与系统python版本一致，从而保证franka_bindings.cpython-312-x86_64-linux-gnu.so的兼容性,3.12为经过测试的版本。

git clone https://github.com/BarisYazici/libfranka-python.git

# Navigate to the franka_bindings directory
cd libfranka-python

# Create a build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build the bindings
make -j4

# Return to the franka_bindings directory
cd ..

# Install the Python package
pip install -e .

cd ..

```


Run tests like
```
python test.py
```
