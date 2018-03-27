# Build

```
mkdir build
cd build
cmake ..
make
```

# Run

```
./Remapping -i <input_folder> -n <num_threads>
```

This will recursively search in input_folder for anything that has "png" extension and starts with "n" (from post processing).
It will output in the same folder with "r_" instead of "n_" in the file name.

# Requirements
* OpenCV
* Cuda 8+ (nvcc in path)

# Notes
This is very specific for our purpose of remapping our colours (after post processing) to cityscapes, not much is done for extendability.
