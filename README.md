# Download

```
git clone --recursive git@github.com:wavelab/GTA-Cuda-Remapping.git
```

# Build

```
cd GTA-Cuda-Remapping
mkdir build
cd build
cmake ..
make
```

# Run

```
./Remapping -i <input_folder> -n <num_threads> -o <output_path> -m <map_file>
```

This will recursively search in input_folder for anything that has "png" extension.
It will output in `<output_path>` with the same file name and folder structure.

# Map Files

See the `examples` folder for examples of useful map files. Every line in the map file must be formatted as follows:

```
<old_r> <old_g> <old_b> <new_r> <new_b> <new_g>
```

This program will find all values in each image with the `old_{r,g,b}` and replace it with `new_{r,g,b}` respectively. Values that are not found in the map file are set to (0,0,0).

# Requirements
* OpenCV
* Cuda 8+ (nvcc in path)

# Notes
This is very specific for our purpose of remapping our colours (after post processing) to cityscapes, not much is done for extendability.
