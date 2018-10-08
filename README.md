# Build

```
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

# Requirements
* OpenCV
* Cuda 8+ (nvcc in path)

# Notes
This is very specific for our purpose of remapping our colours (after post processing) to cityscapes, not much is done for extendability.
