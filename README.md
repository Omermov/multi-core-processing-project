# multi-core-processing-project

_There are 3 main folders in this repository:_

## 1. reference

This is the original implementation from the repository GMAP/NPB-CPP
All of our code is based on this implementation

## 2. ours_CPU

This is our CPU implementation that includes various improvements (detailed in the report)

Use the notebook `ours_CPU/cpu_notebook.ipynb` to test it.

## 3. ours_CPU

This is our GPU implementation which is a rewrite of the logic to be suitable for GPU offloading.
This folder includes 4 different implementations: `single_gpu`, `single_gpu_alloc`, `single_gpu_less_mem` and `multiple_gpu`.

Use the notebook `ours_GPU/gpu_notebook.ipynb` to test it.
