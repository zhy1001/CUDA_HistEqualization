# CUDA_HistEqualization
Optimized for 8-bit B&amp;W pictures on NVIDIA GPU with Kepler and previous architecture

By using the bitwise operation, every thread processes 8 pixels at the same time so it could reach the maximum bandwidth and reduce the number of uses of the function "atomicAdd"

For newer GPU architecture such as Maxwell and Pascal, use shared memory for better performance

Change the location of "cuda_runtime_api" and "opencv library" if needed

Reference:
https://devblogs.nvidia.com/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
