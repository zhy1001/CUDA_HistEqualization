
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define TIMER_CREATE(t)               \
  cudaEvent_t t##_start, t##_end;     \
  cudaEventCreate(&t##_start);        \
  cudaEventCreate(&t##_end);               
 
 
#define TIMER_START(t)                \
  cudaEventRecord(t##_start);         \
  cudaEventSynchronize(t##_start);    \
 
 
#define TIMER_END(t)                             \
  cudaEventRecord(t##_end);                      \
  cudaEventSynchronize(t##_end);                 \
  cudaEventElapsedTime(&t, t##_start, t##_end);  \
  cudaEventDestroy(t##_start);                   \
  cudaEventDestroy(t##_end);     
  
#define TILE_SIZE 16
#define BLOCK_SIZE_X 1024
#define BLOCK_SIZE_Y 1
#define BLOCK_SIZE_X2 1024
#define BLOCK_SIZE_Y2 1
#define CUDA_TIMING

unsigned char *input_gpu;
unsigned char *output_gpu;
unsigned int *hist;
unsigned char *lut;

double CLOCK() {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC,  &t);
	return (t.tv_sec * 1000)+(t.tv_nsec*1e-6);
}

/*******************************************************/
/*                 Cuda Error Function                 */
/*******************************************************/
inline cudaError_t checkCuda(cudaError_t result) {
	#if defined(DEBUG) || defined(_DEBUG)
		if (result != cudaSuccess) {
			fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
			exit(-1);
		}
	#endif
		return result;
}
                
// Add GPU kernel and functions
__global__ void kernel(unsigned char *input, 
			unsigned int imgSize,
                       unsigned char *output){
        
  	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
                
        int location = 	y*blockDim.x*gridDim.x+x;
	
        if (location<imgSize) output[location] = x%255;

}

__global__ void genHist(unsigned int *input,
			unsigned int width,
			unsigned int height,
			unsigned int *hist) {

	int x = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ unsigned int tempHist[256];
	tempHist[threadIdx.x]=0;
	__syncthreads();

	const unsigned int temp=input[x];

	//Calculate Histogram
	atomicAdd(&tempHist[(temp & 0x000000FF)], 1);
	atomicAdd(&tempHist[(temp & 0x0000FF00) >> 8], 1);
	atomicAdd(&tempHist[(temp & 0x00FF0000) >> 16], 1);
	atomicAdd(&tempHist[(temp & 0xFF000000) >> 24], 1);
	__syncthreads();
	
	atomicAdd(&(hist[threadIdx.x]),tempHist[threadIdx.x]);
	
}

__global__ void genHist2(unsigned char *input,
			int numPixel,
			unsigned int *hist){
        
  	int x = blockIdx.x*blockDim.x+threadIdx.x;
                
	//Generate new gray value
	if (x<numPixel){
		atomicAdd(&hist[input[x]], 1);
	}
}

__global__ void genLUT(	unsigned int *hist,
			float imgSize,
			unsigned char *lut){
        
  	int x = blockIdx.x*blockDim.x+threadIdx.x;
	int y = blockIdx.y*blockDim.y+threadIdx.y;
                
        int location = 	y*blockDim.x*gridDim.x+x;

	__shared__ unsigned int cdfHist[256];
	__shared__ unsigned int tempHist[256];
	__shared__ int mincdf;

	tempHist[location]=hist[location];
	__syncthreads();

	//Accumulate
	unsigned int cdfTemp=0;	
	int i = location;
	do {
		cdfTemp += tempHist[i--];
	} while (i >= 0);
	cdfHist[location]=cdfTemp;
	__syncthreads();

	//Find minimum CDF
	if (threadIdx.x==0&&threadIdx.y==0) {
		int j=0;
		while (j<256 && cdfHist[j]==0) {
			++j;		
		}
		mincdf=j;
	}
	__syncthreads();

	//Generate look-up table
	float lutf=0;
	if (location>mincdf) {
		lutf=255.0*(cdfHist[location]-cdfHist[mincdf])/(imgSize-cdfHist[mincdf]);
	}
	//Write look-up table
	lut[location]=(unsigned char)roundf(lutf);
}

__global__ void applyLUT(unsigned int *input,
			unsigned int width,
			unsigned char *lut,
			unsigned int *output){
        
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	__shared__ unsigned char lutTemp[256];
	lutTemp[threadIdx.x]=lut[threadIdx.x];
	__syncthreads();
	
	unsigned int temp=input[x];
	unsigned char temp1=lutTemp[(temp & 0xFF000000) >> 24];
	unsigned char temp2=lutTemp[(temp & 0x00FF0000) >> 16];
	unsigned char temp3=lutTemp[(temp & 0x0000FF00) >> 8];
	unsigned char temp4=lutTemp[(temp & 0x000000FF)];
	
	temp=(((unsigned int)temp1) << 24)+(((unsigned int)temp2) << 16)+(((unsigned int)temp3) << 8)+((unsigned int)temp4);

	output[x]=temp;
}

__global__ void applyLUT2(unsigned char *input,
			int numPixel,
			unsigned char *lut,
			unsigned char *output){
        
  	int x = blockIdx.x*blockDim.x+threadIdx.x;
                
	//Generate new gray value
	if (x<numPixel){
		output[x]=lut[input[x]];
	}
}

void histogram_gpu(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){

	int gridXSize = width*height / BLOCK_SIZE_X;
	int gridYSize = 1;
                         
	int gridXSize2 = width*height / BLOCK_SIZE_X2;
	int gridYSize2 = 1;

	int restPixel = width*height % BLOCK_SIZE_X2;
	int lutOffset = gridXSize2 * BLOCK_SIZE_X2;
	
	// Both are the same size (CPU/GPU).
	unsigned int size = height*width;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&hist  , 256*sizeof(unsigned int)));
	checkCuda(cudaMalloc((void**)&lut  , 256*sizeof(unsigned char)));
	
        checkCuda(cudaMemset(hist , 0 , 256*sizeof(unsigned int)));
        checkCuda(cudaMemset(lut , 0 , 256*sizeof(unsigned char)));
	checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
				
        // Copy data to GPU
        checkCuda(cudaMemcpy(input_gpu, 
			data, 
			size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
        // Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(BLOCK_SIZE_X/4, BLOCK_SIZE_Y);
	dim3 dimGridforLUT(1, 1);
        dim3 dimBlockforLUT(16, 16);
	dim3 dimGrid2(gridXSize2, gridYSize2);
	dim3 dimBlock2(BLOCK_SIZE_X2/4, BLOCK_SIZE_Y2);

        // Kernel Call
	#if defined(CUDA_TIMING)
		float Ktime;
		TIMER_CREATE(Ktime);
		TIMER_START(Ktime);
	#endif
        
        genHist<<<dimGrid, dimBlock>>>((unsigned int*)input_gpu, 
					width,
					height,
					hist);
        if (restPixel != 0){
		
		int gridXSize3 = (restPixel-1) / (BLOCK_SIZE_X2/4) + 1;
		int gridYSize3 = 1;
		dim3 dimGrid3(gridXSize3, gridYSize3);

		genHist2<<<dimGrid3, dimBlock>>>(input_gpu+lutOffset, 
						restPixel,
						hist);
			
	}

        genLUT<<<dimGridforLUT, dimBlockforLUT>>>(hist, 
                                      		size,
						lut);
        applyLUT<<<dimGrid2, dimBlock2>>>((unsigned int*)input_gpu, 
					width,
					lut,
					(unsigned int*)output_gpu);

        if (restPixel != 0){
		
		int gridXSize3 = (restPixel-1) / (BLOCK_SIZE_X2/4) + 1;
		int gridYSize3 = 1;
		dim3 dimGrid3(gridXSize3, gridYSize3);

		applyLUT2<<<dimGrid3, dimBlock2>>>(input_gpu+lutOffset, 
						restPixel,
						lut,
						output_gpu+lutOffset);
			
	}
                                             
        checkCuda(cudaDeviceSynchronize());
	
	#if defined(CUDA_TIMING)
		TIMER_END(Ktime);
		printf("Kernel Execution Time: %f ms\n", Ktime);
	#endif
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
                        
        // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));
	checkCuda(cudaFree(hist));
	checkCuda(cudaFree(lut));

}

void histogram_gpu_warmup(unsigned char *data, 
                   unsigned int height, 
                   unsigned int width){
                         
	int gridXSize = 1 + (( width - 1) / TILE_SIZE);
	int gridYSize = 1 + ((height - 1) / TILE_SIZE);
	
	// Both are the same size (CPU/GPU).
	unsigned int size = height*width;
	
	// Allocate arrays in GPU memory
	checkCuda(cudaMalloc((void**)&input_gpu   , size*sizeof(unsigned char)));
	checkCuda(cudaMalloc((void**)&output_gpu  , size*sizeof(unsigned char)));
	
        checkCuda(cudaMemset(output_gpu , 0 , size*sizeof(unsigned char)));
				
        // Copy data to GPU
        checkCuda(cudaMemcpy(input_gpu, 
			data, 
			size*sizeof(char), 
			cudaMemcpyHostToDevice));

	checkCuda(cudaDeviceSynchronize());
        
        // Execute algorithm
        
	dim3 dimGrid(gridXSize, gridYSize);
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        
        kernel<<<dimGrid, dimBlock>>>(input_gpu, 
					size,
                                      output_gpu);
                                             
        checkCuda(cudaDeviceSynchronize());
        
	// Retrieve results from the GPU
	checkCuda(cudaMemcpy(data, 
			output_gpu, 
			size*sizeof(unsigned char), 
			cudaMemcpyDeviceToHost));
                        
        // Free resources and end the program
	checkCuda(cudaFree(output_gpu));
	checkCuda(cudaFree(input_gpu));

}

