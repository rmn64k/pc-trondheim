// This file is to work around potential difficulties with
// separate compilation for cuda code.

#include "common.h"
#include "device.h"

struct cudaDeviceProp devprop;

struct thread_layout {
	dim3 nblocks;
	dim3 nthreads;
} xy_tile, x_wide;

static struct device_init {
	device_init() {
		check_cuda(cudaGetDeviceProperties(&devprop, 0));

		int maxtpb = 128;//devprop.maxThreadsPerBlock;

		xy_tile.nblocks  = dim3(NX/NX_TILE, NY/NY_TILE);
		xy_tile.nthreads = dim3(NX_TILE, NY_TILE);

		int nx = NX, ny = NY, xblocks = 1, yblocks;
		while (nx > maxtpb) {
			do {
				xblocks++;
			} while ((NX/xblocks)*xblocks != NX);
			nx = NX/xblocks;
		}
		ny = maxtpb/nx;
		while ((NY/ny)*ny != NY)
			ny--;
		yblocks = NY/ny;
		x_wide.nblocks  = dim3(xblocks, yblocks);
		x_wide.nthreads = dim3(nx, ny);
	}
} device_init;

__global__ void
clear_gpu_mem_kernel(real *mem, const int n)
{
	for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
	     idx < n;
	     idx += blockDim.x * gridDim.x) {
		mem[idx] = 0.0;
	}
}

void
clear_gpu_mem(real *mem, const int n)
{
	int nthreads = devprop.maxThreadsPerBlock;
	int nblocks = devprop.multiProcessorCount * 4;
	clear_gpu_mem_kernel<<<nblocks, nthreads>>>(mem, n);
}

#include "boundary.cu"
#include "initial.cu"

// Basic math operators

#include "basic_ops.cu"

// Derivates

#include "derivative.cu"

// Gradient implementations

#include "grad/default.cu"
#include "grad/old_order.cu"
#include "grad/grad_flags.cu"
#include "grad/linear_load.cu"
#include "grad/noshared.cu"
#include "grad/simple.cu"
#include "grad/x_load.cu"
#include "grad/y_load.cu"
#include "grad/three.cu"

// Divergence implementations

#include "div/default.cu"
#include "div/same.cu"
#include "div/three.cu"

// Curl implementations

#include "curl/default.cu"
#define CURL_LAUNCH_BOUNDS
#include "curl/default.cu"
#undef CURL_LAUNCH_BOUNDS

// Del^2 implementations

#include "del2/default.cu"
#include "del2/same.cu"

#include "gpuomega.cu"

