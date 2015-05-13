#include "common.h"
#include "fd_base.h"

__global__ void
grad_kernel(const real* __restrict__ f, const size_t fystride, real * __restrict__ u, const size_t uystride,
    const real xfactor, const real yfactor, const real zfactor)
{
	__shared__ real fs[NY_TILE + 2 * NGHOST][NX_TILE + 2 * NGHOST];
	// Shared memory indices
	const int xsi = threadIdx.x + NGHOST;
	const int ysi = threadIdx.y + NGHOST;
	// Global indices
	const int xi = blockIdx.x * blockDim.x + threadIdx.x + NGHOST;
	const int yi = blockIdx.y * blockDim.y + threadIdx.y + NGHOST;

	const size_t uzstride = simple_vfield3::gpu_field_index(0, 0, 1, 0, uystride);
	const size_t uvstride = simple_vfield3::gpu_field_index(0, 0, 0, 1, uystride);
	const size_t fzstride = ghost_sfield3::gpu_field_index(0, 0, 1, 0, fystride);

	size_t uidx = xi - NGHOST + uystride * (yi - NGHOST);
	size_t fidx = xi + fystride * yi + fzstride * 2;

	// Z-wise iteration values
	real behind3,
	    behind2  = f[fidx - fzstride * 2],
	    behind1  = f[fidx - fzstride    ],
	    current  = f[fidx               ],
	    forward1 = f[fidx + fzstride    ],
	    forward2 = f[fidx + fzstride * 2],
	    forward3 = f[fidx + fzstride * 3];

	for (int zi = NGHOST; zi < NZ + NGHOST; zi++) {
		fidx += fzstride;
		// Iterate through z dimension in registers
		behind3  = behind2;
		behind2  = behind1;
		behind1  = current;
		current  = forward1;
		forward1 = forward2;
		forward2 = forward3;
		forward3 = f[fidx + fzstride * 3];

		// Load x-y tile to shared memory
		__syncthreads();
		fs[ysi][xsi] = current;
		if (threadIdx.x < NGHOST) {
			fs[ysi][xsi - NGHOST]  = f[fidx - NGHOST];
			fs[ysi][xsi + NX_TILE] = f[fidx + NX_TILE];
		}
		if (threadIdx.y < NGHOST) {
			fs[ysi - NGHOST][xsi]  = f[fidx - fystride * NGHOST];
			fs[ysi + NY_TILE][xsi] = f[fidx + fystride * NY_TILE];
		}
		__syncthreads();

		// Compute the gradient
		u[uidx] =
		    xfactor * fd1D(fs[ysi][xsi - 3], fs[ysi][xsi - 2], fs[ysi][xsi - 1], fs[ysi][xsi + 1],
		    fs[ysi][xsi + 2], fs[ysi][xsi + 3]);

		u[uidx + uvstride] =
		    yfactor * fd1D(fs[ysi - 3][xsi], fs[ysi - 2][xsi], fs[ysi - 1][xsi], fs[ysi + 1][xsi],
		    fs[ysi + 2][xsi], fs[ysi + 3][xsi]);

		u[uidx + uvstride * 2] =
		    zfactor * fd1D(behind3, behind2, behind1, forward1, forward2, forward3);

		uidx += uzstride;
	}
}

void
grad(ghost_sfield3 &f, simple_vfield3 &u, real dx, real dy, real dz)
{
	cudaEvent_t start, stop;
	check_cuda(cudaEventCreate(&start));
	check_cuda(cudaEventCreate(&stop));

	dim3 threadBlocks(NX / NX_TILE, NY / NY_TILE);
	dim3 threadsPerBlock(NX_TILE, NY_TILE);

	f.mem().copy_to;

	check_cuda(cudaEventRecord(start));
	grad_kernel<<<threadBlocks, threadsPerBlock>>>(f.mem(), f.mem().gwidth(), u.mem(),
	    u.mem().gwidth(), 1.0 / dx, 1.0 / dy, 1.0 / dz);
	check_cuda(cudaPeekAtLastError());
	check_cuda(cudaEventRecord(stop));
	check_cuda(cudaDeviceSynchronize());

	u.mem().copy_to_host();

	check_cuda(cudaEventSynchronize(stop));
	float elapsedMs;
	check_cuda(cudaEventElapsedTime(&elapsedMs, start, stop));
	printf("Elapsed time: %f\n", elapsedMs);
}
