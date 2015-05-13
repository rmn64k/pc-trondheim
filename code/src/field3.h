#ifndef CUDA_FD_FIELD3_H
#define CUDA_FD_FIELD3_H

//ensure alignment to cache line boundaries
#if !defined(SINGLE_PRECISION)
const int ALIGN_X = 16;
#else
const int ALIGN_X = 32;
#endif
const int PAD_X_SIZE = (ALIGN_X + (NX + NGHOST + ALIGN_X - 1)/ALIGN_X*ALIGN_X);

__host__ __device__ inline size_t
vfystride()
{
#ifndef NOALIGN
	return PAD_X_SIZE;
#else
	return NX + 2 * NGHOST;
#endif
}

__host__ __device__ inline size_t
vfzstride()
{
	return (NY + 2 * NGHOST) * vfystride();
}

__host__ __device__ inline size_t
vfvstride()
{
	return (NZ + 2 * NGHOST) * vfzstride();
}

inline size_t
vfmemsize(int nv)
{
	return nv * vfvstride() * sizeof(real);
}

__host__ __device__ inline size_t
vfidx(int xi, int yi, int zi, int vi = 0)
{
#ifndef NOALIGN
	return vi * (NZ + 2 * NGHOST) * (NY + 2 * NGHOST) * PAD_X_SIZE +
	    (zi * (NY + 2 * NGHOST) + yi) * PAD_X_SIZE + xi + ALIGN_X - NGHOST;
#else
	return vi * (NZ + 2 * NGHOST) * (NY + 2 * NGHOST) * (NX + 2 * NGHOST) +
	    (zi * (NY + 2 * NGHOST) + yi) * (NX + 2 * NGHOST) + xi;
#endif
}

inline real *
vfvar(real *base, int vi)
{
	return &base[vfvstride() * vi];
}

__host__ __device__ inline size_t
vfinsideidx(int xi, int yi, int zi, int vi = 0)
{
	return vfidx(xi + NGHOST, yi + NGHOST, zi + NGHOST, vi);
}

// Classes for variable fields memory storage.
class vf3dhost {
	real *m_mem;
	int m_nv;
	vf3dhost(real *mem, int nv): m_mem(mem), m_nv(nv) { }
public:
	vf3dhost(): m_mem(NULL), m_nv(0) { }
	vf3dhost(int nv): m_nv(nv) {
		check_cuda(cudaHostAlloc(&m_mem, vfmemsize(nv), cudaHostAllocDefault));
	}
	real *mem() { return m_mem; }
	void free() { cudaFreeHost(m_mem); }
	int varcount() { return m_nv; }
	vf3dhost subfield(int vi, int nv) {
		return vf3dhost(m_mem + vfvstride() * vi, nv);
	}
};

class vf3dgpu {
	real *m_mem;
	int m_nv;
	vf3dgpu(real *mem, int nv): m_mem(mem), m_nv(nv) { }
public:
	vf3dgpu(): m_mem(NULL), m_nv(0) { }
	vf3dgpu(int nv): m_nv(nv) {
		check_cuda(cudaMalloc(&m_mem, vfmemsize(nv)));
	}
	real *mem() { return m_mem; };
	void free() { cudaFree(m_mem); }
	int varcount() { return m_nv; }
	vf3dgpu subfield(int vi, int nv) {
		return vf3dgpu(m_mem + vfvstride() * vi, nv);
	}
	void copy_to_host(vf3dhost &dst) {
		check_cuda(cudaMemcpy(dst.mem(), m_mem, vfmemsize(m_nv), cudaMemcpyDeviceToHost));
	}
	void copy_from_host(vf3dhost &src) {
		check_cuda(cudaMemcpy(m_mem, src.mem(), vfmemsize(m_nv), cudaMemcpyHostToDevice));
	}
};

#endif
