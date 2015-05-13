__global__ void
dotmul3_kernel(const real * __restrict__ a, const real * __restrict__ b,
    real * __restrict__ c)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + NGHOST;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + NGHOST;

	int idx1 = vfidx(xi, yi, NGHOST);
	int idx2 = vfidx(xi, yi, NGHOST, 1);
	int idx3 = vfidx(xi, yi, NGHOST, 2);

	int stride = vfzstride();

	for (int i = 0; i < NZ; i++) {
		c[idx1 + i*stride] = a[idx1 + i*stride] * b[idx1 + i*stride]
		    + a[idx2 + i*stride] * b[idx2 + i*stride]
		    + a[idx3 + i*stride] * b[idx3 + i*stride];
	}
}

void
dotmul3(vf3dgpu &a, vf3dgpu &b, vf3dgpu &c)
{
	dotmul3_kernel<<<x_wide.nblocks, x_wide.nthreads>>>(a.mem(), b.mem(), c.mem());
}

__global__ void
add2_kernel(real c1, const real *t1, real c2, const real *t2, real *res)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + NGHOST;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + NGHOST;
	int idx = vfidx(xi, yi, NGHOST);
	int stride = vfzstride();

	for (int i = 0; i < NZ; i++) {
		res[idx] = c1 * t1[idx] + c2 * t2[idx];
		idx += stride;
	}
}

__global__ void
add3_kernel(real c1, const real *t1, real c2, const real *t2,
    real c3, const real *t3, real *res)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + NGHOST;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + NGHOST;
	int idx = vfidx(xi, yi, NGHOST);
	int stride = vfzstride();

	for (int i = 0; i < NZ; i++) {
		res[idx] = c1 * t1[idx] + c2 * t2[idx] + c3 * t3[idx];
		idx += stride;
	}
}

void
add2(real c1, vf3dgpu &a, real c2, vf3dgpu &b, vf3dgpu &c)
{
	for (int vi = 0; vi < a.varcount(); vi++) {
		add2_kernel<<<x_wide.nblocks, x_wide.nthreads>>>(c1, vfvar(a.mem(), vi),
		    c2, vfvar(b.mem(), vi), vfvar(c.mem(), vi));
	}
}

void
add3(real c1, vf3dgpu &a, real c2, vf3dgpu &b, real c3, vf3dgpu &c, vf3dgpu &d)
{
	for (int vi = 0; vi < a.varcount(); vi++) {
		add3_kernel<<<x_wide.nblocks, x_wide.nthreads>>>(c1, vfvar(a.mem(), vi),
		    c2, vfvar(b.mem(), vi), c3, vfvar(c.mem(), vi), vfvar(d.mem(), vi));
	}
}
