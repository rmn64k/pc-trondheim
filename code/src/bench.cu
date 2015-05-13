#include <sys/time.h>

#include <vector>

#include "common.h"
#include "device.h"
#include "verify_ops.h"
#include "opt_parser.h"

real x_0 = 0.0;
real y_0 = 0.0;
real z_0 = 0.0;
real dx = (2 * M_PI - 0.0) / (NX - 1);
real dy = (2 * M_PI - 0.0) / (NY - 1);
real dz = (2 * M_PI - 0.0) / (NZ - 1);

struct timings {
	float low, high, avg;
};

typedef timings(*bench_op_type)(vf3dgpu &, vf3dgpu &, int);
typedef void(*call_op_type)(vf3dgpu &, vf3dgpu &);

template<call_op_type call_op> timings
bench_op(vf3dgpu &a, vf3dgpu &b, int count)
{
	float elapsedMs;
	timings timings;
	double start, stop;

	timings.avg = 0;
	for (int i = 0; i < count; i++) {
		check_cuda(cudaDeviceSynchronize());
		start = read_time_ms();
		call_op(a, b);
		check_cuda(cudaPeekAtLastError());
		check_cuda(cudaDeviceSynchronize());
		stop = read_time_ms();
		elapsedMs = stop - start;
		if (i == 0)
			timings.low = timings.high = elapsedMs;
		else if (elapsedMs < timings.low)
			timings.low = elapsedMs;
		else if (elapsedMs > timings.high)
			timings.high = elapsedMs;
		timings.avg += elapsedMs;
	}
	timings.avg /= count;
	return timings;
}

void
warm_up(bench_op_type bench, vf3dgpu &a, vf3dgpu &b)
{
	puts("Warm up.");
	float ttot = 0.0;
	while (ttot < 2000.0) {
		timings timings = bench(a, b, 10);
		ttot += timings.avg * 10.0;
	}
	puts("Done.");
}

void
print_timings(std::string name, timings timings)
{
	printf("%s: %.20f %.20f %.20f\n", name.c_str(), timings.low, timings.high, timings.avg);
}

void
run_bench_grad(std::string name, bench_op_type bench, vf3dgpu &f, vf3dgpu &u, int count, bool verify)
{
	timings timings;
	if (verify)
		clear_gpu_mem(u.mem(), vfmemsize(3)/sizeof(real));
	timings = bench(f, u, count);
	print_timings(name, timings);
	if (verify) {
		vf3dhost uh(3);
		u.copy_to_host(uh);
		real3 maxErr = check_grad(uh);
		uh.free();
		printf("maxErr: (%.20f, %.20f, %.20f).\n", maxErr.x, maxErr.y, maxErr.z);
		printf("relErr: (%.20f, %.20f, %.20f).\n", maxErr.x/pow(dx,6), maxErr.y/pow(dy,6), maxErr.z/pow(dz,6));
	}
}

void
bench_grad(std::string name, int count, bool verify)
{
	printf("Test gradient:\n");
	printf("f memory: %.2f MB.\n", vfmemsize(1)/(1024.0*1024.0));
	printf("u memory: %.2f MB.\n", vfmemsize(3)/(1024.0*1024.0));

	puts("Initalizing function values.");
	vf3dgpu f(1);
	vf3dgpu u(3);
	init_field(f, TEST_TRIG_INIT);
	apply_periodic_bc(f);
	puts("Done.");

	warm_up(bench_op<grad_default>, f, u);

	if (name == "default")
		run_bench_grad("default", bench_op<grad_default>, f, u, count, verify);
	else if (name == "old_order")
		run_bench_grad("old_order", bench_op<grad_old_order>, f, u, count, verify);
	else if (name == "simple")
		run_bench_grad("simple", bench_op<grad_simple>, f, u, count, verify);
	else if (name == "noshared")
		run_bench_grad("noshared", bench_op<grad_noshared>, f, u, count, verify);
	else if (name == "flags")
		run_bench_grad("flags", bench_op<grad_flags>, f, u, count, verify);
	else if (name == "x_load")
		run_bench_grad("x_load", bench_op<grad_x_load>, f, u, count, verify);
	else if (name == "y_load")
		run_bench_grad("y_load", bench_op<grad_y_load>, f, u, count, verify);
	else if (name == "linear_load")
		run_bench_grad("linear_load", bench_op<grad_linear_load>, f, u, count, verify);
	else if (name == "three")
		run_bench_grad("three", bench_op<grad_three>, f, u, count, verify);
	else if (name == "all") {
		run_bench_grad("default", bench_op<grad_default>, f, u, count, verify);
		run_bench_grad("old_order", bench_op<grad_old_order>, f, u, count, verify);
		run_bench_grad("simple", bench_op<grad_simple>, f, u, count, verify);
		run_bench_grad("noshared", bench_op<grad_noshared>, f, u, count, verify);
		run_bench_grad("flags", bench_op<grad_flags>, f, u, count, verify);
		run_bench_grad("x_load", bench_op<grad_x_load>, f, u, count, verify);
		run_bench_grad("y_load", bench_op<grad_y_load>, f, u, count, verify);
		run_bench_grad("linear_load", bench_op<grad_linear_load>, f, u, count, verify);
		run_bench_grad("three", bench_op<grad_three>, f, u, count, verify);
	}
	else {
		printf("No grad kernel named '%s'\n", name.c_str());
		return;
	}

	f.free();
	u.free();
}

void
run_bench_div(std::string name, bench_op_type bench, vf3dgpu &u, vf3dgpu &f, int count, bool verify)
{
	timings timings;
	if (verify)
		clear_gpu_mem(f.mem(), vfmemsize(1)/sizeof(real));
	timings = bench(u, f, count);
	print_timings(name, timings);
	if (verify) {
		vf3dhost fh(1);
		f.copy_to_host(fh);
		real maxErr = check_div(fh);
		fh.free();
		printf("maxErr: (%.20f).\n", maxErr);
		printf("relErr: (%.20f).\n", maxErr/pow(max(max(dx,dy),dz),6));
	}
}

void
bench_div(std::string name, int count, bool verify)
{
	puts("Test divergence:");
	printf("u memory: %.2f MB.\n", vfmemsize(3)/(1024.0*1024.0));
	printf("f memory: %.2f MB.\n", vfmemsize(1)/(1024.0*1024.0));

	puts("Initalizing vector field.");
	vf3dgpu u(3);
	vf3dgpu f(1);
	init_field(u, TEST_TRIG_INIT);
	apply_periodic_bc(u);
	puts("Done.");

	warm_up(bench_op<div_default>, u, f);

	if (name == "default")
		run_bench_div("default", bench_op<div_default>, u, f, count, verify);
	else if (name == "same")
		run_bench_div("same", bench_op<div_same>, u, f, count, verify);
	else if (name == "three")
		run_bench_div("three", bench_op<div_three>, u, f, count, verify);
	else if (name == "all") {
		run_bench_div("default", bench_op<div_default>, u, f, count, verify);
		run_bench_div("same", bench_op<div_same>, u, f, count, verify);
		run_bench_div("three", bench_op<div_three>, u, f, count, verify);
	}
	else {
		printf("No div kernel named '%s'\n", name.c_str());
		return;
	}

	f.free();
	u.free();
}

void
run_bench_curl(std::string name, bench_op_type bench, vf3dgpu &u, vf3dgpu &omega, int count, bool verify)
{
	timings timings;
	if (verify)
		clear_gpu_mem(omega.mem(), vfmemsize(3)/sizeof(real));
	timings = bench(u, omega, count);
	print_timings(name, timings);
	if (verify) {
		vf3dhost omegah(3);
		omega.copy_to_host(omegah);
		real3 maxErr = check_curl(omegah);
		omegah.free();
		printf("maxErr: (%.20f, %.20f, %.20f).\n", maxErr.x, maxErr.y, maxErr.z);
		printf("relErr: (%.20f, %.20f, %.20f).\n", maxErr.x/pow(dx,6), maxErr.y/pow(dy,6), maxErr.z/pow(dz,6));
	}
}

void
bench_curl(std::string name, int count, bool verify)
{
	puts("Test curl:");
	printf("u memory: %.2f MB.\n", vfmemsize(3)/(1024.0*1024.0));
	printf("omega memory: %.2f MB.\n", vfmemsize(3)/(1024.0*1024.0));

	puts("Initalizing vector field.");
	vf3dgpu u(3);
	vf3dgpu omega(3);

	init_field(u, TEST_TRIG_INIT);
	apply_periodic_bc(u);
	puts("Done.");

	warm_up(bench_op<curl_default>, u, omega);

	if (name == "default")
		run_bench_curl("default", bench_op<curl_default>, u, omega, count, verify);
	else if (name == "lb")
		run_bench_curl("lb", bench_op<curl_lb>, u, omega, count, verify);
	else if (name == "all") {
		run_bench_curl("default", bench_op<curl_default>, u, omega, count, verify);
		run_bench_curl("lb", bench_op<curl_lb>, u, omega, count, verify);
	}
	else {
		printf("No curl kernel named '%s'\n", name.c_str());
		return;
	}

	omega.free();
	u.free();
}

void
run_bench_del2(std::string name, bench_op_type bench, vf3dgpu &f, vf3dgpu &d2f, int count, bool verify)
{
	timings timings;
	if (verify)
		clear_gpu_mem(d2f.mem(), vfmemsize(1)/sizeof(real));
	timings = bench(f, d2f, count);
	print_timings(name, timings);
	if (verify) {
		vf3dhost d2fh(1);
		d2f.copy_to_host(d2fh);
		real maxErr = check_del2(d2fh);
		d2fh.free();
		printf("maxErr: %.20f.\n", maxErr);
		printf("relErr: (%.20f).\n", maxErr/pow(max(max(dx,dy),dz),6));
	}
}

void
bench_del2(std::string name, int count, bool verify)
{
	puts("Test del2:");
	printf("d memory: %.2f MB.\n", vfmemsize(1)/(1024.0*1024.0));
	printf("d2f memory: %.2f MB.\n", vfmemsize(1)/(1024.0*1024.0));

	puts("Initalizing vector field.");
	vf3dgpu f(1);
	vf3dgpu d2f(1);
	init_field(f, TEST_TRIG_INIT);
	apply_periodic_bc(f);
	puts("Done.");

	warm_up(bench_op<del2_default>, f, d2f);

	if (name == "default")
		run_bench_del2("default", bench_op<del2_default>, f, d2f, count, verify);
	else if (name == "same")
		run_bench_del2("same", bench_op<del2_same>, f, d2f, count, verify);
	else if (name == "all") {
		run_bench_del2("default", bench_op<del2_default>, f, d2f, count, verify);
		run_bench_del2("same", bench_op<del2_same>, f, d2f, count, verify);
	}
	else {
		printf("No del2 kernel named '%s'\n", name.c_str());
		return;
	}

	d2f.free();
	f.free();
}

void
bench_add2(std::string name, int count, bool verify)
{
	puts("Test add2:");

	puts("Initalizing fields.");
	vf3dgpu a(1);
	vf3dgpu b(1);
	init_field(a, TEST_TRIG_INIT);
	puts("Done.");

	clear_gpu_mem(b.mem(), vfmemsize(1)/sizeof(real));

	add2(-1, a, 5, a, b);

	vf3dhost bh(1);
	b.copy_to_host(bh);

	real maxErr = check_add2(bh);
	bh.free();
	printf("maxErr: %.20f.\n", maxErr);

	a.free();
	b.free();
}

void
bench_add3(std::string name, int count, bool verify)
{
	puts("Test add3:");

	puts("Initalizing fields.");
	vf3dgpu a(1);
	vf3dgpu b(1);
	init_field(a, TEST_TRIG_INIT);
	puts("Done.");

	clear_gpu_mem(b.mem(), vfmemsize(1)/sizeof(real));

	add3(-1, a, 2, a, 3, a, b);

	vf3dhost bh(1);
	b.copy_to_host(bh);

	real maxErr = check_add2(bh);
	bh.free();
	printf("maxErr: %.20f.\n", maxErr);

	a.free();
	b.free();
}

void
bench_dotmul(std::string name, int count, bool verify)
{
	puts("Test dotmul:");

	puts("Initalizing fields.");
	vf3dgpu a(3);
	vf3dgpu b(1);
	init_field(a, TEST_TRIG_INIT);
	puts("Done.");

	clear_gpu_mem(b.mem(), vfmemsize(1)/sizeof(real));

	dotmul3(a, a, b);

	vf3dhost bh(1);
	b.copy_to_host(bh);

	real maxErr = check_dotmul(bh);
	bh.free();
	printf("maxErr: %.20f.\n", maxErr);

	a.free();
	b.free();
}

struct bench_opts {
	int count;
	std::string op;
	std::string name;
	bool verify;
	bool help;

	bench_opts(): count(100), op("all"), name("all"), verify(false), help(false) { }

	void usage() {
		puts(
			"\n"
			"Usage: bench [opt=val | verify | help] ...\n"
			"\n"
			"Options with defaults:\n"
			"count=100           - number of iterations\n"
			"op=all              - [all|grad|div|curl|...]\n"
			"name=all            - [all|name of op]\n"
			"verify              - test the op for correctness\n"
			"help                - only show this message\n"
			"\n"
			"Example: bench count=200 op=grad name=default\n");
		exit(1);
	}

	void parse(int argc, char *argv[]) {
		int pstop;
		opt_parser optp;
		optp.int_opt("count", &count);
		optp.string_opt("op", &op);
		optp.string_opt("name", &name);
		optp.toggle_opt("verify", &verify);
		optp.toggle_opt("help", &help);
		if ((pstop = optp.parse(argc, argv)) != argc) {
			printf("Unknow option: '%s'\n", argv[pstop]);
			usage();
		}
		if (help)
			usage();
	}
};

int
main(int argc, char *argv[])
{
	bench_opts opts;
	opts.parse(argc, argv);
	if (opts.op == "all") {
		bench_grad("all", opts.count, opts.verify);
		bench_div("all", opts.count, opts.verify);
		bench_curl("all", opts.count, opts.verify);
		bench_del2("all", opts.count, opts.verify);
	}
	else {
		if (opts.op == "grad")
			bench_grad(opts.name, opts.count, opts.verify);
		else if (opts.op == "div")
			bench_div(opts.name, opts.count, opts.verify);
		else if (opts.op == "curl")
			bench_curl(opts.name, opts.count, opts.verify);
		else if (opts.op == "del2")
			bench_del2(opts.name, opts.count, opts.verify);
		else if (opts.op == "add2")
			bench_add2(opts.name, opts.count, opts.verify);
		else if (opts.op == "add3")
			bench_add3(opts.name, opts.count, opts.verify);
		else if (opts.op == "dotmul")
			bench_dotmul(opts.name, opts.count, opts.verify);
		else
			opts.usage();
	}
	printf("Reported timer resolution: %ld ns\n", time_resolution_ns());
	return 0;
}
