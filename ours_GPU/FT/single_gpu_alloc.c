/*
MIT License

Copyright (c) 2021 Parallel Applications Modelling Group - GMAP
	GMAP website: https://gmap.pucrs.br

	Pontifical Catholic University of Rio Grande do Sul (PUCRS)
	Av. Ipiranga, 6681, Porto Alegre - Brazil, 90619-900

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do sny, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

------------------------------------------------------------------------------

The original NPB 3.4.1 version was written in Fortran and belongs to:
	http://www.nas.nasa.gov/Software/NPB/

Authors of the Fortran code:
	D. Bailey
	W. Saphir
	H. Jin

------------------------------------------------------------------------------

The serial C++ version is a translation of the original NPB 3.4.1
Serial C++ version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-SER

Authors of the C++ code:
	Dalvan Griebler <dalvangriebler@gmail.com>
	Gabriell Araujo <hexenoften@gmail.com>
	Júnior Löff <loffjh@gmail.com>

------------------------------------------------------------------------------

The OpenMP version is a parallel implementation of the serial C++ version
OpenMP version: https://github.com/GMAP/NPB-CPP/tree/master/NPB-OMP

Authors of the OpenMP code:
	Júnior Löff <loffjh@gmail.com>

*/

#include "omp.h"
#include "../common/npb-CPP.h"
#include "npbparams.h"

#include <string.h>
#include <stdbool.h>
#include <ittnotify.h>

/*
 * ---------------------------------------------------------------------
 * u0, u1, u2 are the main arrays in the problem.
 * depending on the decomposition, these arrays will have different
 * dimensions. to accomodate all possibilities, we allocate them as
 * one-dimensional arrays and pass them to subroutines for different
 * views
 * - u0 contains the initial (transformed) initial condition
 * - u1 and u2 are working arrays
 * - twiddle contains exponents for the time evolution operator.
 * ---------------------------------------------------------------------
 * large arrays are in common so that they are allocated on the
 * heap rather than the stack. this common block is not
 * referenced directly anywhere else. padding is to avoid accidental
 * cache problems, since all array sizes are powers of two.
 * ---------------------------------------------------------------------
 * we need a bunch of logic to keep track of how
 * arrays are laid out.
 *
 * note: this serial version is the derived from the parallel 0D case
 * of the ft NPB.
 * the computation proceeds logically as
 *
 * set up initial conditions
 * fftx(1)
 * transpose (1->2)
 * ffty(2)
 * transpose (2->3)
 * fftz(3)
 * time evolution
 * fftz(3)
 * transpose (3->2)
 * ffty(2)
 * transpose (2->1)
 * fftx(1)
 * compute residual(1)
 *
 * for the 0D, 1D, 2D strategies, the layouts look like xxx
 *
 *            0D        1D        2D
 * 1:        xyz       xyz       xyz
 * 2:        xyz       xyz       yxz
 * 3:        xyz       zyx       zxy
 * the array dimensions are stored in dims(coord, phase)
 * ---------------------------------------------------------------------
 * if processor array is 1x1 -> 0D grid decomposition
 *
 * cache blocking params. these values are good for most
 * RISC processors.
 * FFT parameters:
 * fftblock controls how many ffts are done at a time.
 * the default is appropriate for most cache-based machines
 * on vector machines, the FFT can be vectorized with vector
 * length equal to the block size, so the block size should
 * be as large as possible. this is the size of the smallest
 * dimension of the problem: 128 for class A, 256 for class B
 * and 512 for class C.
 * ---------------------------------------------------------------------
 */
#define FFTBLOCK_DEFAULT DEFAULT_BEHAVIOR
#define FFTBLOCKPAD_DEFAULT DEFAULT_BEHAVIOR
#define FFTBLOCK FFTBLOCK_DEFAULT
#define FFTBLOCKPAD FFTBLOCKPAD_DEFAULT
#define SEED 314159265.0
#define A 1220703125.0
#define PI 3.141592653589793238
#define ALPHA 1.0e-6
#define T_TOTAL 1
#define T_SETUP 2
#define T_FFT 3
#define T_EVOLVE 4
#define T_CHECKSUM 5
#define T_FFTX 6
#define T_FFTY 7
#define T_FFTZ 8
#define T_TRANSPOSE 9
#define T_MAX 9

/* global variables */
#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
static dcomplex sums[NITER_DEFAULT + 1];
static double twiddle[NTOTAL];
static dcomplex u[MAXDIM];
static dcomplex u0[NTOTAL];
static dcomplex u1[NTOTAL];
#else
static dcomplex(*sums);
static double (*twiddle_d)[NY][NX];
static dcomplex(*u_h);
static dcomplex(*u_d);
static dcomplex (*u0_d)[NY][NX];
static dcomplex (*u1_h)[NY][NX];
static dcomplex (*u1_d)[NY][NX];
static dcomplex(*yy1_d);
#endif

static int logd1;
static int logd2;
static int logd3;

/* function prototypes */
static void cfftz(int const is,
									int const m,
									int const n,
									int const ny,
									int const batch_size,
									dcomplex const u[MAXDIM],
									dcomplex x[],
									dcomplex y[]);
static void checksum(int i,
										 dcomplex u1[NZ][NY][NX],
										 dcomplex sums[NITER_DEFAULT + 1]);
static void compute_indexmap(double twiddle[NZ][NY][NX]);
static void compute_initial_conditions(dcomplex u0[NZ][NY][NX]);
static void evolve(dcomplex u0[NZ][NY][NX],
									 dcomplex u1[NZ][NY][NX],
									 double const twiddle[NZ][NY][NX]);
static void fft(dcomplex const u[MAXDIM],
								dcomplex x[NTOTAL],
								dcomplex xout[NTOTAL],
								dcomplex y[NTOTAL]);
static void ifft(dcomplex const u[MAXDIM],
								 dcomplex x[NTOTAL],
								 dcomplex xout[NTOTAL],
								 dcomplex y[NTOTAL]);
static void fft_init(int n);
static void fftz2(int const is,
									int const l,
									int const m,
									int const n,
									int const ny,
									int const batch_size,
									dcomplex const u[MAXDIM],
									dcomplex const x[],
									dcomplex y[]);
static int ilog2(int n);
static void init_ui(dcomplex u0[NZ][NY][NX],
										dcomplex u1[NZ][NY][NX],
										double twiddle[NZ][NY][NX]);
static void ipow46(double a,
									 int exponent,
									 double *result);
static void print_timers(void);
static void setup(void);
static void verify(int nt,
									 bool *verified,
									 char *class_npb);

/* ft */
int main(int argc, char **argv)
{
	setenv("OverrideDefaultFP64Settings", "1", 1);
	setenv("IGC_EnableDPEmulation", "1", 1);
	setenv("OMP_TARGET_OFFLOAD", "MANDATORY", 1);

	int host = omp_get_initial_device();
	int device = omp_get_default_device();

#if defined(DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION)
	printf(" DO_NOT_ALLOCATE_ARRAYS_WITH_DYNAMIC_MEMORY_AND_AS_SINGLE_DIMENSION mode on\n");
#else
	sums = malloc(sizeof(dcomplex) * (NITER_DEFAULT + 1));
	twiddle_d = omp_target_alloc_device(sizeof(double) * (NTOTAL), device);
	u_h = malloc(sizeof(dcomplex) * (MAXDIM));
	u_d = omp_target_alloc_device(sizeof(dcomplex) * (MAXDIM), device);
	u0_d = omp_target_alloc_device(sizeof(dcomplex) * (NTOTAL), device);
	u1_h = malloc(sizeof(dcomplex) * (NTOTAL));
	u1_d = omp_target_alloc_device(sizeof(dcomplex) * (NTOTAL), device);
	yy1_d = omp_target_alloc_device(sizeof(dcomplex) * (NTOTAL), device);
#endif
	int i;
	int iter;
	double total_time, mflops;
	bool verified;
	char class_npb;
	double t0, t1;

	/*
	 * ---------------------------------------------------------------------
	 * run the entire problem once to make sure all data is touched.
	 * this reduces variable startup costs, which is important for such a
	 * short benchmark. the other NPB 2 implementations are similar.
	 * ---------------------------------------------------------------------
	 */
#if defined(TIMERS_ENABLED)
	for (i = 0; i < T_MAX; i++)
	{
		timer_clear(i);
	}
#endif

	setup();

	/*
	 * ---------------------------------------------------------------------
	 * start over from the beginning. note that all operations must
	 * be timed, in contrast to other benchmarks.
	 * ---------------------------------------------------------------------
	 */

	__itt_resume();

#if defined(TIMERS_ENABLED)
	for (i = 0; i < T_MAX; i++)
	{
		timer_clear(i);
	}
#endif

	timer_start(T_TOTAL);

#if defined(TIMERS_ENABLED)
	timer_start(T_SETUP);
#endif

	// Initialize on CPU
	compute_initial_conditions(u1_h);
	fft_init(MAXDIM);

	// copy to device
	omp_target_memcpy(u1_d, u1_h, sizeof(dcomplex) * NTOTAL, 0, 0, device, host);
	omp_target_memcpy(u_d, u_h, sizeof(dcomplex) * MAXDIM, 0, 0, device, host);

#pragma omp target data use_device_ptr(twiddle_d, u_d, u0_d, u1_d, yy1_d)
	{
		compute_indexmap(twiddle_d);

#if defined(TIMERS_ENABLED)
		timer_stop(T_SETUP);
		timer_start(T_FFT);
#endif

		fft(u_d, (dcomplex *)u1_d, (dcomplex *)u0_d, yy1_d);

#if defined(TIMERS_ENABLED)
		timer_stop(T_FFT);
#endif
		for (iter = 1; iter <= NITER_DEFAULT; iter++)
		{
#if defined(TIMERS_ENABLED)
			timer_start(T_EVOLVE);
#endif

			evolve(u0_d, u1_d, twiddle_d);

#if defined(TIMERS_ENABLED)
			timer_stop(T_EVOLVE);
			timer_start(T_FFT);
#endif

			ifft(u_d, (dcomplex *)u1_d, (dcomplex *)u1_d, yy1_d);

#if defined(TIMERS_ENABLED)
			timer_stop(T_FFT);
			timer_start(T_CHECKSUM);
#endif

			checksum(iter, u1_d, sums);

#if defined(TIMERS_ENABLED)
			timer_stop(T_CHECKSUM);
#endif
		}
	} /* end of target data */

	timer_stop(T_TOTAL);
	total_time = timer_read(T_TOTAL);

	__itt_pause();

	verify(NITER_DEFAULT, &verified, &class_npb);

	if (total_time != 0.0)
	{
		mflops = 1.0e-6 * ((double)(NTOTAL)) *
						 (14.8157 + 7.19641 * log((double)(NTOTAL)) + (5.23518 + 7.21113 * log((double)(NTOTAL))) * NITER_DEFAULT) / total_time;
	}
	else
	{
		mflops = 0.0;
	}

	c_print_results((char *)"FT",
									class_npb,
									NX,
									NY,
									NZ,
									NITER_DEFAULT,
									total_time,
									mflops,
									(char *)"          floating point",
									verified,
									(char *)NPBVERSION,
									(char *)COMPILETIME,
									(char *)COMPILERVERSION,
									(char *)LIBVERSION,
									getenv("OMP_NUM_THREADS"),
									(char *)CS1,
									(char *)CS2,
									(char *)CS3,
									(char *)CS4,
									(char *)CS5,
									(char *)CS6,
									(char *)CS7);
#if defined(TIMERS_ENABLED)
	print_timers();
#endif

	return 0;
}

/*
 * ---------------------------------------------------------------------
 * computes NY N-point complex-to-complex FFTs of X using an algorithm due
 * to swarztrauber. X is both the input and the output array, while Y is a
 * scratch array. it is assumed that N = 2^M. before calling CFFTZ to
 * perform FFTs, the array U must be initialized by calling CFFTZ with is
 * set to 0 and M set to MX, where MX is the maximum value of M for any
 * subsequent call.
 * ---------------------------------------------------------------------
 */
static void cfftz(int const is,
									int const m,
									int const n,
									int const ny,
									int const batch_size,
									dcomplex const u[MAXDIM],
									dcomplex x[],
									dcomplex y[])
{
	int i, l, mx;

	/*
	 * ---------------------------------------------------------------------
	 * check if input parameters are invalid.
	 * ---------------------------------------------------------------------
	 */
	/*
	mx = (int)(u[0].real);
	if ((is != 1 && is != -1) || m < 1 || m > mx)
	{
		printf("CFFTZ: Either U has not been initialized, or else\n"
					 "one of the input parameters is invalid%5d%5d%5d\n",
					 is, m, mx);
		exit(EXIT_FAILURE);
	}
	*/

	/*
	 * ---------------------------------------------------------------------
	 * perform one variant of the Stockham FFT.
	 * ---------------------------------------------------------------------
	 */
#if defined(REF)
	for (l = 1; l <= m; l += 2)
	{
		fftz2(is, l, m, n, ny, u, x, y);
		if (l == m)
		{
			/*
			 * ---------------------------------------------------------------------
			 * copy Y to X.
			 * ---------------------------------------------------------------------
			 */
#pragma omp target teams distribute parallel for simd collapse(2)
			for (j = 0; j < n; j++)
			{
				for (i = 0; i < ny; i++)
				{
					x[j][i] = y[j][i];
				}
			}
			break;
		}
		fftz2(is, l + 1, m, n, ny, u, y, x);
	}
#else
	for (l = 1; l < m; l += 2)
	{
		fftz2(is, l, m, n, ny, batch_size, u, x, y);
		fftz2(is, l + 1, m, n, ny, batch_size, u, y, x);
	}

	if (m % 2 != 0)
	{
		fftz2(is, m, m, n, ny, batch_size, u, x, y);

#pragma omp target teams distribute parallel for simd
		for (i = 0; i < 2 * n * ny * batch_size; i++)
		{
			((double *)x)[i] = ((double *)y)[i];
		}
	}
#endif
}

static void checksum(int i,
										 dcomplex u1[NZ][NY][NX],
										 dcomplex sums[NITER_DEFAULT + 1])
{

	int j, q, r, s;
	double chk_worker[2] = {0.0, 0.0};

#pragma omp target teams distribute parallel for reduction(+ : chk_worker[0 : 2]) map(tofrom : chk_worker[0 : 2])
	for (j = 1; j <= 1024; j++)
	{
		q = j % NX;
		r = (3 * j) % NY;
		s = (5 * j) % NZ;
		chk_worker[0] += u1[s][r][q].real;
		chk_worker[1] += u1[s][r][q].imag;
	}

	double chk_worker_real = chk_worker[0] / (double)(NTOTAL);
	double chk_worker_imag = chk_worker[1] / (double)(NTOTAL);
	printf(" T =%5d     Checksum =%22.12e%22.12e\n", i, chk_worker_real, chk_worker_imag);
	sums[i].real = chk_worker_real;
	sums[i].imag = chk_worker_imag;
}

#define r23 (0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5)
#define r46 (r23 * r23)
#define t23 (2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0 * 2.0)
#define t46 (t23 * t23)

/*
 * compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2
 * for time evolution exponent.
 */
static void compute_indexmap(double twiddle[NZ][NY][NX])
{
	int i, j, k, kk, kk2, jj, kj2, ii;
	double ap;

	/*
	 * ---------------------------------------------------------------------
	 * basically we want to convert the fortran indices
	 * 1 2 3 4 5 6 7 8
	 * to
	 * 0 1 2 3 -4 -3 -2 -1
	 * the following magic formula does the trick:
	 * mod(i-1+n/2, n) - n/2
	 * ---------------------------------------------------------------------
	 */
	ap = -4.0 * ALPHA * PI * PI;
#pragma omp target teams distribute parallel for simd collapse(3) private(i, j, kk, kk2, jj, kj2, ii)
	for (k = 0; k < NZ; k++)
	{
		kk = ((k + NZ / 2) % NZ) - NZ / 2;
		kk2 = kk * kk;
		for (j = 0; j < NY; j++)
		{
			jj = ((j + NY / 2) % NY) - NY / 2;
			kj2 = jj * jj + kk2;
			for (i = 0; i < NX; i++)
			{
				ii = ((i + NX / 2) % NX) - NX / 2;
				twiddle[k][j][i] = exp(ap * (double)(ii * ii + kj2));
			}
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * fill in array u0 with initial conditions from
 * random number generator
 * ---------------------------------------------------------------------
 */
static void compute_initial_conditions(dcomplex u0[NZ][NY][NX])
{
	int k, j;
	double x0, start, an, starts[NZ];
	start = SEED;

	/*
	 * ---------------------------------------------------------------------
	 * jump to the starting element for our first plane.
	 * ---------------------------------------------------------------------
	 */
	ipow46(A, 0, &an);
	randlc(&start, an);
	ipow46(A, 2 * NX * NY, &an);

	starts[0] = start;
	for (int k = 1; k < NZ; k++)
	{
		randlc(&start, an);
		starts[k] = start;
	}

/*
 * ---------------------------------------------------------------------
 * go through by z planes filling in one square at a time.
 * ---------------------------------------------------------------------
 */
// NOTE: Only outer loop can be run in parallel
#pragma omp parallel for
	for (k = 0; k < NZ; k++)
	{
		x0 = starts[k];
		for (j = 0; j < NY; j++)
		{
			// vranlc(2 * NX, &x0, A, (double *)&u0[k][j][0]);
			int n = 2 * NX;
			double *x_seed = &x0;
			double a = A;
			double *y = (double *)&u0[k][j][0];

			int i;
			double x, t1, t2, t3, t4, a1, a2, x1, x2, z;

			/*
			 * ---------------------------------------------------------------------
			 * break A into two parts such that A = 2^23 * A1 + A2.
			 * ---------------------------------------------------------------------
			 */
			t1 = r23 * a;
			a1 = (int)t1;
			a2 = a - t23 * a1;
			x = *x_seed;

			/*
			 * ---------------------------------------------------------------------
			 * generate N results. this loop is not vectorizable.
			 * ---------------------------------------------------------------------
			 */
			for (i = 0; i < n; i++)
			{
				/*
				 * ---------------------------------------------------------------------
				 * break X into two parts such that X = 2^23 * X1 + X2, compute
				 * Z = A1 * X2 + A2 * X1  (mod 2^23), and then
				 * X = 2^23 * Z + A2 * X2  (mod 2^46).
				 * ---------------------------------------------------------------------
				 */
				t1 = r23 * x;
				x1 = (int)t1;
				x2 = x - t23 * x1;
				t1 = a1 * x2 + a2 * x1;
				t2 = (int)(r23 * t1);
				z = t1 - t23 * t2;
				t3 = t23 * z + a2 * x2;
				t4 = (int)(r46 * t3);
				x = t3 - t46 * t4;
				y[i] = r46 * x;
			}
			*x_seed = x;
		}
	}
}

/*
 * ---------------------------------------------------------------------
 * evolve u0 -> u1 (t time steps) in fourier space
 * ---------------------------------------------------------------------
 */
static void evolve(dcomplex u0[NZ][NY][NX],
									 dcomplex u1[NZ][NY][NX],
									 double const twiddle[NZ][NY][NX])
{
	int i, j, k;
#pragma omp target teams distribute parallel for simd collapse(3)
	for (k = 0; k < NZ; k++)
	{
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
#if defined(REF)
				u0[k][j][i] = dcomplex_mul2(u0[k][j][i], twiddle[k][j][i]);
				u1[k][j][i] = u0[k][j][i];
#else
				u0[k][j][i].real *= twiddle[k][j][i];
				u0[k][j][i].imag *= twiddle[k][j][i];
				u1[k][j][i].real = u0[k][j][i].real;
				u1[k][j][i].imag = u0[k][j][i].imag;

				// u1[k][j][i].real = u0[k][j][i].real * twiddle[k][j][i];
				// u1[k][j][i].imag = u0[k][j][i].imag * twiddle[k][j][i];
				// u0[k][j][i].real = u1[k][j][i].real;
				// u0[k][j][i].imag = u1[k][j][i].imag;
#endif
			}
		}
	}
}

#define INDEX_3D(n3, n2, n1, i3, i2, i1) (((i3) * (n2) * (n1)) + ((i2) * (n1)) + (i1))

#define INDEX_ZYX INDEX_3D(NZ, NY, NX, k, j, i)
#define INDEX_YZX INDEX_3D(NY, NZ, NX, j, k, i)
#define INDEX_ZXY INDEX_3D(NZ, NX, NY, k, i, j)

static void fft(dcomplex const u[MAXDIM],
								dcomplex x[NTOTAL],
								dcomplex xout[NTOTAL],
								dcomplex y[NTOTAL])
{
	// fft:
	// zyx -> zxy
	// cfftz (1)
	// zxy -> zyx
	// cfftz (2)
	// zyx -> yzx
	// cfftz (3)
	// yzx -> zyx

#if defined(TIMERS_ENABLED)
	timer_start(T_TRANSPOSE);
#endif

// zyx -> zxy
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_ZYX;
				long idx_dst = INDEX_ZXY;
				y[idx_dst].real = x[idx_src].real;
				y[idx_dst].imag = x[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
	timer_start(T_FFTX);
#endif

	cfftz(1, logd1, NX, NY, NZ, u, y, xout);

#if defined(TIMERS_ENABLED)
	timer_stop(T_FFTX);
	timer_start(T_TRANSPOSE);
#endif

// zxy -> zyx
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_ZXY;
				long idx_dst = INDEX_ZYX;
				xout[idx_dst].real = y[idx_src].real;
				xout[idx_dst].imag = y[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
	timer_start(T_FFTY);
#endif

	cfftz(1, logd2, NY, NX, NZ, u, xout, y);

#if defined(TIMERS_ENABLED)
	timer_stop(T_FFTY);
	timer_start(T_TRANSPOSE);
#endif

// zyx -> yzx
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_ZYX;
				long idx_dst = INDEX_YZX;
				y[idx_dst].real = xout[idx_src].real;
				y[idx_dst].imag = xout[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
	timer_start(T_FFTZ);
#endif

	cfftz(1, logd3, NZ, NX, NY, u, y, xout);

#if defined(TIMERS_ENABLED)
	timer_stop(T_FFTZ);
	timer_start(T_TRANSPOSE);
#endif

	// yzx -> zyx
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_YZX;
				long idx_dst = INDEX_ZYX;
				xout[idx_dst].real = y[idx_src].real;
				xout[idx_dst].imag = y[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
#endif
}

static void ifft(dcomplex const u[MAXDIM],
								 dcomplex x[NTOTAL],
								 dcomplex xout[NTOTAL],
								 dcomplex y[NTOTAL])
{
	// ifft:
	// zyx -> yzx
	// cfftz (3)
	// yzx -> zyx
	// cfftz (2)
	// zyx -> zxy
	// cfftz (1)
	// zxy -> zyx

#if defined(TIMERS_ENABLED)
	timer_start(T_TRANSPOSE);
#endif

// zyx -> yzx
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_ZYX;
				long idx_dst = INDEX_YZX;
				y[idx_dst].real = x[idx_src].real;
				y[idx_dst].imag = x[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
	timer_start(T_FFTZ);
#endif

	cfftz(-1, logd3, NZ, NX, NY, u, y, xout);

#if defined(TIMERS_ENABLED)
	timer_stop(T_FFTZ);
	timer_start(T_TRANSPOSE);
#endif

// yzx -> zyx
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_YZX;
				long idx_dst = INDEX_ZYX;
				xout[idx_dst].real = y[idx_src].real;
				xout[idx_dst].imag = y[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
	timer_start(T_FFTY);
#endif

	cfftz(-1, logd2, NY, NX, NZ, u, xout, y);

#if defined(TIMERS_ENABLED)
	timer_stop(T_FFTY);
	timer_start(T_TRANSPOSE);
#endif

// zyx -> zxy
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_ZYX;
				long idx_dst = INDEX_ZXY;
				y[idx_dst].real = xout[idx_src].real;
				y[idx_dst].imag = xout[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
	timer_start(T_FFTX);
#endif

	cfftz(-1, logd1, NX, NY, NZ, u, y, xout);

#if defined(TIMERS_ENABLED)
	timer_stop(T_FFTX);
	timer_start(T_TRANSPOSE);
#endif

	// zxy -> zyx
#pragma omp target teams distribute parallel for simd collapse(3)
	for (long k = 0; k < NZ; k++)
	{
		for (long j = 0; j < NY; j++)
		{
			for (long i = 0; i < NX; i++)
			{
				long idx_src = INDEX_ZXY;
				long idx_dst = INDEX_ZYX;
				xout[idx_dst].real = y[idx_src].real;
				xout[idx_dst].imag = y[idx_src].imag;
			}
		}
	}

#if defined(TIMERS_ENABLED)
	timer_stop(T_TRANSPOSE);
#endif
}

/*
 * ---------------------------------------------------------------------
 * compute the roots-of-unity array that will be used for subsequent FFTs.
 * ---------------------------------------------------------------------
 */
static void fft_init(int n)
{
	int m, nu, ku, i, j, ln;
	double t, ti;
	/*
	 * ---------------------------------------------------------------------
	 * initialize the U array with sines and cosines in a manner that permits
	 * stride one access at each FFT iteration.
	 * ---------------------------------------------------------------------
	 */
	nu = n;
	m = ilog2(n);
	u_h[0] = dcomplex_create((double)m, 0.0);
	ku = 2;
	ln = 1;

	for (j = 1; j <= m; j++)
	{
		t = PI / ln;

#pragma omp parallel for firstprivate(t, j, ku) private(i, ti)
		for (i = 0; i <= ln - 1; i++)
		{
			ti = i * t;
#if defined(REF)
			u[i + ku - 1] = dcomplex_create(cos(ti), sin(ti));
#else
			u_h[i + ku - 1].real = cos(ti);
			u_h[i + ku - 1].imag = sin(ti);
#endif
		}

		ku = ku + ln;
		ln = 2 * ln;
	}
}

/*
 * ---------------------------------------------------------------------
 * performs the l-th iteration of the second variant of the stockham FFT
 * ---------------------------------------------------------------------
 */
static void fftz2(int const is,
									int const l,
									int const m,
									int const n,
									int const ny,
									int const batch_size,
									dcomplex const u[MAXDIM],
									dcomplex const x[],
									dcomplex y[])
{
	int k, n1, li, lj, lk, ku, i, j, i11, i12, i21, i22, b;
	dcomplex u1;

	/*
	 * ---------------------------------------------------------------------
	 * set initial parameters.
	 * ---------------------------------------------------------------------
	 */
	n1 = n / 2;
	lk = 1 << (l - 1);
	li = 1 << (m - l);
	lj = 2 * lk;
	ku = li;

#pragma omp target teams distribute parallel for simd collapse(4) \
		firstprivate(n1, lk, li, lj, ku) private(b, i, k, j, i11, i12, i21, i22, u1)
	for (b = 0; b < batch_size; b++)
	{
		for (i = 0; i <= li - 1; i++)
		{
			i11 = i * lk;
			i12 = i11 + n1;
			i21 = i * lj;
			i22 = i21 + lk;

			u1 = u[ku + i];
			if (is < 0)
			{
				u1.imag *= -1;
			}

			for (k = 0; k <= lk - 1; k++)
			{
				for (j = 0; j < ny; j++)
				{
#if defined(REF)
					dcomplex x11 = x[i11 + k][j];
					dcomplex x21 = x[i12 + k][j];
					y[i21 + k][j] = dcomplex_add(x11, x21);
					y[i22 + k][j] = dcomplex_mul(u1, dcomplex_sub(x11, x21));
#else
					int x11_idx = INDEX_3D(batch_size, n, ny, b, i11 + k, j);
					int x21_idx = INDEX_3D(batch_size, n, ny, b, i12 + k, j);
					dcomplex const *p_x11 = &x[x11_idx];
					dcomplex const *p_x21 = &x[x21_idx];

					int y_idx_a = INDEX_3D(batch_size, n, ny, b, i21 + k, j);
					y[y_idx_a].real = p_x11->real + p_x21->real;
					y[y_idx_a].imag = p_x11->imag + p_x21->imag;

					double x11_sub_x21_real = p_x11->real - p_x21->real;
					double x11_sub_x21_imag = p_x11->imag - p_x21->imag;

					int y_idx_b = INDEX_3D(batch_size, n, ny, b, i22 + k, j);
					y[y_idx_b].real = u1.real * x11_sub_x21_real - u1.imag * x11_sub_x21_imag;
					y[y_idx_b].imag = u1.real * x11_sub_x21_imag + u1.imag * x11_sub_x21_real;
#endif
				}
			}
		}
	}
}

static int ilog2(int n)
{
	int nn, lg;
	if (n == 1)
	{
		return 0;
	}
	lg = 1;
	nn = 2;
	while (nn < n)
	{
		nn *= 2;
		lg += 1;
	}
	return lg;
}

/*
 * ---------------------------------------------------------------------
 * touch all the big data
 * ---------------------------------------------------------------------
 */
static void init_ui(dcomplex u0[NZ][NY][NX],
										dcomplex u1[NZ][NY][NX],
										double twiddle[NZ][NY][NX])
{
	int i, j, k;
#pragma omp parallel for
	for (k = 0; k < NZ; k++)
	{
#if defined(REF)
		for (j = 0; j < NY; j++)
		{
			for (i = 0; i < NX; i++)
			{
				u0[k][j][i] = dcomplex_create(0.0, 0.0);
				u1[k][j][i] = dcomplex_create(0.0, 0.0);
				twiddle[k][j][i] = 0.0;
			}
		}
#else
		memset((void *)u0[k], 0, NX * NY * sizeof(dcomplex));
		memset((void *)u1[k], 0, NX * NY * sizeof(dcomplex));
		memset((void *)twiddle[k], 0, NX * NY * sizeof(double));
#endif
	}
}

/*
 * ---------------------------------------------------------------------
 * compute a^exponent mod 2^46
 * ---------------------------------------------------------------------
 */
static void ipow46(double a,
									 int exponent,
									 double *result)
{
	double q, r;
	int n, n2;

	/*
	 * --------------------------------------------------------------------
	 * use
	 * a^n = a^(n/2)*a^(n/2) if n even else
	 * a^n = a*a^(n-1)       if n odd
	 * -------------------------------------------------------------------
	 */
	*result = 1;
	if (exponent == 0)
	{
		return;
	}
	q = a;
	r = 1;
	n = exponent;

	while (n > 1)
	{
		n2 = n / 2;
		if (n2 * 2 == n)
		{
			randlc(&q, q);
			n = n2;
		}
		else
		{
			randlc(&r, q);
			n = n - 1;
		}
	}
	randlc(&r, q);
	*result = r;
}

static void print_timers(void)
{
	int i;
	double t, t_m;
	char *tstrings[T_MAX + 1];
	tstrings[1] = (char *)"          total ";
	tstrings[2] = (char *)"          setup ";
	tstrings[3] = (char *)"            fft ";
	tstrings[4] = (char *)"         evolve ";
	tstrings[5] = (char *)"       checksum ";
	tstrings[6] = (char *)"           fftx ";
	tstrings[7] = (char *)"           ffty ";
	tstrings[8] = (char *)"           fftz ";
	tstrings[9] = (char *)"      transpose ";

	t_m = timer_read(T_TOTAL);
	if (t_m <= 0.0)
	{
		t_m = 1.00;
	}
	for (i = 1; i <= T_MAX; i++)
	{
		t = timer_read(i);
		printf(" timer %2d(%16s) :%9.4f (%6.2f%%)\n",
					 i, tstrings[i], t, t * 100.0 / t_m);
	}
}

static void setup(void)
{
	FILE *fp;

	printf(" Size                : %4dx%4dx%4d\n", NX, NY, NZ);
	printf(" Iterations                  :%7d\n", NITER_DEFAULT);
	printf("\n");

	logd1 = ilog2(NX);
	logd2 = ilog2(NY);
	logd3 = ilog2(NZ);

	/*
	 * ---------------------------------------------------------------------
	 * set up info for blocking of ffts and transposes. this improves
	 * performance on cache-based systems. blocking involves
	 * working on a chunk of the problem at a time, taking chunks
	 * along the first, second, or third dimension.
	 *
	 * - in cffts1 blocking is on 2nd dimension (with fft on 1st dim)
	 * - in cffts2/3 blocking is on 1st dimension (with fft on 2nd and 3rd dims)
	 *
	 * since 1st dim is always in processor, we'll assume it's long enough
	 * (default blocking factor is 16 so min size for 1st dim is 16)
	 * the only case we have to worry about is cffts1 in a 2d decomposition.
	 * so the blocking factor should not be larger than the 2nd dimension.
	 * ---------------------------------------------------------------------
	 */
	/* block values were already set */
	/* fftblock = FFTBLOCK_DEFAULT; */
	/* fftblockpad = FFTBLOCKPAD_DEFAULT; */
	/* if(fftblock!=FFTBLOCK_DEFAULT){fftblockpad=fftblock+3;} */
}

static void verify(int nt,
									 bool *verified,
									 char *class_npb)
{
	int i;
	double err, epsilon;

	/*
	 * ---------------------------------------------------------------------
	 * reference checksums
	 * ---------------------------------------------------------------------
	 */
	dcomplex csum_ref[25 + 1];

	*class_npb = 'U';

	epsilon = 1.0e-12;
	*verified = false;

	if (NX == 64 && NY == 64 && NZ == 64 && nt == 6)
	{
		/*
		 * ---------------------------------------------------------------------
		 * sample size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'S';
		csum_ref[1] = dcomplex_create(5.546087004964E+02, 4.845363331978E+02);
		csum_ref[2] = dcomplex_create(5.546385409189E+02, 4.865304269511E+02);
		csum_ref[3] = dcomplex_create(5.546148406171E+02, 4.883910722336E+02);
		csum_ref[4] = dcomplex_create(5.545423607415E+02, 4.901273169046E+02);
		csum_ref[5] = dcomplex_create(5.544255039624E+02, 4.917475857993E+02);
		csum_ref[6] = dcomplex_create(5.542683411902E+02, 4.932597244941E+02);
	}
	else if (NX == 128 && NY == 128 && NZ == 32 && nt == 6)
	{
		/*
		 * ---------------------------------------------------------------------
		 * class_npb W size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'W';
		csum_ref[1] = dcomplex_create(5.673612178944E+02, 5.293246849175E+02);
		csum_ref[2] = dcomplex_create(5.631436885271E+02, 5.282149986629E+02);
		csum_ref[3] = dcomplex_create(5.594024089970E+02, 5.270996558037E+02);
		csum_ref[4] = dcomplex_create(5.560698047020E+02, 5.260027904925E+02);
		csum_ref[5] = dcomplex_create(5.530898991250E+02, 5.249400845633E+02);
		csum_ref[6] = dcomplex_create(5.504159734538E+02, 5.239212247086E+02);
	}
	else if (NX == 256 && NY == 256 && NZ == 128 && nt == 6)
	{
		/*
		 * ---------------------------------------------------------------------
		 * class_npb A size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'A';
		csum_ref[1] = dcomplex_create(5.046735008193E+02, 5.114047905510E+02);
		csum_ref[2] = dcomplex_create(5.059412319734E+02, 5.098809666433E+02);
		csum_ref[3] = dcomplex_create(5.069376896287E+02, 5.098144042213E+02);
		csum_ref[4] = dcomplex_create(5.077892868474E+02, 5.101336130759E+02);
		csum_ref[5] = dcomplex_create(5.085233095391E+02, 5.104914655194E+02);
		csum_ref[6] = dcomplex_create(5.091487099959E+02, 5.107917842803E+02);
	}
	else if (NX == 512 && NY == 256 && NZ == 256 && nt == 20)
	{
		/*
		 * --------------------------------------------------------------------
		 * class_npb B size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'B';
		csum_ref[1] = dcomplex_create(5.177643571579E+02, 5.077803458597E+02);
		csum_ref[2] = dcomplex_create(5.154521291263E+02, 5.088249431599E+02);
		csum_ref[3] = dcomplex_create(5.146409228649E+02, 5.096208912659E+02);
		csum_ref[4] = dcomplex_create(5.142378756213E+02, 5.101023387619E+02);
		csum_ref[5] = dcomplex_create(5.139626667737E+02, 5.103976610617E+02);
		csum_ref[6] = dcomplex_create(5.137423460082E+02, 5.105948019802E+02);
		csum_ref[7] = dcomplex_create(5.135547056878E+02, 5.107404165783E+02);
		csum_ref[8] = dcomplex_create(5.133910925466E+02, 5.108576573661E+02);
		csum_ref[9] = dcomplex_create(5.132470705390E+02, 5.109577278523E+02);
		csum_ref[10] = dcomplex_create(5.131197729984E+02, 5.110460304483E+02);
		csum_ref[11] = dcomplex_create(5.130070319283E+02, 5.111252433800E+02);
		csum_ref[12] = dcomplex_create(5.129070537032E+02, 5.111968077718E+02);
		csum_ref[13] = dcomplex_create(5.128182883502E+02, 5.112616233064E+02);
		csum_ref[14] = dcomplex_create(5.127393733383E+02, 5.113203605551E+02);
		csum_ref[15] = dcomplex_create(5.126691062020E+02, 5.113735928093E+02);
		csum_ref[16] = dcomplex_create(5.126064276004E+02, 5.114218460548E+02);
		csum_ref[17] = dcomplex_create(5.125504076570E+02, 5.114656139760E+02);
		csum_ref[18] = dcomplex_create(5.125002331720E+02, 5.115053595966E+02);
		csum_ref[19] = dcomplex_create(5.124551951846E+02, 5.115415130407E+02);
		csum_ref[20] = dcomplex_create(5.124146770029E+02, 5.115744692211E+02);
	}
	else if (NX == 512 && NY == 512 && NZ == 512 && nt == 20)
	{
		/*
		 * ---------------------------------------------------------------------
		 * class_npb C size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'C';
		csum_ref[1] = dcomplex_create(5.195078707457E+02, 5.149019699238E+02);
		csum_ref[2] = dcomplex_create(5.155422171134E+02, 5.127578201997E+02);
		csum_ref[3] = dcomplex_create(5.144678022222E+02, 5.122251847514E+02);
		csum_ref[4] = dcomplex_create(5.140150594328E+02, 5.121090289018E+02);
		csum_ref[5] = dcomplex_create(5.137550426810E+02, 5.121143685824E+02);
		csum_ref[6] = dcomplex_create(5.135811056728E+02, 5.121496764568E+02);
		csum_ref[7] = dcomplex_create(5.134569343165E+02, 5.121870921893E+02);
		csum_ref[8] = dcomplex_create(5.133651975661E+02, 5.122193250322E+02);
		csum_ref[9] = dcomplex_create(5.132955192805E+02, 5.122454735794E+02);
		csum_ref[10] = dcomplex_create(5.132410471738E+02, 5.122663649603E+02);
		csum_ref[11] = dcomplex_create(5.131971141679E+02, 5.122830879827E+02);
		csum_ref[12] = dcomplex_create(5.131605205716E+02, 5.122965869718E+02);
		csum_ref[13] = dcomplex_create(5.131290734194E+02, 5.123075927445E+02);
		csum_ref[14] = dcomplex_create(5.131012720314E+02, 5.123166486553E+02);
		csum_ref[15] = dcomplex_create(5.130760908195E+02, 5.123241541685E+02);
		csum_ref[16] = dcomplex_create(5.130528295923E+02, 5.123304037599E+02);
		csum_ref[17] = dcomplex_create(5.130310107773E+02, 5.123356167976E+02);
		csum_ref[18] = dcomplex_create(5.130103090133E+02, 5.123399592211E+02);
		csum_ref[19] = dcomplex_create(5.129905029333E+02, 5.123435588985E+02);
		csum_ref[20] = dcomplex_create(5.129714421109E+02, 5.123465164008E+02);
	}
	else if (NX == 2048 && NY == 1024 && NZ == 1024 && nt == 25)
	{
		/*
		 * ---------------------------------------------------------------------
		 * class_npb D size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'D';
		csum_ref[1] = dcomplex_create(5.122230065252E+02, 5.118534037109E+02);
		csum_ref[2] = dcomplex_create(5.120463975765E+02, 5.117061181082E+02);
		csum_ref[3] = dcomplex_create(5.119865766760E+02, 5.117096364601E+02);
		csum_ref[4] = dcomplex_create(5.119518799488E+02, 5.117373863950E+02);
		csum_ref[5] = dcomplex_create(5.119269088223E+02, 5.117680347632E+02);
		csum_ref[6] = dcomplex_create(5.119082416858E+02, 5.117967875532E+02);
		csum_ref[7] = dcomplex_create(5.118943814638E+02, 5.118225281841E+02);
		csum_ref[8] = dcomplex_create(5.118842385057E+02, 5.118451629348E+02);
		csum_ref[9] = dcomplex_create(5.118769435632E+02, 5.118649119387E+02);
		csum_ref[10] = dcomplex_create(5.118718203448E+02, 5.118820803844E+02);
		csum_ref[11] = dcomplex_create(5.118683569061E+02, 5.118969781011E+02);
		csum_ref[12] = dcomplex_create(5.118661708593E+02, 5.119098918835E+02);
		csum_ref[13] = dcomplex_create(5.118649768950E+02, 5.119210777066E+02);
		csum_ref[14] = dcomplex_create(5.118645605626E+02, 5.119307604484E+02);
		csum_ref[15] = dcomplex_create(5.118647586618E+02, 5.119391362671E+02);
		csum_ref[16] = dcomplex_create(5.118654451572E+02, 5.119463757241E+02);
		csum_ref[17] = dcomplex_create(5.118665212451E+02, 5.119526269238E+02);
		csum_ref[18] = dcomplex_create(5.118679083821E+02, 5.119580184108E+02);
		csum_ref[19] = dcomplex_create(5.118695433664E+02, 5.119626617538E+02);
		csum_ref[20] = dcomplex_create(5.118713748264E+02, 5.119666538138E+02);
		csum_ref[21] = dcomplex_create(5.118733606701E+02, 5.119700787219E+02);
		csum_ref[22] = dcomplex_create(5.118754661974E+02, 5.119730095953E+02);
		csum_ref[23] = dcomplex_create(5.118776626738E+02, 5.119755100241E+02);
		csum_ref[24] = dcomplex_create(5.118799262314E+02, 5.119776353561E+02);
		csum_ref[25] = dcomplex_create(5.118822370068E+02, 5.119794338060E+02);
	}
	else if (NX == 4096 && NY == 2048 && NZ == 2048 && nt == 25)
	{
		/*
		 * ---------------------------------------------------------------------
		 * class_npb E size reference checksums
		 * ---------------------------------------------------------------------
		 */
		*class_npb = 'E';
		csum_ref[1] = dcomplex_create(5.121601045346E+02, 5.117395998266E+02);
		csum_ref[2] = dcomplex_create(5.120905403678E+02, 5.118614716182E+02);
		csum_ref[3] = dcomplex_create(5.120623229306E+02, 5.119074203747E+02);
		csum_ref[4] = dcomplex_create(5.120438418997E+02, 5.119345900733E+02);
		csum_ref[5] = dcomplex_create(5.120311521872E+02, 5.119551325550E+02);
		csum_ref[6] = dcomplex_create(5.120226088809E+02, 5.119720179919E+02);
		csum_ref[7] = dcomplex_create(5.120169296534E+02, 5.119861371665E+02);
		csum_ref[8] = dcomplex_create(5.120131225172E+02, 5.119979364402E+02);
		csum_ref[9] = dcomplex_create(5.120104767108E+02, 5.120077674092E+02);
		csum_ref[10] = dcomplex_create(5.120085127969E+02, 5.120159443121E+02);
		csum_ref[11] = dcomplex_create(5.120069224127E+02, 5.120227453670E+02);
		csum_ref[12] = dcomplex_create(5.120055158164E+02, 5.120284096041E+02);
		csum_ref[13] = dcomplex_create(5.120041820159E+02, 5.120331373793E+02);
		csum_ref[14] = dcomplex_create(5.120028605402E+02, 5.120370938679E+02);
		csum_ref[15] = dcomplex_create(5.120015223011E+02, 5.120404138831E+02);
		csum_ref[16] = dcomplex_create(5.120001570022E+02, 5.120432068837E+02);
		csum_ref[17] = dcomplex_create(5.119987650555E+02, 5.120455615860E+02);
		csum_ref[18] = dcomplex_create(5.119973525091E+02, 5.120475499442E+02);
		csum_ref[19] = dcomplex_create(5.119959279472E+02, 5.120492304629E+02);
		csum_ref[20] = dcomplex_create(5.119945006558E+02, 5.120506508902E+02);
		csum_ref[21] = dcomplex_create(5.119930795911E+02, 5.120518503782E+02);
		csum_ref[22] = dcomplex_create(5.119916728462E+02, 5.120528612016E+02);
		csum_ref[23] = dcomplex_create(5.119902874185E+02, 5.120537101195E+02);
		csum_ref[24] = dcomplex_create(5.119889291565E+02, 5.120544194514E+02);
		csum_ref[25] = dcomplex_create(5.119876028049E+02, 5.120550079284E+02);
	}

	if (*class_npb != 'U')
	{
		*verified = TRUE;
		for (i = 1; i <= nt; i++)
		{
			err = dcomplex_abs(dcomplex_div(dcomplex_sub(sums[i], csum_ref[i]),
																			csum_ref[i]));
			if (!(err <= epsilon))
			{
				*verified = FALSE;
				break;
			}
		}
	}

	if (*class_npb != 'U')
	{
		if (*verified)
		{
			printf(" Result verification successful\n");
		}
		else
		{
			printf(" Result verification failed\n");
		}
	}
	printf(" class_npb = %c\n", *class_npb);
}
