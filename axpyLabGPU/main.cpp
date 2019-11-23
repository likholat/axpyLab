#include <CL/cl.h>
#include <iostream>
#include <vector>
#include "Source.h"

using namespace std;


typedef float FOrDType;

int main() {
	const size_t n = pow(2,10), incx = 1, incy = 1;
	const FOrDType a = (FOrDType)(1);
	FOrDType *x = new FOrDType[n], *y = new FOrDType[n];

	for (size_t i = 0; i < n; i++) {
		x[i] = (FOrDType)(1);
		y[i] = (FOrDType)(2);
	}

	cout << "y = a*x + y\na: " << a << "\nx:";
	for (size_t i = 0; i < 15; ++i)
		cout << " " << x[i];
	cout << "\ny:";
	for (size_t i = 0; i < 15; ++i)
		cout << " " << y[i];
	cout << endl;

	// CPU
	clock_t cpuTime = cpu_axpy(n, a, x, incx, y, incy);
	cout << "CPU res:";
	for (size_t i = 0; i < 15; i++)
		cout << " " << y[i];
	std::cout << endl;
	cout << "CPU time: " << (float)cpuTime / CLOCKS_PER_SEC;
	cout << endl;

	//OpenCL GPU
	for (size_t i = 0; i < n; i++) {
		y[i] = (FOrDType)(2);
	}

	clock_t gpuTime = opencl_axpy(n, a, x, incx, y, incy, CL_DEVICE_TYPE_GPU);
	std::cout << "OCL GPU res:";
	for (size_t i = 0; i < 15; ++i)
		cout << " " << y[i];
	cout << endl;
	cout << "OCL GPU time: " << (float)gpuTime / CLOCKS_PER_SEC;
	cout << endl;

	//OpenCL CPU
	for (size_t i = 0; i < n; i++) {
		y[i] = (FOrDType)(2);
	}

	clock_t cpuOCLTime = opencl_axpy(n, a, x, incx, y, incy, CL_DEVICE_TYPE_CPU);
	std::cout << "OCL CPU res:";
	for (size_t i = 0; i < 15; ++i)
		cout << " " << y[i];
	cout << endl;
	cout << "OCL CPU time: " << (float)cpuOCLTime / CLOCKS_PER_SEC;
	cout << endl;
	

	// OpenMP
	for (size_t i = 0; i < n; ++i)
		y[i] = static_cast<FOrDType>(2);

	clock_t ompTime = omp_axpy(n, a, x, incx, y, incy);
	cout << "OMP res:";
	for (size_t i = 0; i < 15; ++i)
		cout << " " << y[i];
	cout << endl;
	cout << "OMP time: " << (float)ompTime / CLOCKS_PER_SEC;
	cout << endl;

}