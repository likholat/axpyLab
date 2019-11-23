#pragma once
#include <time.h>
#include <chrono>

 size_t myGroupSize = 256;

const char *saxpyKer =
"__kernel void saxpy(const size_t n, const float a, __global const float *x, const size_t incx, __global float *y, const size_t incy) { \n"
"  int i = get_global_id(0);                                \n"
"  if (i * incy < n && i * incx < n)                                \n"
"        y[i * incy] = y[i * incy] + a * x[i * incx];               \n"
"}";

const char *daxpyKer =
"__kernel void daxpy(const size_t n, const double a, __global const double *x, const size_t incx, __global double *y, const size_t incy) { \n"
"  int i = get_global_id(0);                                \n"
"  if (i * incy < n && i * incx < n)                                \n"
"        y[i * incy] = y[i * incy] + a * x[i * incx];               \n"
"}";

template <typename FOrDType>
clock_t cpu_axpy(const size_t n, const FOrDType a, const FOrDType* x, const size_t incx, FOrDType* y, const size_t incy) {
	clock_t time = clock();
	for (size_t i = 0; i * incy < n && i * incx < n; ++i)
		y[i * incy] = y[i * incy] + a * x[i * incx];
	time = clock() - time;
	return time;
}

template <typename FOrDType>
const char* chooseKernel() {
	if ((sizeof FOrDType) == sizeof(float))
		return saxpyKer;
	else return daxpyKer;
}

template <typename FOrDType>
void initializeKernel(cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
	cl_device_id& device, cl_program& program, cl_int& error, cl_device_type deviceType) {

	cl_uint platformsCount = 0;
	clGetPlatformIDs(0, nullptr, &platformsCount);

	cl_platform_id* platforms = new cl_platform_id[platformsCount];
	clGetPlatformIDs(platformsCount, platforms, nullptr);

	cl_platform_id platform = nullptr;
	cl_device_id cpu_device = nullptr;
	cl_device_id gpu_device = nullptr;

	for (cl_uint i = 0; i < platformsCount; ++i) {
		char platformName[128];
		clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
			128, platformName, nullptr);
		std::cout << platformName << std::endl;

		cl_uint devicesNum = 0;
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, 0, &devicesNum);
		cl_device_id* deviceIDs = new cl_device_id[devicesNum];
		clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devicesNum, deviceIDs, 0);

		for (cl_uint j = 0; j < devicesNum; ++j) {
			char deviceName[128];
			clGetDeviceInfo(deviceIDs[j], CL_DEVICE_NAME, 128, deviceName, nullptr);

			cl_device_type type = CL_DEVICE_TYPE_ALL;
			clGetDeviceInfo(deviceIDs[j], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
			if (type == CL_DEVICE_TYPE_GPU)
				gpu_device = deviceIDs[j];
			if (type == CL_DEVICE_TYPE_CPU)
				cpu_device = deviceIDs[j];
			std::cout << "device: " << deviceName << std::endl;
		}
	}

	platform = platforms[0];

	if (deviceType == CL_DEVICE_TYPE_GPU)
		device = gpu_device;
	else if (deviceType == CL_DEVICE_TYPE_CPU)
		device = cpu_device;

	context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &error);

	//cl_context_properties properties[3] = {
	//	CL_CONTEXT_PLATFORM,
	//	(cl_context_properties)platform,
	//	0
	//};

	//context = clCreateContextFromType(
	//	(platform == nullptr) ? nullptr : properties,
	//	deviceType, 0, 0, &error);

	////р-р массива для хранения списка устройств
	//size_t deviceCount = 0;
	//clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &deviceCount);

	//cl_device_id* devices = new cl_device_id[deviceCount];
	//clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceCount, devices, 0);
	//	
	////устройство для вычислений
	//if (deviceType == CL_DEVICE_TYPE_GPU)
	//	device = devices[0];
	//else if (deviceType == CL_DEVICE_TYPE_CPU)
	//	device = devices[1];
	//

	//char deviceName[128];
	//clGetDeviceInfo(device, CL_DEVICE_NAME, 128, deviceName, nullptr);
	//std::cout << "device: " << deviceName << std::endl;

	queue = clCreateCommandQueueWithProperties(context, device, 0, &error);

	const char* kernelSource;
	kernelSource = chooseKernel<FOrDType>();
	size_t kernelLen[] = { strlen(kernelSource) };

	program = clCreateProgramWithSource(context, 1, &kernelSource, kernelLen, &error);

	clBuildProgram(program, 1, &device, 0, 0, 0);

	size_t length = 10000;
	char* buffer = new char[length];
	clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, buffer, NULL);
	printf(buffer);

	if (sizeof(FOrDType) == sizeof(float))
		kernel = clCreateKernel(program, "saxpy", &error);
	else if (sizeof(FOrDType) == sizeof(double))
		kernel = clCreateKernel(program, "daxpy", &error);

}

template <typename FOrDType>
void setKernelArguments(const size_t n, const FOrDType a, const FOrDType* x, const size_t incx, FOrDType* y,
	const size_t incy, cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
	cl_device_id& device, cl_int& error, cl_mem& xBuffer, cl_mem& yBuffer, size_t& groupSize) {

	clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &groupSize, 0);
	groupSize = myGroupSize;
	
	error = clSetKernelArg(kernel, 0, sizeof(size_t), &n);
	error = clSetKernelArg(kernel, 1, sizeof(FOrDType), &a);

	size_t biteSize = sizeof(FOrDType) * (n / groupSize + !!(n % groupSize)) * groupSize;

	xBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, biteSize, 0, &error);
	error = clEnqueueWriteBuffer(queue, xBuffer, CL_TRUE, 0, biteSize, x, 0, 0, 0);
	error = clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuffer);
	error = clSetKernelArg(kernel, 3, sizeof(size_t), &incx);

	yBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, biteSize, 0, &error);
	error = clEnqueueWriteBuffer(queue, yBuffer, CL_TRUE, 0, biteSize, y, 0, 0, 0);
	error = clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuffer);
	error = clSetKernelArg(kernel, 5, sizeof(size_t), &incy);
}

template <typename FOrDType>
clock_t opencl_axpy(const size_t n, const FOrDType a, const FOrDType* x, const size_t incx, FOrDType* y, const size_t incy, cl_device_type deviceType) {
	
	cl_context context;
	cl_command_queue queue;
	cl_kernel kernel;
	cl_device_id device;
	cl_program program;
	cl_mem xBuffer, yBuffer;
	cl_int error = 0;
	size_t groupSize = 0;

	initializeKernel<FOrDType>(kernel, context, queue, device, program, error, deviceType);
	setKernelArguments(n, a, x, incx, y, incy, kernel, context, queue, device, error, xBuffer, yBuffer, groupSize);

	size_t itemsNum = (n / groupSize + !!(n % groupSize)) * groupSize;
	cl_event event;
	clock_t time = clock();
	//выполнение ядра над множеством входных данных
	clEnqueueNDRangeKernel(queue, kernel, 1, 0, &itemsNum, &groupSize, 0, 0, &event);
	clWaitForEvents(1, &event);
	time = clock() - time;
	//загрузка результатов вычислений с устройства
	clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, sizeof(FOrDType) * n, y, 0, 0, 0);

	clReleaseMemObject(xBuffer);
	clReleaseMemObject(yBuffer);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);

	return time;
}

template <typename FOrDType>
clock_t omp_axpy(const size_t n, const FOrDType a, const FOrDType *x, const size_t incx, FOrDType *y, const size_t incy) {
	int i;
	clock_t time = clock();

#pragma omp parallel for shared (y, x ,a, n, incx, incy) private(i)
	for (i = 0; i < n; i++)
		if (i * incy < n && i * incx < n)
			y[i * incy] = y[i * incy] + a * x[i * incx];

	time = clock() - time;

	return time;
}
