#include "gpu_gaussprocess_cu.h"

#include <cutil_inline.h>
#include <iostream>
#include <ctime>



__global__ static void cuda_gmm_calcDiff_impl(float* diff, const float* value, const float* mean, const int* size)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int i = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(i < *size)
	{
		diff[i] = abs(value[i] - mean[i]);
	}
}

__global__ static void cuda_gmm_updatePara_impl(const float* value, const float* diff, float* weight, float* mean, float* stdDev, float*p)
{
}



__global__ static void cuda_gmm_match_impl(const float* value, float* diff, float* weight, float* mean, float* stdDev, float* p, bool* match, const int *len, const int* num, const float *alpha, const float* compAlpha)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int i = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(i < *len * *num)
	{
		diff[i] = abs(value[i] - mean[i]);
		
		if(diff[i] <= 4 * stdDev[i])
		{
			match[i] = true;
			
			weight[i] = *compAlpha * weight[i] + *alpha;

			p[i] = 1 / (pow(2 * 3.141, 1.5) * stdDev[i] ) * exp( -0.5 * pow((double)(value[i] - stdDev[i]), 2.0) / pow((double)stdDev[i], 2.0));

			mean[i] = (1.0 - p[i]) * mean[i] + p[i] * value[i];

			stdDev[i] = sqrt((1.0 - p[i]) * pow(stdDev[i], 2 ) + p[i] * pow(value[i] - mean[i], 2));
		}
		else
		{
			weight[i] *= *compAlpha;
		}
	}
	__syncthreads();
}




__global__ static void cuda_gmm_normalize_impl(float* weight, float* weightSum, const int *len, const int* n)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int i = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(i < *len)
	{
		weightSum[i] = 0.0f;
		for(int j = 0; j < *n; ++j)
		{
			weightSum[i] += weight[i * *n + j];
		}
	}

	__syncthreads();

	if(i < *len)
	{
		for(int j = 0; j < *n; ++j)
		{
			weight[i + j] /= weightSum[i];
		}
	}

}

__global__ static void cuda_gmm_calcCmpFactor_impl(const float* weight, const float* stdDev, float* cmpFactor, const int* len, const int* num)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int i = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(i < *len * *num)
	{
		cmpFactor[i] = weight[i] / stdDev[i];
	}
}




void cudaGMMMatch(const float* value, float* weight, float* mean, float* stdDev, float* p,
				  bool* match, const int len, const int num, const float alpha)
{
	assert(value || mean || stdDev || p || match);

	dim3 blockD(512, 512);
	dim3 threadD(16, 32);


	float compAlpha = 1.0f - alpha;
	
	float* d_value = NULL;
	float* d_diff = NULL;
	float* d_weight = NULL;
	float* d_mean = NULL;
	float* d_stdDev = NULL;
	float* d_p = NULL;
	bool* d_match = NULL;
	int* d_len = NULL;
	int* d_num = NULL;
	float* d_alpha = NULL;
	float* d_compAlpha = NULL;

	float* d_weightSum = NULL;

	for(int i = 0; i < len * num; ++i)
	{
		match[i] = false;
	}

	//to allocate memory on device
	cutilSafeCall(cudaMalloc((void**)&d_value, sizeof(float) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_diff, sizeof(float) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_weight, sizeof(float) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_mean, sizeof(float) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_stdDev, sizeof(float) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_p, sizeof(float) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_match, sizeof(bool) * len * num));
	cutilSafeCall(cudaMalloc((void**)&d_len, sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_num, sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_alpha, sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&d_compAlpha, sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&d_weightSum, sizeof(float) * len));

	//copy data from host to device
	cutilSafeCall(cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_num, &num, sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_compAlpha, &compAlpha, sizeof(float), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_match, match, sizeof(bool) * len * num, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_value, value, sizeof(float) * len * num, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_weight, weight, sizeof(float) * len * num, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_mean, mean, sizeof(float) * len * num, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_stdDev, stdDev, sizeof(float) * len * num, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_p, p, sizeof(float) * len * num, cudaMemcpyHostToDevice));

	cuda_gmm_match_impl<<<blockD, threadD, 0>>>(d_value, d_diff, d_weight, d_mean, d_stdDev, d_p, d_match, d_len, d_num, d_alpha, d_compAlpha);

	//cuda_gmm_normalize_impl<<<blockD, threadD, 0>>>(d_weight, d_weightSum, d_len, d_num);
	
	cutilSafeCall(cudaMemcpy(match, d_match, sizeof(bool) * len * num, cudaMemcpyDeviceToHost));

	//copy back
	cutilSafeCall(cudaMemcpy(weight, d_weight, sizeof(float) * len * num, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(mean, d_mean, sizeof(float) * len * num, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(stdDev, d_stdDev, sizeof(float) * len * num, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(p, d_p, sizeof(float) * len * num, cudaMemcpyDeviceToHost));

		
	//free 
	cutilSafeCall(cudaFree(d_value));
	cutilSafeCall(cudaFree(d_diff));
	cutilSafeCall(cudaFree(d_weight));
	cutilSafeCall(cudaFree(d_mean));
	cutilSafeCall(cudaFree(d_stdDev));
	cutilSafeCall(cudaFree(d_p));
	cutilSafeCall(cudaFree(d_match));
	cutilSafeCall(cudaFree(d_len));
	cutilSafeCall(cudaFree(d_num));
	cutilSafeCall(cudaFree(d_alpha));
	cutilSafeCall(cudaFree(d_compAlpha));

	cutilSafeCall(cudaFree(d_weightSum));

	d_value = NULL;
	d_weight = NULL;
	d_mean = NULL;
	d_stdDev = NULL;
	d_p = NULL;
	d_match = NULL;
	d_len = NULL;
	d_num = NULL;
	d_alpha = NULL;
	d_compAlpha = NULL;
	d_weightSum = NULL;

}



/*
**to calculate the gauss value in device
*/
__global__ void cuda_gauss_impl(float* result, const float* value, const float* mean, float* stdDev, const int* len)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int index = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(index < *len)
		result[index] = 1.0f / (pow(2.0f * 3.141f, 1.5f) * stdDev[index] ) *
		exp( -0.5f * pow((double)(value[index] - stdDev[index]), 2.0) / pow((double)stdDev[index], 2.0));

	__syncthreads();
}

void cudaGauss(float* result, const float* value, const float* mean, const float* stdDev, const int len)
{
	assert(mean && value);

	dim3 blockD(512, 512);
	dim3 threadD(16, 32);

	float* d_result = NULL;
	float* d_value = NULL;
	float* d_stdDev = NULL;
	float* d_mean = NULL;
	int* d_len;

	unsigned int timer = 0;

	cutilSafeCall(cudaMalloc((void**)&d_result, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_value, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_stdDev, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_mean, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_len, sizeof(int)));

	cutilSafeCall(cudaMemcpy(d_value, value, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_stdDev, stdDev, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_mean, mean, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice));

	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));

	cuda_gauss_impl<<<blockD, threadD, 0>>>(d_result, d_value, d_mean, d_stdDev, d_len);
	cutilCheckError(cutStopTimer(timer));

	//std::cout << "gpu time: " << cutGetTimerValue(timer) <<  std::endl;

	cutilSafeCall(cudaMemcpy(result, d_result, sizeof(float) * len, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaFree(d_result));
	cutilSafeCall(cudaFree(d_stdDev));
	cutilSafeCall(cudaFree(d_mean));
	cutilSafeCall(cudaFree(d_value));
	cutilSafeCall(cudaFree(d_len));

	d_result = NULL;
	d_stdDev = NULL;
	d_mean = NULL;
	d_value = NULL;
	d_len = NULL;

}

__global__ void cuda_diff_impl(const float* src1, const float* src2, float* dst, const int* len)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int index = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(index < *len)
		dst[index] = src1[index] - src2[index];
}

void cudaDiff(const float* src1, const float* src2, float* dst, const int len)
{
	assert(src1 && src2 && dst);

	dim3 blockD(512, 512);
	dim3 threadD(16, 32);

	float* d_src1 = NULL;
	float* d_src2 = NULL;
	float* d_dst = NULL;
	int* d_len;

	unsigned int timer = 0;

	cutilSafeCall(cudaMalloc((void**)&d_src1, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_src2, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_dst, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_len, sizeof(int)));

	cutilSafeCall(cudaMemcpy(d_src1, src1, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_src2, src2, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice));

	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));

	cuda_diff_impl<<<blockD, threadD, 0>>>(d_src1, d_src2, d_dst, d_len);
	cutilCheckError(cutStopTimer(timer));

	std::cout << "gpu time: " << cutGetTimerValue(timer) <<  std::endl;

	cutilSafeCall(cudaMemcpy(dst, d_dst, sizeof(float) * len, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaFree(d_src1));
	cutilSafeCall(cudaFree(d_src2));
	cutilSafeCall(cudaFree(d_dst));
	cutilSafeCall(cudaFree(d_len));

	d_src1 = NULL;
	d_src2 = NULL;
	d_dst = NULL;

}


__global__ void cuda_updateP_impl(float* p, const float* weight, const float* gaussValue, const int* len, const float* alpha)
{
	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int index = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;

	if(index < *len)
	{
		p[index] = *alpha * weight[index] * gaussValue[index];
	}
}


void cudaUpdateP(float* p, const float* weight, const float* gaussValue, const int len, const float alpha)
{

	assert(weight && gaussValue);

	dim3 blockD(512, 512);
	dim3 threadD(16, 32);

	float* d_p = NULL;
	float* d_weight = NULL;
	float* d_gauss = NULL;
	int* d_len = NULL;
	float* d_alpha = NULL;

	unsigned int timer = 0;

	cutilSafeCall(cudaMalloc((void**)&d_p, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_weight, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_gauss, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_len, sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_alpha, sizeof(float)));

	cutilSafeCall(cudaMemcpy(d_weight, weight, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_gauss, gaussValue, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice));
	

	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));

	cuda_updateP_impl<<<blockD, threadD, 0>>>(d_p, d_weight, d_gauss, d_len, d_alpha);
	cutilCheckError(cutStopTimer(timer));

	std::cout << "gpu time: " << cutGetTimerValue(timer) <<  std::endl;

	cutilSafeCall(cudaMemcpy(p, d_p, sizeof(float) * len, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaFree(d_p));
	cutilSafeCall(cudaFree(d_weight));
	cutilSafeCall(cudaFree(d_gauss));
	cutilSafeCall(cudaFree(d_len));
	cutilSafeCall(cudaFree(d_alpha));


	d_p = NULL;
	d_weight = NULL;
	d_gauss = NULL;
	d_len = NULL;
	d_alpha = NULL;
}




__global__ void cuda_match_impl(const float* value, float* mean,
								float* stdDev, const int* len, const float* alpha, unsigned char* fg)
{

	int blx = blockIdx.x;
	int bly = blockIdx.y;
	int thx = threadIdx.x;
	int thy = threadIdx.y;

	int i = (bly * blockDim.y + thy) * blockDim.x * gridDim.x + blx * blockDim.x + thx;


	
	if(i < *len)
	{
		if(abs(mean[i] - value[i]) < 3.0f * stdDev[i])
		{
			
			mean[i] = (1.0f - *alpha) * mean[i] + *alpha * value[i];
			stdDev[i] = sqrt((1.0f - *alpha) * stdDev[i] * stdDev[i] + *alpha * (value[i] - mean[i]) * (value[i] - mean[i]));
			fg[i] = 0;
		}
		else
		{
			fg[i] = 255;
			mean[i] = value[i];
			stdDev[i] = 8;
		}
	}
	__syncthreads();
}


void cudaMatch(const float* value, float* mean, float* stdDev, const int len, const float alpha, unsigned char* fg)
{
	float* d_value = NULL;
	float* d_mean = NULL;
	float* d_stdDev = NULL;
	int* d_len = NULL;
	float* d_alpha = NULL;
	unsigned char* d_fg = NULL;

	unsigned int timer = 0;

	dim3 blockD(352, 288);
	dim3 threadD(32, 16);

	cutilSafeCall(cudaMalloc((void**)&d_value, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_mean, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_stdDev, sizeof(float) * len));
	cutilSafeCall(cudaMalloc((void**)&d_len, sizeof(int)));
	cutilSafeCall(cudaMalloc((void**)&d_alpha, sizeof(float)));
	cutilSafeCall(cudaMalloc((void**)&d_fg, sizeof(unsigned char) * len));

	cutilSafeCall(cudaMemcpy(d_value, value, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_mean, mean, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_stdDev, stdDev, sizeof(float) * len, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_len, &len, sizeof(int), cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice));


	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutStartTimer(timer));
	cuda_match_impl<<<blockD, threadD, 0>>>(d_value, d_mean, d_stdDev, d_len, d_alpha, d_fg);

	cudaThreadSynchronize();
	cutilCheckError(cutStopTimer(timer));

	//std::cout << "gpu time: " << cutGetTimerValue(timer) <<  std::endl;

	cutilSafeCall(cudaMemcpy(mean, d_mean, sizeof(float) * len, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(stdDev, d_stdDev, sizeof(float) * len, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(fg, d_fg, sizeof(unsigned char) * len, cudaMemcpyDeviceToHost));

	cutilSafeCall(cudaFree(d_value));
	cutilSafeCall(cudaFree(d_mean));
	cutilSafeCall(cudaFree(d_stdDev));
	cutilSafeCall(cudaFree(d_len));
	cutilSafeCall(cudaFree(d_alpha));
	cutilSafeCall(cudaFree(d_fg));

	d_value = NULL;
	d_mean = NULL;
	d_stdDev = NULL;
	d_len = NULL;
	d_alpha = NULL;
	d_fg = NULL;

}

void cudaRun(const float* value, float* mean, float* stdDev, const int len, const float alpha, unsigned char* fg)
{
	cudaMatch(value, mean, stdDev, len, alpha, fg);
}