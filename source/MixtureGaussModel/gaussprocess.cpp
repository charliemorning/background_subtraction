#include "gaussprocess.h"

void init(uchar* data, float* mean, float* stdDev, const int len)
{
	for (int i = 0; i < len;  ++i)
	{
		mean[i] = (float)data[i];
		stdDev[i] = 6.0f;
	}

	
}

void runOnce(uchar* data, float* mean, float* stdDev, const int len, const float alpha, unsigned char* fg)
{

	float* value = (float*)malloc(sizeof(float) * len);

	for (int i = 0; i < len;  ++i)
	{
		value[i] = (float)data[i];
	}
	cudaRun(value, mean, stdDev, len, alpha, fg);

	free(value);
}

inline int min_index(float* arr, const int len)
{
	int index = 0;
	float min_val = arr[0];
	for (int i = 1; i < len; ++i)
	{
		if (min_val > arr[i])
		{
			min_val = arr[i];
			index = i;
		}
	}

	return index;
}


void initGMM(uchar* data, float* weight, float* p, float* mean, float* stdDev, const int len, const int num)
{
	for (int i = 0; i < len * num; ++i)
	{
		mean[i] = data[i];
		stdDev[i] = 6.0f;
		weight[i] = 0.33;
		p[i] = 0.01;
	}
}



void gmmRunOnce(uchar* data, float* weight, float* p, float* mean, float* stdDev, bool* match, float* cmpFactor, const int len, const int num, const float alpha, uchar* fg)
{
	
	float* dataF = (float*)malloc(sizeof(float) * len * num);

	for (int i = 0; i < len * num;  ++i)
	{
		dataF[i] = (float)data[i];
	}

	cudaGMMMatch(dataF, weight, mean, stdDev, p, match, len, num, alpha);
	
	for (int i = 0; i < num * len; ++i)
	{
		cmpFactor[i] = weight[i] / stdDev[i];
	}

	for (int i = 0; i < len; ++i)
	{

		bool m = false;
		for (int j = 0; j < num; ++j)
		{
			m = match[i * num + j] || m;
		}

		if (!m)
		{
			int j = min_index(cmpFactor + i * num, num);

			int k = i * num + j;

			mean[k] = data[k];

			stdDev[k] = 6.0f;
		}

		if (m)
		{
			fg[i] = 0;
		}
		else
		{
			fg[i] = 255;
		}

	}


	free(dataF);


}