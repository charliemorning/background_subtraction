#ifndef _GPU_GAUSS_PRECESS_CU_
#define _GPU_GAUSS_PRECESS_CU_

#include <assert.h>

const int GAUSS_BLOCK_NUM = 32;
const int GAUSS_THREAD_NUM = 256;


//multi-gauss
void cudaGMMMatch(const float* value, float* weight, float* mean, float* stdDev, float* p, bool* match, const int len, const int num, const float alpha);


//to calculate the difference
void cudaDiff(const float* src1, const float* src2, float* dst, const int len);

void cudaGauss(float* result, const float* value, const float* mean, const float* stdDev, const int len);

void cudaUpdateP(float* p, const float* weight, const float* gaussValue, const int len, const float alpha);


//single gauss
void cudaMatch(const float* diff, const float* value, float* mean, float* stdDev, const int len, const float alpha, unsigned char* fg);
void cudaRun(const float* value, float* mean, float* stdDev, const int len, const float alpha, unsigned char* fg);





#endif