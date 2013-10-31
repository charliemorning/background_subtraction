#ifndef _GAUSS_PROCESS_H_
#define _GAUSS_PROCESS_H_

#include "gpu_gaussprocess_cu.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>



void init(uchar* data, float* mean, float* stdDev, const int len);


void runOnce(uchar* data, float* mean, float* stdDev, const int len, const float alpha, unsigned char* fg);


void initGMM(uchar* data, float* weight, float* p, float* mean, float* stdDev, const int len, const int num);


void gmmRunOnce(uchar* data, float* weight, float* p, float* mean, float* stdDev, bool* match, float* cmpFactor, const int len, const int num, const float alpha, uchar* fg);



#endif