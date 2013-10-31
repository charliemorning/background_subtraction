#ifndef _GMM_H_
#define _GMM_H_

const float INIT_SD = 6.0f;
const float INIT_W = 0.33;

void initPara(const float* v, float* w, float* m, float* sd, float* p, const int l, const int n);

#endif