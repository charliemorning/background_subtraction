#include "gmm.h"

void initPara(const float* v, float* w, float* m, float* sd, float* p, const int l, const int n)
{
	int sz = l * n;
	for(int i = 0; i < sz; ++i)
	{
		 w[i] = INIT_W;
	}

	for(int i = 0; i < sz; ++i)
	{
		sd[i] = INIT_SD;
	}

	for(int i = 0; i < sz; ++i)
	{
		m[i] = v[i];
	}

	

}