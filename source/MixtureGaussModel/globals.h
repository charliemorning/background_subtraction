#ifndef _GLOBALS_H_
#define _GLOBALS_H_

#include <opencv/cv.h>
#include <opencv/highgui.h>

#define _BEGIN_NAMESPACE_DETECT_ namespace detect{

#define _END_NAMESPACE_DETECT_ }

//_BEGIN_NAMESPACE_DETECT_

typedef unsigned int uint;

const float DEFUALT_WEIGHT = 0.025;
const float DEFUALT_STD_DEV = 6.0;
const float DEFUALT_MEAN = 128;

//_END_NAMESPACE_DETECT_
#endif