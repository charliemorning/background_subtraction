#include "gpu_globals_cu.h"
#include "gpu_gaussprocess_cu.h"

#include "gaussprocess.h"

#include "structure.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <ctime>


 //single gauss gpu
 //void main()
 //{
 //	string filename("e:/resource/file2.avi");
 //  
	// //string filename("f:/MVI_2344.MOV");

 //  	cv::VideoCapture cap;
 //  
 //  	cap.open(filename);
 //  
 //  	if(!cap.isOpened())
 //  		return;
 //  	
 //  	cv::Mat frame;
 //	
 //  	
 //  	cap >> frame;
 //	
 //
 //  	int width = frame.size().width;
 //  	int height = frame.size().height;
 //
 //	cv::Mat grayFr(width, height, CV_32FC1);
 //	cv::cvtColor(frame, grayFr, CV_RGB2GRAY);
 //	int size = width * height;
 //
 //	float* mean = (float*)malloc(sizeof(float) * size);
 //	float* stdDev = (float*)malloc(sizeof(float) * size);
 //
 //	init(grayFr.data, mean, stdDev, size);
 //
 //	cv::namedWindow("window");
 //  	cv::namedWindow("original");
 //
 //	IplImage* fg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
 //
 //  	while(1)
 //  	{
 //  		cap >> frame;
 //		if(frame.empty())
 //			break;
 //		cv::cvtColor(frame, grayFr, CV_RGB2GRAY);
 //		
	//	clock_t t1 = clock();
 //		runOnce(grayFr.data, mean, stdDev, size, 0.01, (uchar*)fg->imageData);
	//	clock_t t2 = clock();

	//	printf("CPU TIME: %f\n", (t2 - t1)/1000.0);

 //  		cv::imshow("window", fg);
 //  		cv::imshow("original", frame);
 //  		cv::waitKey(30);
 //  
 //  	}
 //
 //}

//
//void main()
//{
//	string filename("e:/resource/file2.avi");
//
//	cv::VideoCapture cap;
//  
//   	cap.open(filename);
//   
//   	if(!cap.isOpened())
//   		return;
//   	
//   	cv::Mat currentFrame;
// 	
//   	cap >> currentFrame;
// 	
// 
//   	int width = currentFrame.size().width;
//   	int height = currentFrame.size().height;
//
//	int num = 3;
// 	int len = width * height;
// 
//	float* weight = (float*)malloc(sizeof(float) * len * num);
// 	float* mean = (float*)malloc(sizeof(float) * len * num);
// 	float* stdDev = (float*)malloc(sizeof(float) * len * num);
//	float* p = (float*)malloc(sizeof(float) * len * num);
//	bool* match = (bool*)malloc(sizeof(bool) * len * num);
// 
// 	initGMM(currentFrame.data, weight, p, mean, stdDev, len, num);
// 
// 	cv::namedWindow("window");
//   	cv::namedWindow("original");
// 
//	cv::Mat foreground(height, width, CV_8UC1);
// 
//   	while(1)
//   	{
//   		cap >> currentFrame;
// 		if(currentFrame.empty())
// 			break;
//
//		gmmRunOnce(currentFrame.data, weight, p, mean, stdDev, match, stdDev, height * width, num, 0.02, (uchar*)(foreground.data));
//   
//   		cv::imshow("window", foreground);
//   		cv::imshow("original", currentFrame);
//   		cv::waitKey(1000);
//   
//   	}
// 
//
//}

 
 void main()
 {
 	string filename("e:/resource/file2.avi");
 
 	cv::VideoCapture cap;
 
 	cap.open(filename);
 
 	if(!cap.isOpened())
 		return;
 
 	cv::Mat frame;
 
 	cap >> frame;
 
 	int width = frame.size().width;
 	int height = frame.size().height;
 	Gaussian<3> g(width, height);
 
 	cv::Mat foreground(height, width, CV_8UC1);
 
 	g.initiate(frame);
 
 	cv::namedWindow("window");
 	cv::namedWindow("original");
 	while(1)
 	{
 		cap >> frame;
 
 		if (frame.empty())
 		{
 			break;
 		}

		clock_t t1 = clock();

		g.process(frame, foreground);
		clock_t t2 = clock();

		printf("CPU TIME: %f\n", (t2 - t1)/1000.0);


		cv::imshow("window", foreground);
 		cv::imshow("original", frame);
 		cv::waitKey(30);
 
 	}
 
 }
 








