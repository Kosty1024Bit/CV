#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/gpu/gpu.hpp"

#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

#define usi unsigned short int
#define si short int

IplImage* image = 0;
IplImage* gray = 0;
IplImage* sobel_x = 0;
IplImage* sobel_y = 0;
IplImage* sobel = 0;
IplImage* dst = 0;
IplImage* fin = 0;
IplImage* sv = 0;

int main(int argc, char* argv[]){
	// èìÿ êàðòèíêè çàäà¸òñÿ ïåðâûì ïàðàìåòðîì
	char* filename = argc == 2 ? argv[1] : "car.jpg";
	// ïîëó÷àåì êàðòèíêó
	image = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);

	fin = cvCloneImage(image);

	// ñîçäà¸ì îäíîêàíàëüíûå êàðòèíêè
	gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	dst = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	// ïðåîáðàçóåì â ãðàäàöèè ñåðîãî
	cvCvtColor(image, gray, CV_RGB2GRAY);

	sobel = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);
	sobel_x = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);
	sobel_y = cvCreateImage(cvGetSize(gray), IPL_DEPTH_8U, 1);

	cvSobel(gray, sobel_x, 1, 0, 3);
	cvSobel(gray, sobel_y, 0, 1, 3);

	uchar* ptr_sob;
	uchar* ptr_sob_x;
	uchar* ptr_sob_y;
	for (int x = 0; x < sobel->height; x++) {
		ptr_sob = (uchar*)(sobel->imageData + x * sobel->widthStep);
		ptr_sob_x = (uchar*)(sobel_x->imageData + x * sobel_x->widthStep);
		ptr_sob_y = (uchar*)(sobel_y->imageData + x * sobel_y->widthStep);
		for (int y = 0; y < fin->width; y++) {
			ptr_sob[y] = (ptr_sob_x[y] + ptr_sob_y[y]) / 2;
		}
	}

	// ïîëó÷àåì ãðàíèöû
	cvCanny(sobel, dst, 10, 100, 3);

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//
	float k = 1; // 0.0 - 1.0
	Mat in_arr = cvarrToMat(dst);
	Mat out_arr = cvarrToMat(dst);

	for (int y = 0; y < dst->height; y++){
		uchar* ptr = (uchar*)(dst->imageData + y * dst->widthStep);
		for (int x = 0; x < dst->width; x++){
			if (ptr[x] == 255)ptr[x] = 0;
			else ptr[x] = 255;}}

	distanceTransform(in_arr, out_arr, CV_DIST_L1, CV_DIST_MASK_3);

	for (int x = 0; x < out_arr.rows; x++) {
		for (int y = 0; y < out_arr.cols; y++) {
			out_arr.at<float>(x, y) *= k;
		}
	}
	uchar* ptr_fin = 0;
	uchar* ptr = 0;

	for (int x = 20; x < fin->height - 20; x++) {
		for (int y = 20; y < fin->width - 20; y++) {
			//ptr_im[3 * x + 0] image blue
			//ptr_im[3 * x + 1] image green
			//ptr_im[3 * x + 2] image red
			//out_arr.at<float>(x, y) distance
			float sum_blue = 0;
			float sum_green = 0;
			float sum_red = 0;
			int p_dst = (int)out_arr.at<float>(x, y);

			if (p_dst % 2 == 0)p_dst--;
			p_dst--;

			for (int i = -p_dst / 2; i < p_dst / 2; i++) {
				for (int j = -p_dst / 2; j < p_dst / 2; j++) {
					
					ptr_fin = (uchar*)(fin->imageData + (x + j) * fin->widthStep);

					sum_blue  += ptr_fin[3 * (y + i) + 0];
					sum_green += ptr_fin[3 * (y + i) + 1];
					sum_red   += ptr_fin[3 * (y + i) + 2];
				}
			}
		
			sum_blue  /= (p_dst * p_dst);
			sum_red   /= (p_dst * p_dst);
			sum_green /= (p_dst * p_dst);

			ptr_fin = (uchar*)(fin->imageData + x * fin->widthStep);

			if (sum_blue > 255) sum_blue = 255;
			if (sum_blue < 0)   sum_blue = 0;
			if (sum_green > 255) sum_green = 255;
			if (sum_green < 0)	 sum_green = 0;
			if (sum_red > 255) sum_red = 255;
			if (sum_red < 0)   sum_red = 0;

			ptr_fin[3 * y + 0] = sum_blue;
			ptr_fin[3 * y + 1] = sum_green;
			ptr_fin[3 * y + 2] = sum_red;

			if (ptr_fin[3 * y + 0] == 0 && ptr_fin[3 * y + 1] == 0 && ptr_fin[3 * y + 2] == 0) {
				ptr = (uchar*)(image->imageData + x * image->widthStep);
				ptr_fin[3 * y + 0] = ptr[3 * y + 0];
				ptr_fin[3 * y + 1] = ptr[3 * y + 1];
				ptr_fin[3 * y + 2] = ptr[3 * y + 2];
			}
		}
	}

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//
	sv = cvCloneImage(fin);

	float mat[3][3];
	mat[0][0] = -0.1, mat[1][0] = -0.1, mat[2][0] = -0.1;
	mat[0][1] = -0.1, mat[1][1] =	 2, mat[2][1] = -0.1;
	mat[0][2] = -0.1, mat[1][2] = -0.1, mat[2][2] = -0.1;

	float sum_r = 0, sum_g = 0, sum_b = 0;

	uchar* ptr_sv = 0;
	uchar* ptr_im = 0;

	for (int x = 1; x < sv->height - 1; x++) {
		ptr_sv = (uchar*)(sv->imageData + x * sv->widthStep);
		for (int y = 1; y < sv->width - 1; y++) {
			
			for (int i = -1; i < 2; i++) {
				for (int j = -1; j < 2; j++) {

					ptr_im = (uchar*)(image->imageData + (x + i) * image->widthStep);

					sum_b += mat[i + 1][j + 1] * ptr_im[3 * (y + j) + 0];
					sum_g += mat[i + 1][j + 1] * ptr_im[3 * (y + j) + 1];
					sum_r += mat[i + 1][j + 1] * ptr_im[3 * (y + j) + 2];
				}
			}

			if (sum_b > 255) sum_b = 255;
			if (sum_b < 0)   sum_b = 0;
			if (sum_g > 255) sum_g = 255;
			if (sum_g < 0)	 sum_g = 0;
			if (sum_r > 255) sum_r = 255;
			if (sum_r < 0)   sum_r = 0;

			ptr_sv[3 * y + 0] = sum_b;
			ptr_sv[3 * y + 1] = sum_g;
			ptr_sv[3 * y + 2] = sum_r;
			sum_b = 0, sum_g = 0, sum_r = 0;
		}
	}

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//

	// îêíî äëÿ îòîáðàæåíèÿ êàðòèíêè
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("cvCanny", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("fin", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("sv", CV_WINDOW_AUTOSIZE);

	// ïîêàçûâàåì êàðòèíêè
	cvShowImage("original", image);
	cvShowImage("gray", gray);
	cvShowImage("cvCanny", dst);
	cvShowImage("fin", fin);
	cvShowImage("sv", sv);

	cvNamedWindow("sobel_x", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("sobel_y", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE);
	cvShowImage("sobel_x", sobel_x);
	cvShowImage("sobel_y", sobel_y);
	cvShowImage("sobel", sobel);

	// æä¸ì íàæàòèÿ êëàâèøè
	cvWaitKey(0);

	// îñâîáîæäàåì ðåñóðñû
	cvReleaseImage(&image);
	cvReleaseImage(&gray);
	cvReleaseImage(&dst);
	// óäàëÿåì îêíà
	cvDestroyAllWindows();
	return 0;
}
