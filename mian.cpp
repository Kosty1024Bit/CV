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

	char* filename = argc == 2 ? argv[1] : "car.jpg";
	image = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);

	fin = cvCloneImage(image);

	gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	dst = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
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
	Mat ing_1 = imread("car.jpg");
	Mat ing_2 = imread("car.jpg");

	integral(ing_1, ing_2);

	uchar* nfin;
	for (int x = 0; x < image->height; x++) {
		for (int y = 0; y < image->width; y++) {

			int k = 1;
			int p_dst = (int)out_arr.at<float>(x, y);
			if (p_dst % 2 == 0)p_dst--;
			int pls = (int)p_dst / (int)2;

			int dedx = p_dst * k;
			int dedy = dedx * 3;

			int ing_x = x + pls;
			
			int ing_y = (y + pls) * 3;

			int sum_b = 0;
			int sum_g = 0;
			int sum_r = 0;

			nfin = (uchar*)(fin->imageData + x * fin->widthStep);

			if ((ing_x - dedx > 0 && ing_y - dedy > 0) && (ing_x < image->height && ing_y < image->width)) {

				sum_b = ing_2.at<int>(ing_x, ing_y + 0) - ing_2.at<int>(ing_x - dedx, ing_y + 0) - ing_2.at<int>(ing_x, ing_y - dedy + 0) + ing_2.at<int>(ing_x - dedx, ing_y - dedy + 0);
				sum_g = ing_2.at<int>(ing_x, ing_y + 1) - ing_2.at<int>(ing_x - dedx, ing_y + 1) - ing_2.at<int>(ing_x, ing_y - dedy + 1) + ing_2.at<int>(ing_x - dedx, ing_y - dedy + 1);
				sum_r = ing_2.at<int>(ing_x, ing_y + 2) - ing_2.at<int>(ing_x - dedx, ing_y + 2) - ing_2.at<int>(ing_x, ing_y - dedy + 2) + ing_2.at<int>(ing_x - dedx, ing_y - dedy + 2);
			
				nfin[3 * y + 0] = sum_b / (dedx * dedx);
				nfin[3 * y + 1] = sum_g / (dedx * dedx);
				nfin[3 * y + 2] = sum_r / (dedx * dedx);
			}
			else
			{
				nfin[3 * y + 0] = ing_1.at<Vec3b>(x, y)[0];
				nfin[3 * y + 1] = ing_1.at<Vec3b>(x, y)[1];
				nfin[3 * y + 2] = ing_1.at<Vec3b>(x, y)[2];
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

	
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE), cvShowImage("original", image);
	cvNamedWindow("gray", CV_WINDOW_AUTOSIZE), cvShowImage("gray", gray);
	cvNamedWindow("cvCanny", CV_WINDOW_AUTOSIZE), cvShowImage("cvCanny", dst);
	cvNamedWindow("sobel_x", CV_WINDOW_AUTOSIZE), cvShowImage("sobel_x", sobel_x);
	cvNamedWindow("sobel_y", CV_WINDOW_AUTOSIZE), cvShowImage("sobel_y", sobel_y);
	cvNamedWindow("sobel", CV_WINDOW_AUTOSIZE), cvShowImage("sobel", sobel);
	cvNamedWindow("fin", CV_WINDOW_AUTOSIZE), cvShowImage("fin", fin);
	cvNamedWindow("sv", CV_WINDOW_AUTOSIZE), cvShowImage("sv", sv);
	
	cvWaitKey(0);
	cvReleaseImage(&image);
	cvReleaseImage(&gray);
	cvReleaseImage(&dst);
	cvDestroyAllWindows();
	return 0;
}
