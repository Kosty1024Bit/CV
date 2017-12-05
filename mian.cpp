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
	// имя картинки задаётся первым параметром
	char* filename = argc == 2 ? argv[1] : "car.jpg";
	// получаем картинку
	image = cvLoadImage(filename, CV_LOAD_IMAGE_COLOR);

	fin = cvCloneImage(image);

	// создаём одноканальные картинки
	gray = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);
	dst = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 1);

	// преобразуем в градации серого
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

	// получаем границы
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

	//cout << "@@@@@@@@@@@@@@@@@ FIN @@@@@@@@@@@@@@@@" << endl;
	//for (int y = 0; y < fin->height; y++) {
	//	uchar* ptr_fin = (uchar*)(fin->imageData + y * fin->widthStep);
	//	for (int x = 0; x < fin->width; x++) {
	//		printf("%d %d %d| ", ptr_fin[3 * x + 0], ptr_fin[3 * x + 1], ptr_fin[3 * x + 2]);
	//	}cout << endl;
	//}
	//cout << "@@@@@@@@@@@@@@@@@ FIN @@@@@@@@@@@@@@@@" << endl;

	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@//
	
	//for (int y = 0; y < image->height; y++) {
	//	uchar* ptr = (uchar*)(image->imageData + y * image->widthStep);
	//	for (int x = 0; x < image->width; x++) {
	//		printf("%d %d %d| ", ptr[3 * x + 0], ptr[3 * x + 1], ptr[3 * x + 2]);}cout << endl;}
	//
	//for (int y = 0; y < gray->height; y++) {
	//	uchar* ptr = (uchar*)(gray->imageData + y * gray->widthStep);
	//	for (int x = 0; x < gray->width; x++) {
	//		printf("%d| ", ptr[x]);}cout << endl;}
	//
	//for (int y = 0; y < dst->height; y++) {
	//	uchar* ptr = (uchar*)(dst->imageData + y * dst->widthStep);
	//	for (int x = 0; x < dst->width; x++) {
	//		printf("%d| ", ptr[x]);}cout << endl;}

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

	// окно для отображения картинки
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("gray", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("cvCanny", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("fin", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("sv", CV_WINDOW_AUTOSIZE);

	// показываем картинки
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

	// ждём нажатия клавиши
	cvWaitKey(0);

	// освобождаем ресурсы
	cvReleaseImage(&image);
	cvReleaseImage(&gray);
	cvReleaseImage(&dst);
	// удаляем окна
	cvDestroyAllWindows();
	return 0;
}

void test_img(){
	Mat im = imread("C:/Users/Пухкий Константин/Desktop/Test_OpenCV_001/Test.jpg");
	if (im.empty()) cout << "Cannot load image!" << endl;
	imshow("Test image", im);}

void test_video_cam(){
	VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened()){  // if not success, exit program
		cout << "Cannot open the video cam" << endl;
	}

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;

	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"

	Mat frame;

	while (1){

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess){ //if not success, break loop
			cout << "Cannot read a frame from video stream" << endl;
			getchar();
			break;
		}

		imshow("MyVideo", frame); //show the frame in "MyVideo" window

		if (waitKey(30) == 27){ //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

}

void test_red_obj_trac(){
	VideoCapture cap(0); //capture the video from webcam

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the web cam" << endl;
	}

	namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

	int iLowH = 170;
	int iHighH = 179;

	int iLowS = 150;
	int iHighS = 255;

	int iLowV = 60;
	int iHighV = 255;

	//Create trackbars in "Control" window
	createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
	createTrackbar("HighH", "Control", &iHighH, 179);

	createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
	createTrackbar("HighS", "Control", &iHighS, 255);

	createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
	createTrackbar("HighV", "Control", &iHighV, 255);

	int iLastX = -1;
	int iLastY = -1;

	//Capture a temporary image from the camera
	Mat imgTmp;
	cap.read(imgTmp);

	//Create a black image with the size as the camera output
	Mat imgLines = Mat::zeros(imgTmp.size(), CV_8UC3);;


	while (true)
	{
		Mat imgOriginal;

		bool bSuccess = cap.read(imgOriginal); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		Mat imgHSV;

		cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

		Mat imgThresholded;

		inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

		//morphological opening (removes small objects from the foreground)
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//morphological closing (removes small holes from the foreground)
		dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		//Calculate the moments of the thresholded image
		Moments oMoments = moments(imgThresholded);

		double dM01 = oMoments.m01;
		double dM10 = oMoments.m10;
		double dArea = oMoments.m00;

		// if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
		if (dArea > 10000)
		{
			//calculate the position of the ball
			int posX = dM10 / dArea;
			int posY = dM01 / dArea;

			if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
			{
				//Draw a red line from the previous point to the current point
				line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0, 0, 255), 2);
			}

			iLastX = posX;
			iLastY = posY;
		}

		imshow("Thresholded Image", imgThresholded); //show the thresholded image

		imgOriginal = imgOriginal + imgLines;
		imshow("Original", imgOriginal); //show the original image

		if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
}

void test_img_pol(){
	Mat im_in = imread("C:/Users/Пухкий Константин/Desktop/Test_OpenCV_001/Test_2.jpg");
	if (im_in.empty()) cout << "Cannot load image!" << endl;
	int x = im_in.cols, y = im_in.rows;
	cout << x << " " << y;
	imshow("Test image", im_in);

	Mat im_pol = im_in;
	usi pol = 0;
	for (usi i = 0; i < y; i++){
		for (usi j = 0; j < x; j++){
			pol = (im_pol.at<Vec3b>(i, j)[0]) + (im_pol.at<Vec3b>(i, j)[1]) + (im_pol.at<Vec3b>(i, j)[2]);
			pol /= 3;
			im_pol.at<Vec3b>(i, j)[0] = pol;
			im_pol.at<Vec3b>(i, j)[1] = pol;
			im_pol.at<Vec3b>(i, j)[2] = pol;
		}
	}
	imshow("Test pol image", im_pol);
	waitKey(0);

	Mat im_gr = im_pol;
}

static void sobel_core(si XC[3][3], si YC[3][3]){
	XC[0][0] = -1, XC[0][1] = 0, XC[0][2] = 1;
	XC[1][0] = -2, XC[1][1] = 0, XC[1][2] = 2;
	XC[2][0] = -1, XC[2][1] = 0, XC[2][2] = 1;

	YC[0][0] = 1, YC[0][1] = 2, YC[0][2] = 1;
	YC[1][0] = 0, YC[1][1] = 0, YC[1][2] = 0;
	YC[2][0] = -1, YC[2][1] = -2, YC[2][2] = -1;
}

static void sharr_core(si XC[3][3], si YC[3][3]){
	XC[0][0] = 3, XC[0][1] = 10, XC[0][2] = 3;
	XC[1][0] = 0, XC[1][1] = 0, XC[1][2] = 0;
	XC[2][0] = -3, XC[2][1] = -10, XC[2][2] = -3;

	YC[0][0] = 3, YC[0][1] = 0, YC[0][2] = -3;
	YC[1][0] = 10, YC[1][1] = 0, YC[1][2] = -10;
	YC[2][0] = 3, YC[2][1] = 0, YC[2][2] = -3;

}

/*int main(){

	Mat im = imread("C:/Users/Пухкий Константин/Desktop/Test_OpenCV_001/5x5.jpg");
	if (im.empty()) cout << "Cannot load image!" << endl;
	imshow("5x5", im);

	for (usi i = 1; i <= 3; i++){
		for (si c = 2; c >= 0; c--){
			for (usi j = 1; j <= 3; j++){
				cout << int(im.at<Vec3b>(i, j)[c]) << ' ';}
			cout << endl;}
		cout << endl;}
	
	
	si XC[3][3];
	si YC[3][3];
	sobel_core(XC, YC);
	si sum_x = 0, sum_y = 0;
	si pix = 0, pix_x = 0, pix_y = 0;

	Mat frame;
	im.copyTo(frame);

	for (usi i = 0; i < frame.rows; i++){
		for (usi j = 0; j < frame.cols; j++){
			sum_x = 0, sum_y = 0;
			if (i == 0 || i == frame.rows - 1 || j == 0 || j == frame.cols - 1) pix = 0;
			else{
				for (si x = -1; x < 2; x++){
					for (si y = -1; y < 2; y++){

						pix_x = i + x;
						pix_y = j + y;

						sum_x = sum_x + (im.at<Vec3b>(pix_x, pix_y)[R]) * XC[x + 1][y + 1];
						sum_y = sum_y + (im.at<Vec3b>(pix_x, pix_y)[R]) * YC[x + 1][y + 1];
					}
				}
				pix = sqrt(pow(sum_x, 2) + pow(sum_y, 2));
			}


			//if (pix > 255) pix = 255;
			//if (pix < 0) pix = 0;
			frame.at<Vec3b>(i, j)[R] = pix;
			frame.at<Vec3b>(i, j)[G] = pix;
			frame.at<Vec3b>(i, j)[B] = pix;
		}
	}

	cout << endl << "--------------------------------------------" << endl;
	for (usi i = 1; i <= 3; i++){
		for (si c = 2; c >= 0; c--){
			for (usi j = 1; j <= 3; j++){
				cout << int(im.at<Vec3b>(i, j)[c]) << ' ';
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl << "--------------------------------------------" << endl;
	for (usi i = 1; i <= 3; i++){
		for (si c = 2; c >= 0; c--){
			for (usi j = 1; j <= 3; j++){
				cout << int(frame.at<Vec3b>(i, j)[c]) << ' ';
			}
			cout << endl;
		}
		cout << endl;
	}
	cout << endl << "--------------------------------------------" << endl;

	waitKey(0);
	return 0;
}*/

/*int main(int argc, char** argv){
	//VideoCapture cap(0); // open the video camera no. 0
	//if (!cap.isOpened()){  // if not success, exit program
	//	cout << "Cannot open the video cam" << endl;}
	//
	//double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	//double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	//
	//cout << "Frame size : " << dWidth << " x " << dHeight << endl;

	test_img();


	//namedWindow("Picture", CV_WINDOW_AUTOSIZE); //create a window called "MyVideo"
	//namedWindow("My_video_pol", CV_WINDOW_AUTOSIZE);
	//namedWindow("My_video_sob", CV_WINDOW_AUTOSIZE);

	//Mat frame;
	//Mat sobel;
	//usi pol = 0;
	//si XC[3][3];
	//si YC[3][3];
	//sobel_core(XC, YC);
	////sharr_core(XC, YC);
	//usi pix_x = 0, pix_y = 0;
	//int sum_x = 0, sum_y = 0;
	//int pix = 0;
	//
	//while (1){
	//	bool bSuccess = cap.read(frame); // read a new frame from video
	//	if (!bSuccess){ //if not success, break loop
	//		cout << "Cannot read a frame from video stream" << endl;
	//		getchar();
	//		break;}
	//	imshow("MyVideo", frame); //show the frame in "MyVideo" window
		
		//frame.copyTo(sobel);
		//
		//for (usi i = 0; i < frame.rows; i++){
		//	for (usi j = 0; j < frame.cols; j++){
		//		pol = ((frame.at<Vec3b>(i, j)[R]) + (frame.at<Vec3b>(i, j)[G]) + (frame.at<Vec3b>(i, j)[B])) / 3;
		//		frame.at<Vec3b>(i, j)[R] = pol;
		//		frame.at<Vec3b>(i, j)[G] = pol;
		//		frame.at<Vec3b>(i, j)[B] = pol;}}
		//imshow("My_video_pol", frame);
		//
		//for (usi i = 0; i < frame.rows; i++){
		//	for (usi j = 0; j < frame.cols; j++){
		//		sum_x = 0, sum_y = 0;
		//		if (i == 0 || i == frame.rows - 1 || j == 0 || j == frame.cols - 1) pix = 0;
		//		else{
		//			for (si x = -1; x < 2; x++){
		//				for (si y = -1; y < 2; y++){
		//					pix_x = i + x;
		//					pix_y = j + y;
		//					sum_x += (frame.at<Vec3b>(pix_x, pix_y)[R]) * XC[x + 1][y + 1];
		//					sum_y += (frame.at<Vec3b>(pix_x, pix_y)[R]) * YC[x + 1][y + 1];}}
		//
		//		pix = sqrt(pow(sum_x,2) + pow(sum_y,2));}
		//		if (pix > 255) pix = 255;
		//		sobel.at<Vec3b>(i, j)[R] = pix;
		//		sobel.at<Vec3b>(i, j)[G] = pix;
		//		sobel.at<Vec3b>(i, j)[B] = pix;}}
		//
		//imshow("My_video_sob", sobel);

	//	if (waitKey(30) == 27){ //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
	//		cout << "esc key is pressed by user" << endl;
	//		break;}}
	
	
	getchar();
	return 0;}

//using namespace gpu;

//int lol(){
//	try	{
//
//		Mat src_host = imread("file.png", CV_LOAD_IMAGE_GRAYSCALE); // 8bit gray
//		GpuMat dst, src; // тоже самое что и Mat но в памяти gpu
//		
//		src.upload(src_host); // загрузка картинки в память gpu
//
//		threshold(src, dst, 128.0, 255.0, CV_THRESH_BINARY); //фильтр
//
//		Mat result_host;
//
//		dst.download(result_host); // выгрузка картинки с памяти gpu
//
//		imshow("Result", result_host);
//		waitKey();
//	}
//
//	catch (const cv::Exception& ex){cout << "Error: " << ex.what() << std::endl;}
//	return 0;
//}*/