#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <fstream>
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	//宣告兩個座標點，boundingbox兩個頂點(哪兩點無所謂)
	CvPoint p1, p2;
	p1.x = 200;
	p1.y = 200;
	p2.x = 50;
	p2.y = 50;

	


	//IplImage是通用的結構，CV_LOAD_IMAGE_UNCHANGED以原圖影像讀取
	IplImage* img = cvLoadImage("C:\\Users\\shon\\Desktop\\JAER\\9\\scale8\\test_img.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//創建一張圖片，大小直接使用img的大小
	IplImage* img2 = cvCreateImage(cvSize(img->height,img->width), IPL_DEPTH_8U, 3);

	

	//以下是通過將每一點轉換為0以實現黑圖
	for (int i = 0; i < img2->height; i++)
	{
		uchar *ptrImage = (uchar*)(img2->imageData + i * img2->widthStep);

		for (int j = 0; j < img2->width; j++)
		{
			ptrImage[3 * j + 0] = 0;
			ptrImage[3 * j + 1] = 0;
			ptrImage[3 * j + 2] = 0;
		}
	}
	
	//宣告一個視窗
	cvNamedWindow("Example1", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("Example2", CV_WINDOW_AUTOSIZE);
	//繪製矩形(輸入、點1、點2、顏色(BGR)、CV_FILLED代表填滿框內、通道連結4或8都可、消除鋸齒)
	//cvRectangle(img, p1, p2, cvScalar(255, 255, 255), CV_FILLED)其實後面也可以都不打
	cvRectangle(img, p1, p2, cvScalar(255, 255, 255), 1, 4, 0);
	cvRectangle(img2, p1, p2, cvScalar(255, 255, 255), CV_FILLED, 4, 0);

	//將img放入這個視窗並顯示
	cvShowImage("Example1", img);
	cvShowImage("Example2", img2);
	
	//使用cvSaveImage一直出bug，因此改變作法，先轉換為Mat形式再用imwrite做儲存
	Mat buffer;
	buffer = cvarrToMat(img2);
	imwrite("C:\\Users\\shon\\Desktop\\JAER\\9\\scale8\\test_img_black.jpg",buffer);



	//cvWaitKey(0)裡面數字x代表會等待xm秒，0則代表按下任何鍵再繼續
	cvWaitKey(0);
	//以下兩個都是釋放記憶體
	cvReleaseImage(&img);
	cvDestroyWindow("Example1");
	cvReleaseImage(&img2);
	cvDestroyWindow("Example2");
	
}