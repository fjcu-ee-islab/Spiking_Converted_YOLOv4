#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	//bbox宣告
	Rect bbox;
	//檔案指標
	FILE *infile;
	double x, y, width, height;
	int c;


	//讀取txt
	infile = fopen("E:\\dataset_MNIST_DVS\\9\\scale8\\bboxout.txt","r");
	
	//開始讀取
	
	fscanf(infile,"%d %lf %lf %lf %lf",&c,&x,&y,&width,&height);
	printf("%d %.3lf %.3lf %.3lf %.3lf\n", c, x, y, width, height);
	bbox.x = x;
	bbox.y = y;
	bbox.width = width;
	bbox.height = height;
	//讀取圖片
	Mat frame = imread("E:\\dataset_MNIST_DVS\\9\\scale8\\   1.jpg", CV_LOAD_IMAGE_COLOR);
	//標框
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
	//宣告視窗
	namedWindow("show", CV_WINDOW_AUTOSIZE);
	imshow("show",frame);



	fclose(infile);
	waitKey(0);
	return 0;
}