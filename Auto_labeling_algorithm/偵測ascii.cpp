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
	int c,as=0;


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
	Mat frame2 = imread("E:\\dataset_MNIST_DVS\\9\\scale8\\   2.jpg", CV_LOAD_IMAGE_COLOR);
	//標框
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
	//宣告視窗
	namedWindow("show", CV_WINDOW_AUTOSIZE);
	imshow("show",frame);

	//鍵盤輸入對圖片影響
	while (1)
	{
		//等待按鍵ascii碼輸入
		as = waitKey(0);
		if (as == 27)//輸入esc
		{
			printf("成功結束");
			break;
		}
		if (as == 97)//輸入a
		{
			imshow("show", frame2);
		}
		if (as == 115)//輸入s
		{
			imshow("show", frame);
		}
	}
	fclose(infile);
	waitKey(3000);
	return 0;
}