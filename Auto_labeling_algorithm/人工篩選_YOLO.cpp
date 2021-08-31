#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <conio.h>
#include <opencv2/core/ocl.hpp>
#include <fstream>

using namespace cv;
using namespace std;

int main()
{
	//bbox宣告
	Rect bbox;
	//檔案指標
	FILE *infile;
	FILE *infileout;
	double x, y, width, height;
	int c, w, h, count;
	char txt[200], img[200],txtout[200],imgout[200];
	int num = 0,as;
	int out = 1;
	Mat frame;
	ofstream bboxout;




	//宣告視窗
	//namedWindow("show", CV_WINDOW_AUTOSIZE);

	for (count=1;count<=2032;count++)
	{
		sprintf(txt,"E:\\dataset_MNIST-DVS_final\\train\\scale8_%d_%d.txt",num,count);
		sprintf(img,"E:\\dataset_MNIST-DVS_final\\train\\scale8_%d_%d.jpg",num,count);
		//讀取txt
		infile = fopen(txt, "r");
		//讀取圖片
		frame = imread(img, CV_LOAD_IMAGE_COLOR);


		//開始讀取
		fscanf(infile, "%d %lf %lf %lf %lf", &c, &x, &y, &width, &height);
		printf("%d %d %lf %lf %lf %lf\n",count, c, x, y, width, height);
		bbox.width = (width*frame.cols);										//YOLO格式轉換
		bbox.height = (height*frame.rows);
		bbox.x = ((x*frame.cols) - (bbox.width / 2));
		bbox.y = ((y*frame.rows) - (bbox.height / 2));
		//標框
		rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
		imshow("show", frame);
		as = waitKey(0);
		if (as == 97)//輸入a
		{
			sprintf(txtout, "E:\\dataset_MNIST-DVS_final\\train_human\\scale8_0_%d.txt",out);
			sprintf(imgout,"E:\\dataset_MNIST-DVS_final\\train_human\\scale8_0_%d.jpg",out);
			frame = imread(img, CV_LOAD_IMAGE_COLOR); //讀回原檔
			bboxout = ofstream(txtout);
			bboxout << c << " " << x << " " << y << " " << width << " " << height;
			imwrite(imgout,frame);
			out++;
		}
		fclose(infile);
	}

	
	system("pause");
	return 0;
}