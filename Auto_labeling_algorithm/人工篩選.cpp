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
	double x, y, width, height;
	int c,as=0,a=1;
	Mat frame,frame_og;
	char filename[200], filename_o[200];

	//宣告視窗
	namedWindow("show", CV_WINDOW_AUTOSIZE);
	//創造座標記事本(存放)
	ofstream bboxout("E:\\dataset_MNIST_DVS_human\\9\\scale8\\bboxout.txt");



	
	//讀取txt
	infile = fopen("E:\\dataset_MNIST_DVS\\9\\scale8\\bboxout.txt","r");
	//開始讀取(feof(infile) == 0讀取結束會跳出)
	while (feof(infile) == 0)
	{
		fscanf(infile, "%d %lf %lf %lf %lf", &c, &x, &y, &width, &height);
		printf("%d %.3lf %.3lf %.3lf %.3lf\n", c, x, y, width, height);
		bbox.x = x;
		bbox.y = y;
		bbox.width = width;
		bbox.height = height;

		//讀取圖片
		sprintf(filename, "E:\\dataset_MNIST_DVS\\9\\scale8\\%4.d.jpg",c);
		frame = imread(filename, CV_LOAD_IMAGE_COLOR);
		frame_og = imread(filename, CV_LOAD_IMAGE_COLOR);
		//標框
		rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

		//鍵盤輸入對圖片影響
		//等待按鍵ascii碼輸入
		imshow("show", frame);					//顯示圖片
		as = waitKey(0);
		if (as == 97)//輸入a，認為可以的圖片
		{
			//儲存座標
			bboxout << a << " " << x << " " << y << " " << width << " " << height ;
			//確保最後一行不換行
			if (feof(infile) == 0)
			{
				bboxout << endl;
			}
			sprintf(filename_o, "E:\\dataset_MNIST_DVS_human\\9\\scale8\\%4.d.jpg", a);
			imwrite(filename_o, frame_og);
			a++;
		}
		//下一張圖片
		c++;

	}



	
	fclose(infile);
	bboxout.close();
	//waitKey(3000);
	return 0;
}