#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <fstream>
using namespace cv;
using namespace std;
double iou(int ax, int ay ,int aw, int ah, int bx, int by, int bw, int bh);

int main(int argc, char** argv)
{
	double buffer = 0;

	buffer = iou(0,0,10,10,5,0,10,10);
	printf("%lf\n", buffer);
	//cout << IOU << endl; 不一樣的印法
	system("Pause");
	return 0;
}
double iou(int ax, int ay, int aw, int ah, int bx, int by, int bw, int bh)
{
	//第一張圖左上角座標以及寬高
	Rect rect;
	rect.x = ax;
	rect.y = ay;
	rect.width = aw;
	rect.height = ah;


	//第二張圖左上角座標以及寬高
	Rect rect1;
	rect1.x = bx;
	rect1.y = by;
	rect1.width = bw;
	rect1.height = bh;

	//計算兩個矩形的交集
	Rect rect2 = rect | rect1;

	//計算兩個矩形的聯集
	Rect rect3 = rect & rect1;

	//根據雅卡爾指數計算IOU，分子為交集，分母為聯集
	double IOU = rect3.area() *1.0 / rect2.area();

	return IOU;
}