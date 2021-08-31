#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
	//宣告兩個架構，一個存儲影片，一個儲存圖像
	VideoCapture capture;
	Mat frame, img, crop_img, blur, gussian;

	Rect roi;
	//左上角座標
	roi.x = 102;
	roi.y = 20;
	//裁切大小
	roi.width = 438;
	roi.height = 438;

	//儲存影片宣告格式(名字、編碼形式、偵數、影片長寬)，也可以加入路徑，否則輸出會在sourcecode一樣的資料夾
	VideoWriter writer("C:\\Users\\shon\\Desktop\\JAER\\9\\scale8\\out.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30.0, Size(438, 438));
	VideoWriter writer2("C:\\Users\\shon\\Desktop\\JAER\\9\\scale8\\out_blur.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30.0, Size(438, 438));

	//先將圖像儲存到frame
	frame = capture.open("C:\\Users\\shon\\Desktop\\JAER\\9\\scale8\\mnist_9_scale08_0001.avi");

	//isOpened()是檢查capture是否有啟動
	if (!capture.isOpened())
	{
		printf("can not open ...\n");
		return -1;
	}
	//宣告一個視窗
	namedWindow("output", CV_WINDOW_AUTOSIZE);
	namedWindow("crop_output", CV_WINDOW_AUTOSIZE); 
	namedWindow("blur", CV_WINDOW_AUTOSIZE);
	namedWindow("gussian", CV_WINDOW_AUTOSIZE);
	//代表有讀取到影像
	while (capture.read(frame))
	{

		//flip(輸入,輸出,參數)
		//0是X軸翻轉，1是用Y軸翻轉
		flip(frame, img, 0);
		flip(img, img, 1);
		//imshow("output", frame);
		imshow("output", img);

		//裁切圖片
		crop_img = img(roi);
		
		//中值濾波
		medianBlur(crop_img,blur, 5);
		//高斯濾波
		GaussianBlur(crop_img,gussian, Size(5, 5), 0, 0);

		//秀出裁切圖片
		imshow("crop_output", crop_img);
		imshow("blur", blur);
		imshow("gussian", gussian);
		//等待33ms類似於30FPS
		waitKey(33);
		//儲存影片
		writer << crop_img;
		writer2 << blur;





	}
	capture.release();
	waitKey(0);
	return 0;
}