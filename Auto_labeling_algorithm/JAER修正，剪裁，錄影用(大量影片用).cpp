#define _CRT_SECURE_NO_WARNINGS
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
	Mat frame, img, crop_img;
	char file[200],file1[200];
	int c = 1,num = 0 ;

	Rect roi;
	//左上角座標
	roi.x = 102;
	roi.y = 20;
	//裁切大小
	roi.width = 438;
	roi.height = 438;

	
	//宣告一個空的儲存檔案
	VideoWriter  writer;
	//最外迴圈控制不同數字，內迴圈控制該數字中的不同檔案
	for (num = 0; num < 10; num++)
	{
		for (c = 1; c <= 10; c++)
		{
			//更新儲存影片宣告格式(名字、編碼形式、偵數、影片長寬)，也可以加入路徑，否則輸出會在sourcecode一樣的資料夾
			sprintf(file, "E:\\dataset_MNIST-DVS_final\\video\\%d\\fix_scale8\\scale8_%d_%d.avi", num, num, c);
			writer = VideoWriter(file, CV_FOURCC('X', 'V', 'I', 'D'), 30.0, Size(438, 438));

			//先將圖像儲存到frame，並不停更新
			sprintf(file1, "E:\\dataset_MNIST-DVS_final\\video\\%d\\scale8\\scale8_%d_%d.avi", num, num, c);
			frame = capture.open(file1);

			//isOpened()是檢查capture是否有啟動
			if (!capture.isOpened())
			{
				printf("can not open ...\n");
				return -1;
			}

			//代表有讀取到影像
			while (capture.read(frame))
			{

				//flip(輸入,輸出,參數)
				//0是X軸翻轉，1是用Y軸翻轉
				flip(frame, img, 0);
				flip(img, img, 1);

				//裁切圖片
				crop_img = img(roi);
				//儲存影片
				writer << crop_img;
			}
			printf("%s已轉換為%s\n", file1, file);

		}
	}

	capture.release();
	writer.release();
	system("pause");
	return 0;
}