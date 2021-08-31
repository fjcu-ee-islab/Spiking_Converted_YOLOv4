#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <conio.h>
#include <opencv2/core/ocl.hpp>
#include <fstream>
#include <windows.h>

using namespace cv;
using namespace std;

int main()
{

		string IMG_PATH = "E:\\fju_event_pedestrian_detection\\no_background\\test_label\\total\\*.jpg";
		vector<cv::String> filenames;
		cv::glob(IMG_PATH, filenames);

		Mat frame;
		FILE *infile;
		double x, y, width, height;
		int c;
		ofstream bboxout;


		cout << "total images are: " << filenames.size() << endl;
		for (int num = 0; num < filenames.size(); num++)
		{
			cout << filenames[num] << endl;
			//獲取路徑下的檔名
			const size_t last_idx = filenames[num].rfind('\\');
			string basename = filenames[num].substr(last_idx + 1);


			string SAVE_DIR = "E:\\fju_event_pedestrian_detection\\no_background\\test_label\\total_rename\\";
			string READ_DIR = "E:\\fju_event_pedestrian_detection\\no_background\\test_label\\total\\";
			string save_pathimg = SAVE_DIR + basename.substr(0, basename.rfind('.'))+"_nobackground.jpg";
			string read_pathtxt = READ_DIR + basename.substr(0, basename.rfind('.')) + ".txt";
			string save_pathtxt = SAVE_DIR + basename.substr(0, basename.rfind('.')) + "_nobackground.txt";

			cout << save_pathimg << endl;
			cout << read_pathtxt << endl;
			cout << save_pathtxt << endl;

			//讀txt，存txt
			infile = fopen(read_pathtxt.c_str(), "r");
			fscanf(infile, "%d %lf %lf %lf %lf", &c, &x, &y, &width, &height);
			bboxout = ofstream(save_pathtxt);
			c = 0;
			bboxout << c << " " << x << " " << y << " " << width << " " << height;

			//讀圖、存圖
			frame = imread(filenames[num], CV_LOAD_IMAGE_COLOR);
			imwrite(save_pathimg, frame);

			fclose(infile);
		}


	system("pause");
	return 0;
}