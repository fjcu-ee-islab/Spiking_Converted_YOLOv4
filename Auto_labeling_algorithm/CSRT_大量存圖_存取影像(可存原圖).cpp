#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

using namespace cv;
using namespace std;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

int main(int argc, char **argv)
{
	//�ŧi�x�s�榡
	VideoWriter writer("C:\\Users\\shon\\Desktop\\JAER\\9\\scale8\\whitetoblack_blur_CSRT.avi", CV_FOURCC('X', 'V', 'I', 'D'), 30.0, Size(438, 438));

	Mat frame_og;
	int c = 1,a = 3053,num = 9,buff;
	char filename[200], filename_o[200],name[200];
	double cx, cy,yw,yh ;
	double xmin, ymin, rew, reh,w,h;



	// List of tracker types in OpenCV 3.4.1
	string trackerTypes[8] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };
	// vector <string> trackerTypes(types, std::end(types));

	// Create a tracker�A�q�o�̽ջs�l�ܤ�k�A0~7�ع����W���ŧi���r��}�C
	string trackerType = trackerTypes[7];

	Ptr<Tracker> tracker;

#if (CV_MINOR_VERSION < 3)
	{
		tracker = Tracker::create(trackerType);
	}
#else
	{
		if (trackerType == "BOOSTING")
			tracker = TrackerBoosting::create();
		if (trackerType == "MIL")
			tracker = TrackerMIL::create();
		if (trackerType == "KCF")
			tracker = TrackerKCF::create();
		if (trackerType == "TLD")
			tracker = TrackerTLD::create();
		if (trackerType == "MEDIANFLOW")
			tracker = TrackerMedianFlow::create();
		if (trackerType == "GOTURN")
			tracker = TrackerGOTURN::create();
		if (trackerType == "MOSSE")
			tracker = TrackerMOSSE::create();
		if (trackerType == "CSRT")
			tracker = TrackerCSRT::create();
	}
#endif

	//�гy�y�аO�ƥ�
	ofstream bboxout("E:\\dataset_test\\bboxout.txt");
	//�гy�y�аO�ƥ�(���)
	ofstream bboxout_o;




	// Read video
	VideoCapture video("E:\\dataset_MNIST-DVS_final\\video\\9\\fix_scale8\\scale8_9_10.avi");//�U���q7�}�l

	// Exit if video is not opened
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		return 1;
	}

	// Read first frame 
	Mat frame;
	bool ok = video.read(frame);

	// Define initial bounding box�A�w�q�̪쪺�ءA�����S�Τ]�i�H
	Rect2d bbox(287, 23, 86, 320);

	// Uncomment the line below to select a different bounding box �A��e�ءA�o�歫�n
	bbox = selectROI(frame, false); 
	// Display bounding box. ��صe�W�Ĥ@�i��
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

	imshow("Tracking", frame);
	//�}�l�ϥ�tracking
	tracker->init(frame, bbox);

	//��Ū�ɴN�i�Jwhile�A�S���N���X
	while (video.read(frame))
	{
		// Start timer
		double timer = (double)getTickCount();

		// Update the tracking result
		bool ok = tracker->update(frame, bbox);

		// Calculate Frames per second (FPS)
		float fps = getTickFrequency() / ((double)getTickCount() - timer);


		//�����쪺���p
		if (ok)
		{
			//���ƻs���
			frame_og = frame.clone();
			//��������N�x�s
			sprintf(filename_o, "E:\\dataset_MNIST-DVS_final\\train\\9\\scale8_%d_%d.jpg",num, a);
			imwrite(filename_o, frame_og);
			//�x�s�y�Ш�O�ƥ�
			sprintf(name, "E:\\dataset_MNIST-DVS_final\\train\\9\\scale8_%d_%d.txt",num,a);
			bboxout_o = ofstream(name);
			xmin = bbox.x;
			ymin = bbox.y;
			rew = bbox.width;
			reh = bbox.height;
			w = frame_og.cols;
			h = frame_og.rows;
			cx = (xmin+((rew)/2))/w;
			cy = (ymin+((reh)/2))/h;
			yw = (rew)/(w);
			yh = (reh)/(h);
			//cout << cx << endl;
			bboxout_o << "9" << " " << cx << " " << cy << " " << yw << " " << yh << endl;
			printf("%d\n",a);
			a++;
			// Tracking success : Draw the tracked object
			rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
			
		}
		else
		{
			// Tracking failure detected.
			putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		}

		// Display frame.
		imshow("Tracking", frame);
		//�x�s�Ϥ��A�q0�}�l�s���Asprintf�O���s�R�W
		sprintf(filename, "E:\\dataset_test\\%4.d.jpg",c);
		imwrite(filename, frame);
		//�x�s�y�Ш�O�ƥ�
		bboxout << c << " " << bbox.x<< " " << bbox.y << " " << bbox.width << " " << bbox.height << endl;
		c++;




		//���G�x�s�v��
		writer << frame;

		// Exit if ESC pressed.
		int k = waitKey(1);
		if (k == 27)
		{
			break;
		}

	}
	bboxout.close();
	bboxout_o.close();
	//waitKey(0);
	system("pause");
	return 0;
}