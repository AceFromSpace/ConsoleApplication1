#include <opencv2/core/core.hpp>                       
#include <opencv2/highgui/highgui.hpp>                    
#include "opencv2/objdetect/objdetect.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include <string> 
#include <iostream>
#include <windows.h>

//odwrócic irisa i zrobic z niego maske

using namespace cv;
using namespace std;

string face_cascade_name = "haarcascade_frontalface_alt.xml";   
string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier face_cascade;                               
CascadeClassifier eyes_cascade;                               
                         

string window_name =  "Img !";
string window_name2 = "Face !";
string window_name3 = "Left eye !";
string window_name4 = "Right eye !";

int slider_param1 = 70;
int slider_param2 = 30;
int slider_param3 = 70;
int slider_param4 = 30;


Point old_pos = Point(250,250);//old position of drawed circle
Point center_l;
Point center_r;
bool calibrated=false;
bool mode = false;
int iter=0;

vector<Point> points(0);
vector<Rect> faces(0);


Mat img_gray;                               
Mat img_face;
Mat img_left_eye;
Mat img_right_eye;


void Control_Mouse(Point pos_left, Point pos_right, Point cent_r, Point centr_l);
int detectFaceAndEyes(Mat img);
Mat detectIris(Mat img,int threshold_level);
Point DrawingConturs(Mat img,Mat img_to_draw, Scalar color,Point left_corner,Point right_corner);
void DrawControlledCircle(Point pos_left,Point pos_right,Point cent_r,Point centr_l);
Rect detectEyeCorner(Mat img, bool & flag, int threshold_level, Point &left_corner, Point &right_corner);
Point GrabPoints(Point position_left,Point position_right);
void Calibration();
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
int CDF(Mat img ,int slider_param);//returns threshold from cumulative 
void GetDesktopResolution(int& horizontal, int& vertical);




int main(int argc, char** argv)
{
	Mat img, frame;
	VideoCapture capture= VideoCapture(0);
	//capture.open("http://192.168.1.1:4747/mjpegfeed");
	if (!capture.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video camera" << endl;
		return -1;
	}
	
	
	double dWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH); 
	double dHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT); 

	cout << "Frame size : " << dWidth << " x " << dHeight << endl;

	if (!face_cascade.load(face_cascade_name)) 
	{
		cout << "Can't find file face_cascade " << face_cascade_name << ".";
		return -2;
	}
	if (!eyes_cascade.load(eyes_cascade_name))       
	{
		cout << "Can't find file eye_cascade " << eyes_cascade_name << ".";
		return -2;
	}

	namedWindow(window_name, CV_WINDOW_AUTOSIZE);    
	namedWindow(window_name2, CV_WINDOW_AUTOSIZE);
	namedWindow(window_name3, CV_WINDOW_AUTOSIZE);
	namedWindow(window_name4, CV_WINDOW_AUTOSIZE);

	createTrackbar("Eye Corner theshold left", window_name, &slider_param1, 255, 0);
	createTrackbar("Iris threshold left", window_name, &slider_param2, 255, 0);
	createTrackbar("Eye Corner theshold right", window_name, &slider_param3, 255, 0);
	createTrackbar("Iris threshold right", window_name, &slider_param4, 255, 0);

	setMouseCallback(window_name, CallBackFunc, NULL);
	moveWindow(window_name3, 700, 500);
	moveWindow(window_name4, 800, 500);
                         
	Mat empty(Size(1, 1), CV_8UC1);
	img_gray = Mat::zeros(Size(40, 40), CV_8UC1);
	img_face = Mat::zeros(Size(40, 40), CV_8UC1);
	img_left_eye= Mat::zeros(Size(40, 40), CV_8UC1);
	img_right_eye = Mat::zeros(Size(40, 40), CV_8UC1);

	cout << "Do you want to show all steps and control circle insted of mouse ?[y/n]" << endl;
	string choice;
	cin >> choice;
	if (choice == "y") mode = true;

	while (waitKey(20) != 27)
	{
		capture >> frame;
		frame.copyTo(img);
		detectFaceAndEyes(img);
	}                                      
	return 0;
}


int detectFaceAndEyes(Mat img)
{

	cvtColor(img, img_gray, CV_BGR2GRAY);
	face_cascade.detectMultiScale(img_gray, faces, 1.1, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT, Size(50, 50));
	for (unsigned i = 0; i < faces.size(); i++)
	{
		Rect rect_face(faces[i]);
		Rect faces_half(faces[i].x, faces[i].y, faces[i].width, faces[i].height * 6 / 10);
		img_face = img_gray(faces_half);
		rectangle(img_gray, faces_half, Scalar(255, 0, 0), 2, 2, 0);
		vector<Rect> eyes;
		eyes_cascade.detectMultiScale(img_face, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		for (unsigned j = 0; j < eyes.size(); j++)
		{
			Rect rect_eye(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y * 25 / 21, eyes[j].width, eyes[j].height * 7 / 10);
			if (eyes[j].x < (faces[i].width / 3)) { img_left_eye = img_gray(rect_eye); /*color_eye_left = hue(rect_eye); */ }
			else	img_right_eye = img_gray(rect_eye);
		}
	}
		bool flag = false;

		int treshold_left = CDF(img_left_eye, slider_param1);
		int treshold_right = CDF(img_right_eye, slider_param3); //gets bright level from cumulative histogram 

		Point left_eye_left_corner(0, 0);
		Point left_eye_right_corner(0, 0);
		Point right_eye_left_corner(0, 0);
		Point right_eye_right_corner(0, 0);

		Rect sclera_left = detectEyeCorner(img_left_eye, flag, treshold_left, left_eye_left_corner, left_eye_right_corner);
		Rect sclera_right = detectEyeCorner(img_right_eye, flag, treshold_right, right_eye_left_corner, right_eye_left_corner);
		
		Mat new_img_left_eye = img_left_eye(sclera_left);
		Mat new_img_right_eye = img_right_eye(sclera_right);
		int treshold_left_iris = CDF(new_img_left_eye, slider_param2);
		int treshold_right_iris = CDF(new_img_right_eye, slider_param4);

		Mat new_img_left_eye_edges = detectIris(new_img_left_eye, treshold_left_iris);
		Mat new_img_right_eye_edges = detectIris(new_img_right_eye, treshold_right_iris);

		Point position_left  = DrawingConturs(new_img_left_eye_edges , new_img_left_eye, 255 , left_eye_left_corner , left_eye_right_corner);
		Point position_right = DrawingConturs(new_img_right_eye_edges, new_img_right_eye, 255, right_eye_left_corner, right_eye_right_corner);

		
		cout << position_left << "  " << position_right << endl;
	
		if (flag == false)
		{

			if (!calibrated);
			else if (iter < 50)
			{
				GrabPoints(position_left, position_right);
				iter++;
			}
			else if (iter == 50) 
			{
				Calibration();
				iter++;
			}
			else
			{
				if(mode)DrawControlledCircle(position_left, position_right, center_l, center_r);
				else Control_Mouse(position_left, position_right, center_l, center_r);
				iter++;
			}
			
		}
		imshow(window_name, img_gray);
		imshow(window_name2, img_face);
		imshow(window_name3, img_left_eye);
		imshow(window_name4, img_right_eye);
		
		return 0;
}

Mat detectIris(Mat img, int threshold_level)
{
	string window_name_input = "input";
	string window_name_eroded = "eroded";
	string window_name_dilatated = "dilatated";
	string window_name_thresholded = "thresholded";
	
	if (mode)
	{
		namedWindow(window_name_input, CV_WINDOW_AUTOSIZE);
		namedWindow(window_name_eroded, CV_WINDOW_AUTOSIZE);
		namedWindow(window_name_dilatated, CV_WINDOW_AUTOSIZE);
		namedWindow(window_name_thresholded, CV_WINDOW_AUTOSIZE);
		moveWindow(window_name_input, 600, 300);
		moveWindow(window_name_eroded, 600, 400);
		moveWindow(window_name_dilatated, 600, 500);
		moveWindow(window_name_thresholded, 600, 600);
	}

	Mat output,eroded,dilatated,thresholded;
	img.copyTo(output);
	medianBlur(output, output, 3);  
	equalizeHist(output, output);
	//GaussianBlur(output, output, Size(3, 3), 0, 0);
	if(mode)imshow(window_name_input, output);
	erode(output, eroded, Mat());
	equalizeHist(eroded, eroded);
	if (mode)imshow(window_name_eroded, eroded);
	dilate(eroded, dilatated, Mat());

	erode(dilatated, dilatated, Mat());
	dilate(dilatated, dilatated, Mat());
	erode(dilatated, dilatated, Mat());
	dilate(dilatated, dilatated, Mat());
	erode(dilatated, dilatated, Mat());
	dilate(dilatated, dilatated, Mat());

	equalizeHist(dilatated, dilatated);
	if (mode)imshow(window_name_dilatated, dilatated);
	threshold(dilatated, thresholded, threshold_level, 255, 0);
	if (mode)imshow(window_name_thresholded, thresholded);
	Canny(thresholded, thresholded, 30, 130, 3);

	return thresholded;

}
Point DrawingConturs(Mat img, Mat img_to_draw, Scalar color, Point left_corner, Point right_corner)
{
	Mat output;
	img.copyTo(output);
  
	vector<vector<Point> > contours(1);
	vector<Vec4i> hierarchy;
	dilate(output, output, Mat());
	
	if(mode)imshow("xvc", output);
	if(mode)moveWindow("xvc", 400, 400);
	findContours(output, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_TC89_KCOS);//NONE- all points , simple separates directions and return end points
	int biggest=0;
	double area = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		if (contourArea(Mat(contours[i]))>area)
		{
			area = contourArea(Mat(contours[i]));
			biggest = i;
		}
	}
	Point left(output.cols,0);
	Point right=(0,0);
	Point top = (0, output.rows);
	Point bottom = (0, 0);
	for (int i = 0; i < contours[biggest].size(); i++)
	{
		if (contours[biggest][i].x>right.x)
		{
			right = contours[biggest][i];
		}
		if (contours[biggest][i].x<left.x)
		{
			left = contours[biggest][i];
		}
		if (contours[biggest][i].y>bottom.y)
		{
			bottom = contours[biggest][i];
		}
		if (contours[biggest][i].y<top.y)
		{
			top = contours[biggest][i];
		}

	}
	
	RotatedRect box;
	Mat drawing = Mat::zeros(output.size(), CV_8UC3);
	if(contours[biggest].size()>5)box = fitEllipse(contours[biggest]);
	if (box.size.width > 0.1)
	{
		ellipse(img, box, Scalar(0, 255, 255), 1, LINE_AA);
	}
	
	int center_x = ((left.x + right.x) * 500) / img.cols;
	int center_y = ((left.y + right.y) * 500) / (left_corner.y + right_corner.y+1);
	//int center_y = ((bottom.y + top.y) * 500) / (left_corner.y + right_corner.y + 1);
	
	Point center(center_x,center_y);
	//Point center_circle((left.x + right.x) / 2, (left.y + right.y)/2);
	Point center_circle((left.x + right.x) / 2, (bottom.y + top.y) / 2);
	circle(img_to_draw, center_circle, 2, 100, -1);
	Rect reccc(0, 0, img.cols, img.rows);
	rectangle(img_to_draw, reccc, 0, 1);
	
	return center;
}
void DrawControlledCircle(Point pos_left,Point pos_right,Point cent_l,Point cent_r)
{

	Mat window = Mat::zeros(Size(500, 500), CV_8UC3);
	Point cir_pos = Point(old_pos.x, old_pos.y);
	int step = 20;
	int required_diff_x = 40;
	int required_diff_y = 40;

		if ((pos_left.x > (cent_l.x+required_diff_x))&& (pos_right.x > (cent_r.x + required_diff_x)))
		{
			if ((cir_pos.x - step) > 0) cir_pos.x = cir_pos.x - step;
		}
		else if ((pos_left.x < (cent_l.x - required_diff_x))&& (pos_right.x < (cent_r.x - required_diff_x)))
		{
			if ((cir_pos.x + step) < window.cols) cir_pos.x = cir_pos.x + step;
		}
	
		if ((pos_left.y > (cent_l.y + required_diff_y))&& (pos_right.y >(cent_r.y + required_diff_y)))
		{
			if ((cir_pos.y + step) < window.cols) cir_pos.y = cir_pos.y + step;
		}
		else if ((pos_left.y < (cent_l.y - required_diff_y))&& (pos_right.y < (cent_r.y - required_diff_y)))
		{
			if ((cir_pos.y - step) > 0) cir_pos.y = cir_pos.y - step;
		}
		circle(window, cir_pos, 10, Scalar(0, 255, 0), -1, 8, 0);
		old_pos.x = cir_pos.x;
		old_pos.y = cir_pos.y;
		imshow("CIRCLE!", window);

}
void Control_Mouse(Point pos_left, Point pos_right, Point cent_r, Point cent_l)
{
	int monitor_width, monitor_height;
	GetDesktopResolution(monitor_width, monitor_height);


	POINT pt;
	GetCursorPos(&pt);
	Mat window = Mat::zeros(Size(500, 500), CV_8UC3);
	Point cir_pos = Point(old_pos.x, old_pos.y);
	int step = 20;
	int required_diff_x = 40;
	int required_diff_y = 40;

	if ((pos_left.x > (cent_l.x + required_diff_x)) && (pos_right.x > (cent_r.x + required_diff_x)))
	{
		if ((pt.x - step) > 0) SetCursorPos(pt.x - step, pt.y);
	}
	else if ((pos_left.x < (cent_l.x - required_diff_x)) && (pos_right.x < (cent_r.x - required_diff_x)))
	{
		if ((pt.x + step) < monitor_width) SetCursorPos(pt.x + step, pt.y);
	}

	if ((pos_left.y > (cent_l.y + required_diff_y)) && (pos_right.y >(cent_r.y + required_diff_y)))
	{
		if ((pt.y + step) < monitor_height) SetCursorPos(pt.x, pt.y + step);
	}
	else if ((pos_left.y < (cent_l.y - required_diff_y)) && (pos_right.y < (cent_r.y - required_diff_y)))
	{
		if ((pt.y - step) > 0) SetCursorPos(pt.x, pt.y - step);
	}

}
Rect detectEyeCorner(Mat img,bool &flag,int threshold_level, Point &left_corner, Point &right_corner)
{
	string window_name_input = "input_corner";
	string window_name_eroded = "eroded_corner";
	string window_name_dilatated = "dilatated_corner";
	string window_name_thresholded = "thresholded_corner";

	if (mode)
	{
		namedWindow(window_name_input, CV_WINDOW_AUTOSIZE);
		namedWindow(window_name_eroded, CV_WINDOW_AUTOSIZE);
		namedWindow(window_name_dilatated, CV_WINDOW_AUTOSIZE);
		namedWindow(window_name_thresholded, CV_WINDOW_AUTOSIZE);
		moveWindow(window_name_input, 800, 400);
		moveWindow(window_name_eroded, 800, 500);
		moveWindow(window_name_dilatated, 800, 600);
		moveWindow(window_name_thresholded, 800, 700);
	}

	Mat input, eroded, dilatated, thresholded;
	img.copyTo(input);
	//Mat new_image = Mat::zeros(img.size(), img.depth());
	//img.copyTo(new_image);
	medianBlur(input, input, 3);
	//GaussianBlur(input, input, Size(3, 3), 0, 0);
	equalizeHist(input, input);
	if (mode)imshow(window_name_input, input);
	erode(input, eroded, Mat(),cv::Point(-1,-1),2); //opening
	equalizeHist(eroded, eroded);
	if (mode)imshow(window_name_eroded, eroded);
	dilate(eroded, dilatated, Mat());
	equalizeHist(dilatated, dilatated);
	if (mode)imshow(window_name_dilatated, dilatated);
	threshold(dilatated, thresholded, threshold_level, 255,0);
	if (mode)imshow(window_name_thresholded, thresholded);
	Point left(thresholded.rows, 0);
	Point right = (0, 0);
	Point bottom = (0, 0);
	for (int i = 0; i < thresholded.rows; i++)
	{
		for (int j = 0; j < thresholded.cols; j++)
		{
			
			if ((thresholded.at<uchar>(i, j) <100)&&(j>right.x))
			{
				right.x = j;
				right.y = i;
			}
			if ((thresholded.at<uchar>(i, j)<100)&&(j<left.x))
			{
				left.x = j;
				left.y = i;
			}
			if ((thresholded.at<uchar>(i, j)<10) && (i>bottom.y))
			{
				bottom.y = i;
			}
					}
	}
	int rect_width = right.x-left.x;

	//int rect_height = rect_width *4/10;//bottom.y;
	int rect_height = rect_width / 2;
	//int rect_y = ((left.y + right.y) / 2) - (rect_height / 2);
	int rect_y = 5;
	if (rect_y <= 0)rect_y = 0;
	if ((rect_width <= 0)|| ((left.x + rect_width)>= thresholded.cols)|| ((rect_y + rect_height)>= thresholded.rows))
	{
		cout << "1  "<< rect_width << endl;
		cout << "2 " << left.x + rect_width <<"  "<<img.cols<< endl;
		cout << "3  " << rect_y + rect_height / 2 <<"  "<<img.rows<< endl;
		cout << "4  " << rect_y << endl;
		flag = true;
		return Rect(1, 1, 1, 1);
	}
	

	circle(img, left, 2, 255, -1);
	circle(img, right, 2, 255, -1);
	left_corner = left;
	right_corner = right;
	Rect eye_mask(left.x,rect_y, rect_width, rect_height);
	//rectangle(img, eye_mask, 0, 1);
	return eye_mask;
	
}
Point GrabPoints(Point position_left,Point position_right)
{
	
	Mat output = Mat::zeros(Size(1366,768), CV_8UC3);
	circle(output, Point(output.cols/2, output.rows/2), 10, Scalar(0, 0, 255), -1);
	namedWindow("Calibration", CV_WINDOW_AUTOSIZE);
	moveWindow("Calibration", 0, 0);
	cv::imshow("Calibration", output);
	points.push_back(position_left);
	points.push_back(position_right);
	return Point(1, 1);
}
void Calibration()
{
	int sum_x_l = 0;
	int sum_y_l = 0;
	int sum_x_r = 0;
	int sum_y_r = 0;
	int i = 0;
	for ( i = 0; i < points.size(); i=i+2)
	{
		sum_y_l =sum_y_l+ points[i].y;
		sum_x_l =sum_x_l+ points[i].x;
	}
	for (i = 1; i < points.size(); i = i + 2)
	{
		sum_y_r = sum_y_r + points[i].y;
		sum_x_r = sum_x_r + points[i].x;
	}
	i = i / 2;
	cout << "Iteracji:: " << i << endl;
	center_l = Point(sum_x_l / i, sum_y_l / i);
	center_r = Point(sum_x_r / i, sum_y_r / i);
	cout << "Center:: " << center_l << endl;
	cout << "Center:: " << center_r << endl;

	

}
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		calibrated = true;
	}
}
int CDF(Mat img,int slider_param)
{
	Mat new_image;
	img.copyTo(new_image);
	equalizeHist(new_image, new_image);
	
	Mat temp;
	const int chan = 0;
	const int size = 256;
	float range[] = { 0,256 };
	const float *wskrange = { range };
	Mat Histimage(512, 400, CV_8UC3, Scalar(0, 0, 0));
	calcHist(&new_image,1 ,&chan, Mat(), temp,1, &size, &wskrange);
	
	int bin_w = cvRound((double)512 / size);
	float sum1 = 0;
	float sum2 = cvRound(temp.at<float>(1));
	bool flag=true;
	int threshold;
	for (int i = 1; i < size; i++)
	{
		sum1 = sum1 + cvRound(temp.at<float>(i - 1)) ;
		sum2 = sum2 + cvRound(temp.at<float>(i));
		line(Histimage, Point(bin_w*(i - 1), (400 - 400*(sum1/(img.total())))), Point(bin_w*(i), (400 - 400*(sum2/ (img.total())))), Scalar(0,0,255), 2, 8, 0);
		double temp = static_cast<double> (slider_param) / 255;
		
		if ((sum2 / (new_image.total()) >= temp) && flag)
		{
			line(Histimage, Point(bin_w*(i - 1), 400), Point(bin_w*(i - 1), 0), Scalar(255, 255, 255), 2, 8, 0);
			flag = false;
			threshold = i;
			//cout << i << endl;
		}
	}
	if(mode)cv::imshow("Histogram", Histimage);
	return threshold;
	
}
void GetDesktopResolution(int & horizontal, int & vertical)
{
	
		RECT desktop;
		// Get a handle to the desktop window
		const HWND hDesktop = GetDesktopWindow();
		// Get the size of screen to the variable desktop
		GetWindowRect(hDesktop, &desktop);
		// The top left corner will have coordinates (0,0)
		// and the bottom right corner will have coordinates
		// (horizontal, vertical)
		horizontal = desktop.right;
		vertical = desktop.bottom;
		
}
 