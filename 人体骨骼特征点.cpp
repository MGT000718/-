#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
using namespace std;
const int POSE_PAIRS[3][20][2] = {
{   // COCO body
	{1,2}, {1,5}, {2,3},
	{3,4}, {5,6}, {6,7},
	{1,8}, {8,9}, {9,10},
	{1,11}, {11,12}, {12,13},
	{1,0}, {0,14},
	{14,16}, {0,15}, {15,17}
},
{   // MPI body
	{0,1}, {1,2}, {2,3},
	{3,4}, {1,5}, {5,6},
	{6,7}, {1,14}, {14,8}, {8,9},
	{9,10}, {14,11}, {11,12}, {12,13}
},
{   // hand
	{0,1}, {1,2}, {2,3}, {3,4},         // thumb
	{0,5}, {5,6}, {6,7}, {7,8},         // pinkie
	{0,9}, {9,10}, {10,11}, {11,12},    // middle
	{0,13}, {13,14}, {14,15}, {15,16},  // ring
	{0,17}, {17,18}, {18,19}, {19,20}   // small
} };

int main(int argc, char **argv)
{


	String modelTxt = "pose_deploy_linevec.prototxt";
	String modelBin = "pose_iter_440000.caffemodel";
	String imageFile = "pose.jpg";
	String dataset = "COCO";
	int W_in = 368;
	int H_in = 368;
	float thresh = 0.1;//过滤掉小于0.1的那些点
	float scale = 0.003922;//1/255 意思就是把每个数据重新置于[0-1]区间，原图片的数据区间是0-255

	int midx, npairs, nparts;
	if (!dataset.compare("COCO")) { midx = 0; npairs = 17; nparts = 18; }
	else if (!dataset.compare("MPI")) { midx = 1; npairs = 14; nparts = 16; }
	else if (!dataset.compare("HAND")) { midx = 2; npairs = 20; nparts = 22; }
	else
	{
		std::cerr << "Can't interpret dataset parameter: " << dataset << std::endl;
		exit(-1);
	}
	// read the network model
	cout << modelBin << endl;
	Net net = readNet(modelBin, modelTxt);
	net.setPreferableTarget(0);
	net.setPreferableBackend(0);
	// and the image
	Mat img = imread(imageFile);
	if (img.empty())
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	// send it through the network
	//W_in = img.cols*H_in / img.rows;
	Mat inputBlob = blobFromImage(img, scale, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
	cout << inputBlob.size << endl;//打印出输入的维度信息（n*c*h*w）因为我们定义了W_in，H_in为368，所以这里应该是1*3*368*368

	net.setInput(inputBlob);
	Mat result = net.forward();
	// the result is an array of "heatmaps", the probability of a body part being in location x,y
	cout << result.size << endl;//打印出输出的维度信息
	int H = result.size[2];
	int W = result.size[3];

	// find the position of the body parts
	vector<Point> points(22);
	for (int n = 0; n < nparts; n++)
	{
		// Slice heatmap of corresponding body's part.
		Mat heatMap(H, W, CV_32F, result.ptr(0, n));
		// 1 maximum per heatmap
		Point p(-1, -1), pm;
		double conf;
		minMaxLoc(heatMap, 0, &conf, 0, &pm);
		if (conf > thresh)
			p = pm;
		points[n] = p;
	}

	// connect body parts and draw it !
	float SX = float(img.cols) / W;
	float SY = float(img.rows) / H;
	for (int n = 0; n < npairs; n++)
	{
		// lookup 2 connected body/hand parts
		Point2f a = points[POSE_PAIRS[midx][n][0]];
		Point2f b = points[POSE_PAIRS[midx][n][1]];

		// we did not find enough confidence before
		if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
			continue;

		// scale to image size
		a.x *= SX; a.y *= SY;
		b.x *= SX; b.y *= SY;

		line(img, a, b, Scalar(0, 200, 0), 2);
		circle(img, a, 3, Scalar(0, 0, 200), -1);
		circle(img, b, 3, Scalar(0, 0, 200), -1);
	}

	imshow("OpenPose", img);
	waitKey();

	return 0;
}
