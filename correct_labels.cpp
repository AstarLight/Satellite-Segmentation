#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"


using namespace cv;

#define ALL 0
#define VEGETATION 1
#define ROAD 4
#define BUILDING 2
#define WATER 3


int main(int argc, char* argv[])
{
    
	Mat src = imread("5.png",0);


	for (int i = 0; i < src.rows; i++)
	{
		uchar *p_src = src.ptr<uchar>(i);
		for (int j = 0; j < src.cols; j++)
		{			
			if (p_src[j] == WATER)
			{
				p_src[j] = BUILDING;
			}
			else if (p_src[j] == ROAD)
			{
				p_src[j] = WATER;
			}
			else if (p_src[j] == BUILDING)
			{
				p_src[j] = ROAD;
			}
		}
	}

	imwrite("proc5.png", src);


	return 0;
}