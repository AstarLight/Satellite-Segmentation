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
	if (argc != 3)
	{
		printf("invalid parameters! Usage: [input_src] [input_mask]\n");
		return -1;
	}

	string input_src = argv[1];
	string input_mask = argv[2];


	Mat mask = imread(input_mask, 0);
	Mat src = imread(input_src);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			uchar *p_mask = mask.ptr<uchar>(i, j);
			uchar *p_src = src.ptr<uchar>(i, j);

			if (p_mask[0] == VEGETATION)
			{
				p_src[0] = 159;
				p_src[1] = 255;
				p_src[2] = 84;
			}
			else if (p_mask[0] == ROAD)
			{
				p_src[0] = 38;
				p_src[1] = 71;
				p_src[2] = 139;
			}
			else if (p_mask[0] == BUILDING || p_mask[0] == 255)
			{
				p_src[0] = 34;
				p_src[1] = 180;
				p_src[2] = 238;
			}
			else if (p_mask[0] == WATER)
			{
				p_src[0] = 255;
				p_src[1] = 191;
				p_src[2] = 0;
			}


		}
	}

	imwrite("stack.png", src);


	return 0;
}