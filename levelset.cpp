#pragma warning(disable:4819)
#pragma warning(disable : 4996)

#include <iostream>
#include <math.h>
#include <opencv2\opencv.hpp>
#include <opencv2\opencv_lib.hpp>
#include <opencv2\highgui.hpp>

using namespace cv;
using namespace std;

/* calculation conditions  */
#define GRIDSIZE 500  //grid size //width of image
#define WBAND 10  //width of narrow band
#define WRESET 3  //width of reset region
#define INITIALOFFSET 1  //initial offset
#define MAXNARROWBAND 100*GRIDSIZE*WBAND  //max grid of narrow band
#define MAXFRONT 100*GRIDSIZE  //max grid of front

enum {FARAWAY, BAND, RESETBAND, FRONT};  // the labels of status

/* the params of speed funtion */
#define DT 1  //time increment
#define GAIN 0.1  //gain of curvature

/* variable */
double phi[GRIDSIZE][GRIDSIZE];  //supplementary function
double dphi[GRIDSIZE][GRIDSIZE];  //differential of supplementary function
double F[GRIDSIZE][GRIDSIZE];  //speed function
unsigned char gray[GRIDSIZE][GRIDSIZE];  //intensity of image
unsigned char status[GRIDSIZE][GRIDSIZE];  //current status
double Front[MAXFRONT][2];  //grid of front
int NFront;  //the number of grid of front
int NarrowBand[MAXNARROWBAND][2];  //grid of narrow band
int NNarrowBand;  //the number of grid of narrow band
int CircleMap[WBAND + 1][7 * (WBAND + 1)][2];  //for circle map
int NCircleMap[WBAND + 1];  //the number of pixels of each distance in circle map

/* function */
void InitializeFrontPosition();  //initialize front
double SetSpeedFunction(int);  //set speed on narrow band
int ReLabeling();  //crate label of front
void InitializeCircleMap();  //create circle map

Mat img;
Mat dst;

void DrawContour();

#define USAGE(argv)\
{\
	cerr << "Usage: " << argv[0] << " imagefile" << endl;\
	exit(-1);\
}

void InitializeCircleMap()
{
	for (int i = 0; i <= WBAND; i++)
	{
		NCircleMap[i] = 0;
	}

	for (int x = -WBAND; x <= WBAND; x++)
	{
		for (int y = -WBAND; y <= WBAND; y++)
		{
			int d;
			if ((d = int(sqrt(double(x*x + y*y)))) <= WBAND)
			{
				CircleMap[d][NCircleMap[d]][0] = x;
				CircleMap[d][NCircleMap[d]][1] = y;
				NCircleMap[d]++;
			}
		}
	}
}

void InitializeFrontPosition()
{
	int n = 0;
	memset(status, FARAWAY, sizeof(status));

	//set front grid
	for (int x = INITIALOFFSET; x < GRIDSIZE - INITIALOFFSET; x++)
	{
		status[x][INITIALOFFSET] = FRONT;
		Front[n][0] = x;
		Front[n][1] = INITIALOFFSET;
		phi[x][INITIALOFFSET] = 0.0;
		n++;
	}
	for (int y = INITIALOFFSET; y < GRIDSIZE - INITIALOFFSET; y++)
	{
		status[GRIDSIZE-1-INITIALOFFSET][y] = FRONT;
		Front[n][0] = GRIDSIZE-1-INITIALOFFSET;
		Front[n][1] = y;
		phi[GRIDSIZE-1-INITIALOFFSET][y] = 0.0;
		n++;
	}
	for (int x = GRIDSIZE-1-INITIALOFFSET; x >= INITIALOFFSET; x--)
	{
		status[x][GRIDSIZE-1-INITIALOFFSET] = FRONT;
		Front[n][0] = x;
		Front[n][1] = GRIDSIZE-1-INITIALOFFSET;
		phi[x][GRIDSIZE-1-INITIALOFFSET] = 0.0;
		n++;
	}
	for (int y = GRIDSIZE-1-INITIALOFFSET; y >= GRIDSIZE - INITIALOFFSET; y--)
	{
		status[INITIALOFFSET][y] = FRONT;
		Front[n][0] = INITIALOFFSET;
		Front[n][1] = y;
		phi[INITIALOFFSET][y] = 0.0;
		n++;
	}
	NFront = n;

	for (int x = 0; x < GRIDSIZE; x++)
	{
		for (int y = 0; y < GRIDSIZE; y++)
		{
			if (status[x][y] != FRONT)
			{
				if (x > INITIALOFFSET && x < GRIDSIZE - INITIALOFFSET - 1 && y > INITIALOFFSET && y < GRIDSIZE - INITIALOFFSET - 1)
					phi[x][y] = -WBAND;
				else
					phi[x][y] = WBAND;
			}
		}
	}

	//setting of narrow band
	NNarrowBand = 0;
	SetSpeedFunction(1);
}

double SetSpeedFunction(int reset)
{
	double dx, dy;
	double dfx, dfy, dfx2, dfy2, dfxy, df;
	double kappa;
	double Fs = 0;

	for (int i = 0; i < NNarrowBand; i++)
	{
		int x = NarrowBand[i][0];
		int y = NarrowBand[i][1];
		F[x][y] = 0.0;
		dphi[x][y] = 0.0;

		if (reset) {
			if (status[x][y] != FRONT)
				status[x][y] = FARAWAY;
		}
	}

	for (int i = 0; i < NFront; i++)
	{
		int x = int(Front[i][0]);
		int y = int(Front[i][1]);

		if (x < 1 || x > GRIDSIZE - 2 || y < 1 || y > GRIDSIZE - 2)
			continue;

		dx = gray[x + 1][y] - gray[x][y];
		dy = gray[x][y + 1] - gray[x][y];

		F[x][y] = 1.0 / (1.0 + sqrt(dx*dx + dy*dy));

		//calculate curvature
		dfx = ((phi[x + 1][y + 1] - phi[x - 1][y + 1])
			+ 2.0*(phi[x + 1][y] - phi[x - 1][y])
			+ (phi[x + 1][y - 1] - phi[x - 1][y - 1])) / 4.0 / 2.0;
		dfy = ((phi[x + 1][y + 1] - phi[x + 1][y - 1])
			+ 2.0*(phi[x][y + 1] - phi[x][y - 1])
			+ (phi[x - 1][y + 1] - phi[x - 1][y - 1])) / 4.0 / 2.0;
		dfxy = ((phi[x + 1][y + 1] - phi[x - 1][y + 1])
			- (phi[x + 1][y - 1] - phi[x - 1][y - 1])) / 2.0 / 2.0;
		dfx2 = ((phi[x + 1][y + 1] + phi[x - 1][y + 1] - 2.0*phi[x][y + 1])
			+ 2.0*(phi[x + 1][y] + phi[x - 1][y] - 2.0*phi[x][y])
			+ (phi[x + 1][y - 1] + phi[x - 1][y - 1] - 2.0*phi[x][y - 1])) / 4.0;
		dfy2 = ((phi[x + 1][y + 1] + phi[x + 1][y - 1] - 2.0*phi[x + 1][y])
			+ 2.0*(phi[x][y + 1] + phi[x][y - 1] - 2.0*phi[x][y])
			+ (phi[x - 1][y + 1] + phi[x - 1][y - 1] - 2.0*phi[x - 1][y])) / 4.0;

		if ((df = sqrt(dfx*dfx + dfy*dfy)) != 0.0)
			kappa = (dfx2 * dfy * dfy - 2.0 * dfx * dfy * dfxy + dfy2 * dfx * dfx) / (df * df * df);
		else
			kappa = 0.0;

		F[x][y] *= (-1.0 - GAIN * kappa);

		Fs += F[x][y];
	}

	for (int d = WBAND; d > 0; d--)
	{
		for (int i = 0; i < NFront; i++)
		{
			int xf = int(Front[i][0]);
			int yf = int(Front[i][1]);

			if (reset)
				phi[xf][yf] = 0.0;

			for (int j = 0; j < NCircleMap[d]; j++)
			{
				int x = xf + CircleMap[d][j][0];
				int y = yf + CircleMap[d][j][1];

				if (x < 0 || x > GRIDSIZE - 1 || y < 0 || y > GRIDSIZE - 1)
					continue;
				if (status[x][y] == FRONT)
					continue;

				if (reset)
				{
					if (d > WBAND - WRESET)
						status[x][y] = RESETBAND;
					else
						status[x][y] = BAND;
					phi[x][y] = (phi[x][y] < 0) ? -d : d;
				}

				// if not FRONT
				F[x][y] = F[xf][yf];
			}
		}
	}

	if (reset)
	{
		int n = 0;
		for (int x = 0; x < GRIDSIZE; x++)
		{
			for (int y = 0; y < GRIDSIZE; y++)
			{
				if (status[x][y] != FARAWAY)
				{
					NarrowBand[n][0] = x;
					NarrowBand[n][1] = y;
					if (n++ >= MAXNARROWBAND)
					{
						cout << "Too many NarrowBand Points" << endl;
						return 0;
					}
				}
			}
		}
		NNarrowBand = n;
	}
	return Fs;
}

void FrontPropagation()
{
	double fdxm, fdxp, fdym, fdyp;

	for (int i = 0; i < NNarrowBand; i++)
	{
		int x = NarrowBand[i][0];
		int y = NarrowBand[i][1];

		if (x<1 || x>GRIDSIZE - 2 || y<1 || y >GRIDSIZE - 2)
			continue;

		//Upwind scheme
		if (F[x][y] > 0.0)
		{
			fdxm = fmax(phi[x][y] - phi[x - 1][y], 0);
			fdxp = fmin(phi[x + 1][y] - phi[x][y], 0);
			fdym = fmax(phi[x][y] - phi[x][y - 1], 0);
			fdyp = fmin(phi[x][y + 1] - phi[x][y], 0);
			dphi[x][y] = F[x][y] * sqrt(fdxm*fdxm + fdxp*fdxp + fdym*fdym + fdyp*fdyp) * DT;
		}
		else
		{
			fdxm = fmax(phi[x + 1][y] - phi[x][y], 0);
			fdxp = fmin(phi[x][y] - phi[x - 1][y], 0);
			fdym = fmax(phi[x][y + 1] - phi[x][y], 0);
			fdyp = fmin(phi[x][y] - phi[x][y - 1], 0);
			dphi[x][y] = F[x][y] * sqrt(fdxm*fdxm + fdxp*fdxp + fdym*fdym + fdyp*fdyp) * DT;
		}
	}

	for (int i = 0; i < NNarrowBand; i++)
	{
		int x = NarrowBand[i][0];
		int y = NarrowBand[i][1];
		phi[x][y] -= dphi[x][y];
	}
}

int ReLabeling()
{
	int n = 0;
	int flg = 0;
	for (int i = 0; i < NNarrowBand; i++)
	{
		int x = NarrowBand[i][0];
		int y = NarrowBand[i][1];

		if (x < 1 || x > GRIDSIZE - 2 || y < 1 || y > GRIDSIZE - 2)
			continue;

		if ((phi[x][y] >= 0.0 &&
			((phi[x + 1][y] + phi[x][y] <= 0.0) || (phi[x][y + 1] + phi[x][y] <= 0.0) ||
			(phi[x - 1][y] + phi[x][y] <= 0.0) || (phi[x][y - 1] + phi[x][y] <= 0.0))) ||
			(phi[x][y] <= 0.0 &&
			((phi[x + 1][y] + phi[x][y] >= 0.0) || (phi[x][y + 1] + phi[x][y] >= 0.0) ||
			(phi[x - 1][y] + phi[x][y] >= 0.0) || (phi[x][y - 1] + phi[x][y] >= 0.0))))
		{
			if (status[x][y] == RESETBAND)
				flg = 1;

			status[x][y] = FRONT;
			Front[n][0] = x;
			Front[n][1] = y;

			if (n++ >= MAXFRONT)
			{
				cout << "Too many Front Points" << endl;
				return -1;
			}
		}
		else
		{
			if (status[x][y] == FRONT)
				status[x][y] = BAND;
		}
	}

	NFront = n;

	return flg;
}

void DrawContour()
{
	Scalar c = CV_RGB(100, 100, 100);
	img.Mat::copyTo(dst);
	for (int i = 0; i < NFront - 1; i++)
	{
		rectangle(dst, Point(int(Front[i][1]), int(Front[i][0])), Point(int(Front[i][1] + 1), int(Front[i][0]) + 1), c);
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
		USAGE(argv);
	if ((img = imread(argv[1], 0)).empty())
		USAGE(argv);
	if (img.cols != GRIDSIZE || img.rows != GRIDSIZE)
	{
		cout << "Image size must be " << GRIDSIZE << " x " << GRIDSIZE << endl;
		exit(-1);
	}

	dst = img.Mat::clone();

	namedWindow("image", 1);

	InitializeCircleMap();

	InitializeFrontPosition();

	int reset = 1;
	double Fs = 0;

	do {
		imshow("image", dst);

		if ((Fs = SetSpeedFunction(reset)) == 0.0)
			break;

		FrontPropagation();

		if ((reset = ReLabeling()) < 0)
			break;

		DrawContour();

		int key = waitKey(10);
		if (key == 'q' || key == 27)
			break;
	} while (cvGetWindowHandle("image"));
}