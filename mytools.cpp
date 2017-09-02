#include "stdafx.h"
#include "mytools.h"
//���������ռ�
using namespace cv;
using std::vector;
//**************************���ھ���Ĺ���1   *****************************//
//
//*************************************************************************//
bool _EqPredicate1(const cv::Point& a, const cv::Point&  b)
{
	return ((b.y - a.y)*(b.y - a.y) + (b.x - a.x)*(b.x - a.x)<5 * 5);
}
//**************************���ھ���Ĺ���2   *****************************//
//
//*************************************************************************//
bool _EqPredicate2(const cv::Point& a, const cv::Point&  b)
{
	return ((b.y - a.y)*(b.y - a.y) + (b.x - a.x)*(b.x - a.x)<6 * 6);
}
//**************************Rosenfeldϸ���㷨*****************************//
//���ܣ���ͼ�����ϸ��
//������image������ͼ���һά����
//      lx��ͼ����
//      ly��ͼ��߶�
//      �޷���ֵ
//*************************************************************************//
void ThinnerRosenfeld(Mat& BinaryImg, Mat& ThinImg)
{
	unsigned long height = BinaryImg.rows;
	unsigned long width = BinaryImg.cols;
	unsigned char* image;
	image = new uchar[sizeof(char)*width*height]();
	int x, y;
	for (y = 0; y<height; y++)
	{
		unsigned char* ptr = (unsigned char*)(BinaryImg.data + y*BinaryImg.step);
		for (x = 0; x<width; x++)
		{
			image[y*width + x] = ptr[x] > 0 ? 1 : 0;
		}
	}

	char *f, *g;
	char n[10];
	char a[5] = { 0, -1, 1, 0, 0 };
	char b[5] = { 0, 0, 0, 1, -1 };
	char nrnd, cond, n48, n26, n24, n46, n68, n82, n123, n345, n567, n781;
	short k, shori;
	unsigned long i, j;
	long ii, jj, kk, kk1, kk2, kk3, size;
	size = (long)width * (long)height;

	g = (char *)malloc(size);
	if (g == NULL)
	{
		printf("error in alocating mmeory!\n");
		return;
	}

	f = (char *)image;
	for (kk = 0l; kk<size; kk++)
	{
		g[kk] = f[kk];
	}

	do
	{
		shori = 0;
		for (k = 1; k <= 4; k++)
		{
			for (i = 1; i<height - 1; i++)
			{
				ii = i + a[k];

				for (j = 1; j<width - 1; j++)
				{
					kk = i*width + j;

					if (!f[kk])
						continue;

					jj = j + b[k];
					kk1 = ii*width + jj;

					if (f[kk1])
						continue;

					kk1 = kk - width - 1;
					kk2 = kk1 + 1;
					kk3 = kk2 + 1;
					n[3] = f[kk1];
					n[2] = f[kk2];
					n[1] = f[kk3];
					kk1 = kk - 1;
					kk3 = kk + 1;
					n[4] = f[kk1];
					n[8] = f[kk3];
					kk1 = kk + width - 1;
					kk2 = kk1 + 1;
					kk3 = kk2 + 1;
					n[5] = f[kk1];
					n[6] = f[kk2];
					n[7] = f[kk3];

					nrnd = n[1] + n[2] + n[3] + n[4]
						+ n[5] + n[6] + n[7] + n[8];
					if (nrnd <= 1)
						continue;

					cond = 0;
					n48 = n[4] + n[8];
					n26 = n[2] + n[6];
					n24 = n[2] + n[4];
					n46 = n[4] + n[6];
					n68 = n[6] + n[8];
					n82 = n[8] + n[2];
					n123 = n[1] + n[2] + n[3];
					n345 = n[3] + n[4] + n[5];
					n567 = n[5] + n[6] + n[7];
					n781 = n[7] + n[8] + n[1];

					if (n[2] == 1 && n48 == 0 && n567>0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[6] == 1 && n48 == 0 && n123>0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[8] == 1 && n26 == 0 && n345>0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[4] == 1 && n26 == 0 && n781>0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[5] == 1 && n46 == 0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[7] == 1 && n68 == 0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[1] == 1 && n82 == 0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}

					if (n[3] == 1 && n24 == 0)
					{
						if (!cond)
							continue;
						g[kk] = 0;
						shori = 1;
						continue;
					}
					cond = 1;
					if (!cond)
						continue;
					g[kk] = 0;
					shori = 1;
				}
			}
			for (i = 0; i<height; i++)
			{
				for (j = 0; j<width; j++)
				{
					kk = i*width + j;
					f[kk] = g[kk];
				}
			}
		}
	} while (shori);

	for (y = 0; y<ThinImg.rows; y++)
	{
		unsigned char* ptr = (unsigned char*)(ThinImg.data + y*ThinImg.step);
		for (x = 0; x<ThinImg.cols; x++)
		{
			ptr[x] = image[y*ThinImg.cols + x]>0 ? 255 : 0;
		}

	}
	const int height1 = ThinImg.rows - 1;
	const int width1 = ThinImg.cols - 1;
	uchar *pU, *pC, *pD;

	for (int i = 1; i<height1; i++)  //һ�� ���к���ɨ�� ��ʼ
	{
		pU = ThinImg.ptr<uchar>(i - 1);
		pC = ThinImg.ptr<uchar>(i);
		pD = ThinImg.ptr<uchar>(i + 1);
		for (int j = 1; j<width1; j++)
		{
			if (pC[j] > 0)
			{
				int p2 = (pU[j] >0);
				int p3 = (pU[j + 1] >0);
				int p4 = (pC[j + 1] >0);
				int p5 = (pD[j + 1] >0);
				int p6 = (pD[j] >0);
				int p7 = (pD[j - 1] >0);
				int p8 = (pC[j - 1] >0);
				int p9 = (pU[j - 1] >0);
				//if ((p2*p4*p6 == 1) || (p2*p4*p8 == 1) || (p4*p6*p8 == 1) || (p2*p6*p8 == 1)){
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)> 2)
				{
					if ((p9*p2*p3 != 1) && (p4*p2*p3 != 1) && (p4*p5*p3 != 1) && (p4*p5*p6 != 1) &&
						(p5*p6*p7 != 1) && (p6*p7*p8 != 1) && (p7*p8*p9 != 1) && (p9*p2*p8 != 1)){
						//if ((p2*p4*p6 == 1) || (p4*p6*p8 == 1) || (p6*p8*p2 == 1) || (p8*p2*p4 == 1) ||
						//	(p2*p8*p5 == 1) || (p2*p4*p7 == 1) || (p4*p6*p9 == 1) || (p6*p8*p3 == 1) ||
						//	(p3*p9*p6 == 1) || (p3*p5*p8 == 1) || (p5*p7*p2 == 1) || (p7*p9*p4 == 1)){
						Point2i centerPoint;
						centerPoint.x = j;
						centerPoint.y = i;
						CenterPoints0.push_back(centerPoint);//�����Ĵ�����
						//cvCircle(pointImg, cvPoint(j, i), 8, CV_RGB(255, 0, 0), 2);//����ƥ���������
						//cvNamedWindow("Show2");
						//cvShowImage("Show2", pointImg);
					}
				}
			}
		}
	} //һ�� ���к���ɨ�����  
	free(g);
}
//**************************���������㷨ȥ��С�������*****************************//
//
//CheckMode: 0����ȥ��������1����ȥ��������; NeihborMode��0����4����1����8����;  
//
//********************************************************************************//
void RemoveSmallRegion(Mat& Src, Mat& Dst, int AreaLimit, int CheckMode, int NeihborMode)
{
	int RemoveCount = 0;       //��¼��ȥ�ĸ���  
	//��¼ÿ�����ص����״̬�ı�ǩ��0����δ��飬1�������ڼ��,2�����鲻�ϸ���Ҫ��ת��ɫ����3������ϸ������  
	Mat Pointlabel = Mat::zeros(Src.size(), CV_8UC1);

	if (CheckMode == 1)
	{
		//ȥ��С����
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] < 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}
	else
	{
			//ȥ���ն�
		for (int i = 0; i < Src.rows; ++i)
		{
			uchar* iData = Src.ptr<uchar>(i);
			uchar* iLabel = Pointlabel.ptr<uchar>(i);
			for (int j = 0; j < Src.cols; ++j)
			{
				if (iData[j] > 10)
				{
					iLabel[j] = 3;
				}
			}
		}
	}

	vector<Point2i> NeihborPos;  //��¼�����λ��  
	NeihborPos.push_back(Point2i(-1, 0));
	NeihborPos.push_back(Point2i(1, 0));
	NeihborPos.push_back(Point2i(0, -1));
	NeihborPos.push_back(Point2i(0, 1));
	if (NeihborMode == 1)
	{   //8����
		NeihborPos.push_back(Point2i(-1, -1));
		NeihborPos.push_back(Point2i(-1, 1));
		NeihborPos.push_back(Point2i(1, -1));
		NeihborPos.push_back(Point2i(1, 1));
	}
	int NeihborCount = 4 + 4 * NeihborMode;
	int CurrX = 0, CurrY = 0;
	//��ʼ���  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 0)
			{
				//********��ʼ�õ㴦�ļ��**********  
				vector<Point2i> GrowBuffer;                                      //��ջ�����ڴ洢������  
				GrowBuffer.push_back(Point2i(j, i));
				Pointlabel.at<uchar>(i, j) = 1;
				int CheckResult = 0;                                               //�����жϽ�����Ƿ񳬳���С����0Ϊδ������1Ϊ����  

				for (int z = 0; z<GrowBuffer.size(); z++)
				{

					for (int q = 0; q<NeihborCount; q++)                                      //����ĸ������  
					{
						CurrX = GrowBuffer.at(z).x + NeihborPos.at(q).x;
						CurrY = GrowBuffer.at(z).y + NeihborPos.at(q).y;
						if (CurrX >= 0 && CurrX<Src.cols&&CurrY >= 0 && CurrY<Src.rows)  //��ֹԽ��  
						{
							if (Pointlabel.at<uchar>(CurrY, CurrX) == 0)
							{
								GrowBuffer.push_back(Point2i(CurrX, CurrY));  //��������buffer  
								Pointlabel.at<uchar>(CurrY, CurrX) = 1;           //���������ļ���ǩ�������ظ����  
							}
						}
					}

				}
				if (GrowBuffer.size()>AreaLimit) CheckResult = 2;                 //�жϽ�����Ƿ񳬳��޶��Ĵ�С����1Ϊδ������2Ϊ����  
				else { CheckResult = 1;   RemoveCount++; }
				for (int z = 0; z<GrowBuffer.size(); z++)                         //����Label��¼  
				{
					CurrX = GrowBuffer.at(z).x;
					CurrY = GrowBuffer.at(z).y;
					Pointlabel.at<uchar>(CurrY, CurrX) += CheckResult;
				}
				//********�����õ㴦�ļ��**********  


			}
		}
	}

	CheckMode = 255 * (1 - CheckMode);
	//��ʼ��ת�����С������  
	for (int i = 0; i < Src.rows; ++i)
	{
		uchar* iData = Src.ptr<uchar>(i);
		uchar* iDstData = Dst.ptr<uchar>(i);
		uchar* iLabel = Pointlabel.ptr<uchar>(i);
		for (int j = 0; j < Src.cols; ++j)
		{
			if (iLabel[j] == 2)
			{
				iDstData[j] = CheckMode;
			}
			else if (iLabel[j] == 3)
			{
				iDstData[j] = iData[j];
			}
		}
	}
}
//********************������ͼ�еĵ�ӳ��ظ��ӱ任ǰ��ͼ����***********************//
//
//********************************************************************************//
Point2i MapToSrc(short Bird_x, short Bird_y, short SrcWidth, short Srcheight){
	double fx, fy, cx, cy, k1, k2, pitch, h;
	fx = 359.8189;
	fy = 361.4601;
	cx = 687.5108;
	cy = 474.5289;
	k1 = -0.1395;
	k2 = 0.0119;
	pitch = 28 * CV_PI / 180;//720���Ǹ�35
	h = 1;
	double xw = double((Bird_x - SrcWidth*0.5) / 120.0);//��������ϵ����
	double yw = double((Srcheight*0.5 - Bird_y + 250) / 90.0);
	double x1 = xw / ((double(cos(pitch)))*yw + double(h));//�������굽�������ı任
	double y1 = yw*(double(-sin(pitch))) / (cos(pitch)*yw + double(h));
	double u = fx*x1 + cx;//������굽ͼ������
	double v = fy*y1 + cy;
	return Point2i(int(u), int(v));
}
//**************************���ڽ�ͼ���Ϊ����ͼ*****************************//
//Src:����ͼ��Dst���������ͼ
//fx, fy, cx, cy���ڲ�
//pitch, h�����
//*************************************************************************//
void TansformToBirdView(Mat& Src, Mat& Dst, double fx, double fy, double cx, double cy, double pitch, double h)
{
	Mat mapx = Mat::zeros(Src.size(), CV_32FC2);
	Mat mapy = Mat::zeros(Src.size(), CV_32FC1);
	pitch = pitch * CV_PI / 180;//720���Ǹ�35��45
	Dst = Mat::zeros(Size(Src.cols, Src.rows), CV_8UC3);//���ͼ��
	////****** ������� ��ֵ *******
	double k1, k2, k3, p1, p2;
	k1 = -0.1395;
	k2 = 0.0119;
	k3 = 0;
	p1 = 0;
	p2 = 0;

	//fx = 165.6028;
	//fy = 147.2656;
	//cx = 362.3568;
	//cy = 244.5130;
	//k1 = -0.1189;
	//k2 = 0.0096;
	Mat maskBird;//���������߽����ĤMASK
	maskBird = Mat::zeros(Size(Src.cols, Src.rows), CV_8UC1);//���������߽����ĤMASK
	//������ʧ��
	int DisapLine = fy*(-tan(CV_PI / 6)) + cy;
	for (int j = 0; j<Dst.rows; j++)//��
	{
		for (int i = 0; i<Dst.cols; i++)//��
		{
			//��һ�����þ���ķ�ʽ�����������굽�������ı任
			//Mat XYZW = Mat::zeros(Size(1, 4), CV_32FC1);//������ͼӳ�䵽��������ϵ�ϣ�ÿ��������x����y����ĳ�������������ϵ����һ���ı�����ϵ��
			//XYZW.at<float>(0, 0) = float((i - Dst.cols*0.5) / 60.0);
			//XYZW.at<float>(1, 0) = float((Dst.rows*0.5 - j + 250) / 45.0);//225
			//XYZW.at<float>(0, 0) = (i - Src.cols*0.5) / 60.0;
			//XYZW.at<float>(1, 0) = (j - Src.rows*0.5) / 45.0;//225
			//XYZW.at<float>(3, 0) = 1;

			//cv::Mat tpitchp = cv::Mat::zeros(cv::Size(4, 3), CV_32FC1); //��X����ת���Ƕ�Ϊ������
			//tpitchp.at<float>(0, 0) = 1;
			//tpitchp.at<float>(1, 1) = float(-sin(pitch*CV_PI / 180));
			//tpitchp.at<float>(1, 2) = float(-cos(pitch*CV_PI / 180));
			//tpitchp.at<float>(2, 1) = float(cos(pitch*CV_PI / 180));
			//tpitchp.at<float>(2, 2) = float(-sin(pitch*CV_PI / 180));
			//tpitchp.at<float>(2, 3) = float(h);//�������ϵ����������ϵ�ϵĸ߶�

			//cv::Mat transform = cv::Mat::zeros(cv::Size(1, 3), CV_32FC1);
			//transform = tpitchp*XYZW;//�������굽�������ı任
			//double x1 = transform.at<float>(0, 0) / transform.at<float>(2, 0);//��һ��
			//double y1 = transform.at<float>(1, 0) / transform.at<float>(2, 0);

			//��һ����ֱ�ӽ��������Ĺ�������Ϊ����ɹ�ʽ��
			//double xw = double((i - Dst.cols*0.5) / 90.0);//��������ϵ����
			//double yw = double((Dst.rows*0.5 - j) / 60.0);
			double xw = double((i - Dst.cols*0.5) / 72.0);//��������ϵ����
			double yw = double((Dst.rows*0.5 - j + 250) / 54.0);
			double x1 = xw / ((double(cos(pitch)))*yw + double(h));//�������굽�������ı任
			double y1 = yw*(double(-sin(pitch))) / (cos(pitch)*yw + double(h));

			//double x2 = x1*x1, y2 = y1*y1;//�������
			//double r2 = x2 + y2, _2xy = 2 * x1*y1;
			//double kr = (1 + ((k3*r2 + k2)*r2 + k1)*r2);
			//double u = fx*(x1*kr + p1*_2xy + p2*(r2 + 2 * x2)) + cx;
			//double v = fy*(y1*kr + p1*(r2 + 2 * y2) + p2*_2xy) + cy;

			double u = fx*x1 + cx;//������굽ͼ������
			double v = fy*y1 + cy;
			if (u>0 && u<(Src.cols - 1) && v>DisapLine&&v<(Src.rows - 1)){
				mapx.at<Vec2f>(j, i)[0] = float(u);
				mapx.at<Vec2f>(j, i)[1] = float(v);

				uchar* pBinary;
				pBinary = maskBird.ptr<uchar>(j);//Ϊmask��ֵ
				pBinary[i] = 255;

				//�������Լ�д��˫���Բ�ֵ
				//int x1 = int(u); int x2 = int(u + 1);
				//int y1 = int(v); int y2 = int(v + 1);
				//double x0 = u - double(x1);  double y0 = v - double(y1);
				//
				//double val = Src.at<Vec3b>(y1, x1)[0]*(1 - x0)*(1 - y0) + Src.at<Vec3b>(y1, x2)[0]*x0*(1 - y0) + Src.at<Vec3b>(y2, x1)[0]*(1 - x0)*y0 + Src.at<Vec3b>(y2, x2)[0]*x0*y0;
				//Dst.at<Vec3b>(j, i)[0] = uchar(val);
				//val = Src.at<Vec3b>(y1, x1)[1] * (1 - x0)*(1 - y0) + Src.at<Vec3b>(y1, x2)[1] * x0*(1 - y0) + Src.at<Vec3b>(y2, x1)[1] * (1 - x0)*y0 + Src.at<Vec3b>(y2, x2)[1] * x0*y0;
				//Dst.at<Vec3b>(j, i)[1] = uchar(val);
				//val = Src.at<Vec3b>(y1, x1)[2] * (1 - x0)*(1 - y0) + Src.at<Vec3b>(y1, x2)[2] * x0*(1 - y0) + Src.at<Vec3b>(y2, x1)[2] * (1 - x0)*y0 + Src.at<Vec3b>(y2, x2)[2] * x0*y0;
				//Dst.at<Vec3b>(j, i)[2] = uchar(val);
				//Dst.at<Vec3b>(j, i)[0] = Src.at<Vec3b>(int(v), int(u))[0];
				//Dst.at<Vec3b>(j, i)[1] = Src.at<Vec3b>(int(v), int(u))[1];
				//Dst.at<Vec3b>(j, i)[2] = Src.at<Vec3b>(int(v), int(u))[2];
			}
			else{
				mapx.at<Vec2f>(j, i)[0] = 0;
				mapx.at<Vec2f>(j, i)[1] = 0;

				//Dst.at<Vec3b>(j, i)[0] = 0;
				//Dst.at<Vec3b>(j, i)[1] = 0;
				//Dst.at<Vec3b>(j, i)[2] = 0;
			}

		}
	}
	// remap() with integer maps is faster
	//Mat map1 = Mat::zeros(Src.size(), CV_16SC2);
	//Mat map2 = Mat::zeros(Src.size(), CV_16SC2);
	//cv::convertMaps(mapx, mapy, map1, map2, CV_16SC2);

	remap(Src, Dst, mapx, Mat(), INTER_LINEAR);
	namedWindow("TestOutput");
	imshow("TestOutput", Dst);
	string file_path = "birdView10";
	file_path = file_path + to_string(1);
	file_path = file_path + ".png";
	imwrite(file_path, Dst);
	imwrite(file_path, maskBird);
}
//**************************���ڻ���ϵ����Ļ���У��*****************************//

//*************************************************************************//
void CcalibrationByLable(Mat& Src, Mat& Dst, double fx, double fy, double cx, double cy){

	fx = 165.6028;
	fy = 147.2656;
	cx = 362.3568;
	cy = 244.5130;
	Mat mapx = Mat::zeros(Src.size(), CV_32FC2);
	Mat mapy = Mat::zeros(Src.size(), CV_32FC1);
	Dst = Mat::zeros(Size(Src.cols, Src.rows), CV_8UC3);//���ͼ��
	for (int j = 0; j<Dst.rows; j++)//��
	{
		for (int i = 0; i<Dst.cols; i++)//��
		{
			double u, v;
			
			const CvPoint center1 = Point(362, 245);//960?	
			//const CvPoint center1 = Point(640, 360);//720?	
			const CvPoint center = Point(Src.cols / 2, Src.rows / 2);
			const double fovStep = 0.05;
			double cmosPixelSize = 0.0056;//0.0042
			double focalLength = 1.2834;
			double referHeight = sqrt(pow(i - center.x, 2) + pow(j - center.y, 2));
			double theta = referHeight * cmosPixelSize / focalLength;
			//theta = 1.95 * atan(theta / 2) * 180 / pi;      
			//y = f*2tan(theta/2)	theta = theta / 1.2;	
			theta = atan(theta) * 180 / CV_PI;
			int index = cvFloor(theta / fovStep);
			if (index > 2001) return;
			if (index < 0) return;
			double realDist = realHeight[index] / cmosPixelSize;
			u = center1.x + realDist*(i - center.x) / referHeight;
			v = center1.y + realDist*(j - center.y) / referHeight;

			if (u>0 && u<(Src.cols - 1) && v<(Src.rows - 1)){
				mapx.at<Vec2f>(j, i)[0] = float(u);
				mapx.at<Vec2f>(j, i)[1] = float(v);
			}
			else{
				mapx.at<Vec2f>(j, i)[0] = 0;
				mapx.at<Vec2f>(j, i)[1] = 0;
			}

		}
	}
	cv::remap(Src, Dst, mapx, Mat(), INTER_LINEAR);
}
