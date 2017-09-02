// ParkingSlotDect.cpp : �������̨Ӧ�ó������ڵ㡣
//
#include "stdafx.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <algorithm>
#include "mytools.h  "
#include <stdio.h>
#include <iostream>
#include <time.h>
using namespace cv;
using namespace std;
using std::vector;

vector<Point2i> CenterPoints0;//����"T"�νṹ���������

int main(int argc, const char** argv)
{
	Mat src; //����ͼ��
	cout << "This is a demo for Parking slot detection." << endl;
	cout << "��ʼ����ͼ��..." << endl;
	string filename = "Img\\birdView8.png";//ͼ��·��λ�� "Img\\birdView0015.png"   calib\\_70.png
	src = imread(filename, -1);//�������ͼ��
	if (src.empty())//���ܶ�ȡͼ��
	{
		printf("Cannot read image file: %s\n", filename.c_str());
		return -1;
	}
	namedWindow("SrcImg");//�������ڣ���ʾԭʼͼ��
	imshow("SrcImg", src);
	waitKey(0);

	//ͼ���Ϊ����ͼ
	//Mat BirdView; //����ͼ��
	//cout << "���и���ͼ�任..." << endl;
	//BirdView.create(src.size(), src.type());
	//TansformToBirdView(src, BirdView, 359.8189, 361.4601, 687.5108, 474.5289, 27, 1);
	//waitKey(0);

	//ԭͼ��ľֲ���ֵ
	cout << "���оֲ�����Ӧ��ֵ����ֵ��...";
	clock_t start, end;//���ڼ�ʱ
	start = clock();
	Mat binary, gray;
	gray.create(src.size(), 1); binary.create(src.size(), 1);
	cvtColor(src, gray, COLOR_BGR2GRAY);//��Ϊ�Ҷ�ͼ
	adaptiveThreshold(gray, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 23, -5);//�ֲ�����Ӧ��ֵ��ֵ��
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("BinaryShow1");//�������ڣ���ʾ��ֵͼ
	imshow("BinaryShow1", binary);
	waitKey(0);


	//��Mask�����߽�
	cout << "��Mask�����߽�..." << endl;
	filename = "mask720.png";//ͼ��·��λ��  mask960.png
	Mat maskBird = imread(filename, -1);//���������߽����ĤMASK
	Mat element = getStructuringElement(MORPH_RECT, Size(21, 21));//��ȡ�Զ����
	Mat mask;
	cv::erode(maskBird, mask, element);//���и�ʴ
	if (binary.size() != mask.size())
	{
		cout << "��ĤMASK��ͼ���С����ȣ�������ѡ��MASK" << endl;
		return -1;
	}
	binary = binary&mask;//mask���ֵͼ����
	namedWindow("BinaryShow3");//�������ڣ���ʾ�µĶ�ֵͼ
	imshow("BinaryShow3", binary);
	waitKey(0);


	//��̬ѧ��������ȥ������
	cout << "��̬ѧ��������ȥ������...";
	start = clock();
	Mat element0 = getStructuringElement(MORPH_RECT, Size(3, 3));//��ȡ�Զ����
	Mat outputMat;
	outputMat.create(binary.size(), binary.type());//��̬ѧ�����Ķ�ֵͼ
	cv::erode(binary, outputMat, element0);//���и�ʴ	
	dilate(outputMat, outputMat, element0);//��������
	element0 = getStructuringElement(MORPH_RECT, Size(5, 5));//��ȡ�Զ����
	dilate(outputMat, outputMat, element0);//��������
	erode(outputMat, outputMat, element0);//���и�ʴ
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("BinaryShow2");//�������ڣ���ʾ��̬ѧ�����Ķ�ֵͼ
	imshow("BinaryShow2", outputMat);
	waitKey(0);


	//����������ȥ�����С����
	//cout << "����������ȥ�����С�������ն�..." ;
	//Mat RemoveBinary;
	//RemoveBinary.create(binary.size(), binary.type());//ȥ�����С�����Ķ�ֵͼ
	//start = clock();
	//RemoveSmallRegion(outputMat, RemoveBinary, 600, 1, 1);//ȥ��С����
	//RemoveSmallRegion(RemoveBinary, RemoveBinary, 100, 0, 0);//���ն�
	//end = clock();
	//cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	//namedWindow("RemoveSmallRegion", CV_WINDOW_AUTOSIZE);
	//imshow("RemoveSmallRegion", RemoveBinary);
	//imwrite("binaryImg2.png", RemoveBinary);
	//waitKey(0);

	//����һ��ȥ��С�������ķ��������������ɾ�������С������
	cout << "������ⷨȥ�����С����...";
	start = clock();
	Mat RemoveBinary(binary.size(), binary.type(), Scalar(0));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(outputMat,
		contours, // a vector of contours 
		CV_RETR_EXTERNAL, // ֻ���������
		CHAIN_APPROX_NONE); // retrieve all pixels of each contours
	for (int i = 0; i<contours.size(); i++)
	{
		double tmparea = fabs(contourArea(contours[i]));
		if (contours[i].size()>60)
		{
			drawContours(RemoveBinary, contours, i, Scalar(255), CV_FILLED);//��������
		}
	}
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("RemoveSmallRegion2", CV_WINDOW_AUTOSIZE);
	imshow("RemoveSmallRegion2", RemoveBinary);
	waitKey(0);


	//ͼ��ϸ��
	cout << "ͼ��ϸ��...";
	Mat ThinImg;
	ThinImg.create(binary.size(), binary.type());//ͼ��ϸ�����ϸ��ͼ��
	start = clock();
	ThinnerRosenfeld(RemoveBinary, ThinImg);//����ͼ��ϸ��
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("ThinnerRosenfeld", CV_WINDOW_AUTOSIZE);
	imshow("ThinnerRosenfeld", ThinImg);///��ʾϸ��ͼ��
	waitKey(0);

	cout << "ͣ��λ���ɼ�ɸѡ..." << endl;
	start = clock();
	//�ȶ� CenterPoints0���о���
	vector<Point2i> CenterPoints1;//���ڴ洢����������
	vector<int> labels0;//�����ǩ
	randShuffle(CenterPoints0, 1);
	int count = partition(CenterPoints0, labels0, _EqPredicate1);//�����н�����о��࣬����һ������Ľ��㽫���ֵ
	for (int i = 0; i < count; i++)
	{
		int x = 0, y = 0, n = 0;
		for (int j = 0; j < CenterPoints0.size(); j++)
		{
			if (labels0[j] == i)
			{
				x = x + CenterPoints0[j].x;
				y = y + CenterPoints0[j].y;
				++n;
			}
		}
		CvPoint point_mean;
		point_mean.x = x / n;
		point_mean.y = y / n;
		Point2i P2i(cvRound(x / n), cvRound(y / n));
		CenterPoints1.push_back(P2i);//��ÿ�������Ӧ�ķ���������
	}

	//���С�T���νṹ������ɸѡ����ȡ����ȷ�����������
	vector<Point3i> CenterPoints;//����"T"�νṹ�����ģ�����ά�����ǩ��������ʾ�Ƿ������ճ�λ�ġ�T���νṹ
	vector<Point3f> vPoints;//ÿ��ƥ�����Ķ�Ӧ������ֵ������ά�����ǩ
	vector<Mat> channels;
	split(src, channels);//��ͨ������
	for (int j = 0; j < CenterPoints1.size(); j += 1)
	{
		Point3i centerPoint;//����ÿһ����ѡƥ������
		centerPoint.x = CenterPoints1[j].x;
		centerPoint.y = CenterPoints1[j].y;

		//����ԭͼ��ƥ���������λ�ô���������Ϣ��ƥ��������ɸѡ
		Mat csAvg, csSdv;//��ֵ�ͷ���
		meanStdDev(src, csAvg, csSdv);//�����ֵ�ͱ�׼��
		uchar B = src.at<Vec3b>(centerPoint.y, centerPoint.x)[0];
		uchar G = src.at<Vec3b>(centerPoint.y, centerPoint.x)[1];
		uchar R = src.at<Vec3b>(centerPoint.y, centerPoint.x)[2];
		uchar* pAvg, *pSdv;
		pAvg = csAvg.ptr<uchar>(0);
		pSdv = csAvg.ptr<uchar>(0);
		if (float(R)>(pAvg[0] * 0.2 + pSdv[0] * 0.1) && float(G)>(pAvg[1] * 0.2 + pSdv[1] * 0.1) && float(B)>(pAvg[2] * 0.2 + pSdv[2] * 0.1) && (abs(R - G)<30))
		{
			circle(src, cvPoint(centerPoint.x, centerPoint.y), 8, CV_RGB(0, 0, 255), 2);//����ƥ���������

			vector<Point> vectorPoints;//���н���
			vector<int> labels;//�����ǩ

			//���ÿ��ƥ�������������ĵ�
			int startY, startX, ROIwidth, ROIheight;

			startY = centerPoint.y - 20;//�������Ͻ�����  
			startX = centerPoint.x - 20;
			ROIwidth = 40;//����ROI��С
			ROIheight = 40;
			startY = startY < 0 ? 0 : startY;//������Խ��  
			startX = startX < 0 ? 0 : startX;
			ROIwidth = (centerPoint.x + 20) < outputMat.cols ? 40 : (outputMat.cols - startX);
			ROIheight = (centerPoint.y + 20) < outputMat.rows ? 40 : (outputMat.rows - startY);
			Mat ROI = outputMat(Rect(startX, startY, ROIwidth, ROIheight));//����ROI
			// rectangle���������� ͼƬ�� ���Ͻǣ� ���½ǣ� ��ɫ�� ������ϸ�� �������ͣ�������    
			rectangle(src, cvPoint(startX, startY), cvPoint(startX + ROIwidth, startY + ROIheight), cvScalar(0, 255, 0), 3, 4, 0);//��ROI����

			for (int i = 0; i < ROI.cols; i++){
				for (int j = 0; j < ROI.rows; j++){
					if (ThinImg.ptr<uchar>(startY + j)[startX + i] > 0){//���ж��Ƿ���ͼ��Ǽ��ϵ�һ��
						//��һ��ֹͣ������ԣ������ǰ��ĺᡢ�����굽���ĵĺ�������Ĳ�ֵ����sqrt(diantance2)���������һ��

						int distance2 = (centerPoint.x - startX - i)*(centerPoint.x - startX - i)
							+ (centerPoint.y - startY - j)*(centerPoint.y - startY - j);//�����ĵľ���(int)cvGet2D(ThinImg, startY + j, startX + i).val[0]
						if (distance2 <= 225 && distance2 >= 100){//����10<=d<=15
							CvPoint vectorPoint;
							vectorPoint.x = startX + i;
							vectorPoint.y = startY + j;
							vectorPoints.push_back(vectorPoint);//�����������ķ�������
							//cvCircle(src, vectorPoint, 8, CV_RGB(0, 0, 255), 2);
						}
					}
				}
			}//end �����ɸѡ

			if (vectorPoints.size()>0)
			{
				randShuffle(vectorPoints, 1);
				int count = partition(vectorPoints, labels, _EqPredicate2);//�����н�����о��࣬����һ������Ľ��㽫���ֵ
				if (count == 3){//ֻ�����㡰T���νṹ�ķ���������ȷ�Ľṹ��
					for (int i = 0; i < count; i++)
					{
						int x = 0, y = 0, n = 0;
						for (int j = 0; j < vectorPoints.size(); j++)
						{
							if (labels[j] == i)
							{
								x = x + vectorPoints[j].x;
								y = y + vectorPoints[j].y;
								++n;
							}
						}
						CvPoint point_mean;
						point_mean.x = x / n;
						point_mean.y = y / n;
						circle(src, point_mean, 8, CV_RGB(0, 0, 255), 2);//�������

						Point3f P3f(x / n, y / n, 0);
						vPoints.push_back(P3f);//��ÿ�������Ӧ�ķ���������
					}
					centerPoint.z = 0;
					CenterPoints.push_back(centerPoint);//��ɸѡ�������Ĵ�����
				}
			}

		}//end if
	}//end ����ɸѡ

	//ȷ����T���νṹ�����򣨼���֮ǰȷ��������ÿ����T���νṹ������������������˴�֮����е㣬
	//���е㴦�Ķ�ֵͼ����ֵ����0��˵�����������ǡ�T���νṹ����һ���ϵ������㣩
	for (int n = 0; n < CenterPoints.size(); n++){
		for (int i = 0; i < 3; i++){
			for (int j = i + 1; j < 3; j++){
				Point2f P2f((vPoints[3 * n + i].x + vPoints[3 * n + j].x) / 2, (vPoints[3 * n + i].y + vPoints[3 * n + j].y) / 2);
				//circle(src, P2f, 8, CV_RGB(0, 255, 0), 2);//��ÿ���������������˴�֮����е�
				uchar* pBinary;
				pBinary = binary.ptr<uchar>(int(P2f.y));
				if (pBinary[int(P2f.x)]>'0')
				{
					++vPoints[3 * n + i].z;
					++vPoints[3 * n + j].z;
				}
			}
		}
	}

	vector<Point3i> NormalVector;//����ƥ�����Ĵ���������ĵ�
	vector<Point3i> PerpendicularToNV;//��������֮��ĵ�
	for (int n = 0; n < CenterPoints.size(); n++){
		for (int i = 0; i < 3; i++){
			if (vPoints[3 * n + i].z == 0)
			{
				CenterPoints[n].z = 1;
				NormalVector.push_back(Point3i(vPoints[3 * n + i].x, vPoints[3 * n + i].y, n));//��������������
				circle(src, Point2i(vPoints[3 * n + i].x, vPoints[3 * n + i].y), 8, CV_RGB(255, 0, 0), 2);
				if (i<2){
					PerpendicularToNV.push_back(Point3i(vPoints[3 * n + i + 1].x, vPoints[3 * n + i + 1].y, n));//��������֮��ĵ������
				}
				else{
					PerpendicularToNV.push_back(Point3i(vPoints[3 * n + i - 1].x, vPoints[3 * n + i - 1].y, n));//��������֮��ĵ������
				}
				break;
			}
			else if (i == 2)
			{
				NormalVector.push_back(Point3i(0, 0, 0));
				PerpendicularToNV.push_back(Point3i(0, 0, 0));
			}
		}
	}//end ƥ�����Ĵ���������ĵ���ȡ


	//ȷ������ͣ��λ
	for (int i = 0; i < CenterPoints.size(); i++)
	{
		if (CenterPoints[i].z)
		{
			for (int j = i + 1; j < CenterPoints.size(); j++)
			{
				if (CenterPoints[j].z)
				{
					int distance2 = (CenterPoints[j].x - CenterPoints[i].x)*(CenterPoints[j].x - CenterPoints[i].x)
						+ (CenterPoints[j].y - CenterPoints[i].y)*(CenterPoints[j].y - CenterPoints[i].y);
					if (distance2 >= 10000 && distance2 <= 25921 && NormalVector[i].x != 0 && NormalVector[j].x != 0)
					{//����T�νṹ�ľ��뷶Χ100~161
						if (NormalVector[i].z == i&&NormalVector[j].z == j)
						{
							//������ֵ�����ж�����T�νṹ���������Ƿ�һ�£���������������ƽ�У�AB=x1*y2-x2*y1=0����Ƕ�ֵ���0�Ȼ�360��
							float OA = fastAtan2(NormalVector[i].y - CenterPoints[i].y, NormalVector[i].x - CenterPoints[i].x);//�����һ����������Ƕ�,��������������ǰѷ�������ת��Ϊ�Ƕ�ֵ
							float OB = fastAtan2(NormalVector[j].y - CenterPoints[j].y, NormalVector[j].x - CenterPoints[j].x);

							//int AB = (NormalVector[i].x - CenterPoints[i].x)*(NormalVector[j].y - CenterPoints[j].y)
							//	- (NormalVector[j].x - CenterPoints[j].x)*(NormalVector[i].y - CenterPoints[i].y);//�����ֱ�Ӹ���AB��ֵ�ж�
							if (abs(OA - OB) <= 10 || (abs(OA - OB) <= 370 && abs(OA - OB) >= 350)){
								//if (abs(AB) <= 80){
								//��һ��������T�νṹ������������������ƴ�ֱ(CD1=x1*x2+y1*y2=0����Ƕ�ֵ�ӽ�90��)
								//(��ڶ��ֲ��ԣ��������ߵ������벻����������������������������ƽ��:CD2=x1*y2-x2*y1=0����Ƕ�ֵ���0�Ȼ�360��)
								int CD1 = (CenterPoints[j].x - CenterPoints[i].x)*(NormalVector[i].x - CenterPoints[i].x)
									+ (CenterPoints[j].y - CenterPoints[i].y)*(NormalVector[i].y - CenterPoints[i].y);
								//int CD2 = (CenterPoints[j].x - CenterPoints[i].x)*(PerpendicularToNV[j].y - PerpendicularToNV[i].y)
								//	- (PerpendicularToNV[j].x - PerpendicularToNV[i].x)*(CenterPoints[j].y - CenterPoints[i].y);
								OA = fastAtan2(CenterPoints[j].y - CenterPoints[i].y, CenterPoints[j].x - CenterPoints[i].x);//�����ṹ���ĵ�����
								OB = fastAtan2(PerpendicularToNV[j].y - PerpendicularToNV[i].y, PerpendicularToNV[j].x - PerpendicularToNV[i].x);
								if (abs(OA - OB) <= 10 || (abs(OA - OB) <= 370 && abs(OA - OB) >= 350)){
									if (abs(CD1) <= 300){

										line(src, cvPoint(CenterPoints[i].x, CenterPoints[i].y), cvPoint(NormalVector[i].x, NormalVector[i].y),
											CV_RGB(255, 0, 0), 2);
										line(src, cvPoint(CenterPoints[j].x, CenterPoints[j].y), cvPoint(NormalVector[j].x, NormalVector[j].y),
											CV_RGB(255, 0, 0), 2);
										line(src, cvPoint(CenterPoints[i].x, CenterPoints[i].y), cvPoint(CenterPoints[j].x, CenterPoints[j].y),
											CV_RGB(255, 0, 0), 2);
										cout << "��λ����ڸ���ͼ�е�����λ���ǣ�" << endl;
										cout << "��" << CenterPoints[i].x << "��" << CenterPoints[i].y << "��" << "    "
											<< "��" << CenterPoints[j].x << "��" << CenterPoints[j].y << "��" << endl;
										cout << "��λ����ڸ���ͼ�е����ؿ�" << endl;
										cout << sqrt(distance2) << "    ������" << endl;
										//ӳ�䵽����ͼ֮ǰ��ͼ����
										Mat testImage = imread("calib/calib42.bmp");
										circle(testImage, MapToSrc(CenterPoints[i].x, CenterPoints[i].y, testImage.cols, testImage.rows), 8, CV_RGB(0, 0, 255), 2);
										circle(testImage, MapToSrc(CenterPoints[j].x, CenterPoints[j].y, testImage.cols, testImage.rows), 8, CV_RGB(0, 0, 255), 2);
										line(testImage, MapToSrc(CenterPoints[i].x, CenterPoints[i].y, testImage.cols, testImage.rows),
											MapToSrc(CenterPoints[j].x, CenterPoints[j].y, testImage.cols, testImage.rows), CV_RGB(255, 0, 0), 2);
										namedWindow("ResultImg");
										imshow("ResultImg", testImage);
									}
								}
							}
						}
					}
				}
			}
		}
	}
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("ParkingSlotImg");//�������ڣ���ʾ��̬ѧ�����Ķ�ֵͼ
	imshow("ParkingSlotImg", src);
	waitKey(0);

	return 0;
}

