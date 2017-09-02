// ParkingSlotDect.cpp : 定义控制台应用程序的入口点。
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

vector<Point2i> CenterPoints0;//所有"T"形结构对象的重心

int main(int argc, const char** argv)
{
	Mat src; //输入图像
	cout << "This is a demo for Parking slot detection." << endl;
	cout << "开始读入图像..." << endl;
	string filename = "Img\\birdView8.png";//图像路径位置 "Img\\birdView0015.png"   calib\\_70.png
	src = imread(filename, -1);//载入测试图像
	if (src.empty())//不能读取图像
	{
		printf("Cannot read image file: %s\n", filename.c_str());
		return -1;
	}
	namedWindow("SrcImg");//创建窗口，显示原始图像
	imshow("SrcImg", src);
	waitKey(0);

	//图像变为俯视图
	//Mat BirdView; //输入图像
	//cout << "进行俯视图变换..." << endl;
	//BirdView.create(src.size(), src.type());
	//TansformToBirdView(src, BirdView, 359.8189, 361.4601, 687.5108, 474.5289, 27, 1);
	//waitKey(0);

	//原图像的局部二值
	cout << "进行局部自适应阈值而二值化...";
	clock_t start, end;//用于计时
	start = clock();
	Mat binary, gray;
	gray.create(src.size(), 1); binary.create(src.size(), 1);
	cvtColor(src, gray, COLOR_BGR2GRAY);//变为灰度图
	adaptiveThreshold(gray, binary, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 23, -5);//局部自适应阈值二值化
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("BinaryShow1");//创建窗口，显示二值图
	imshow("BinaryShow1", binary);
	waitKey(0);


	//用Mask消除边界
	cout << "用Mask消除边界..." << endl;
	filename = "mask720.png";//图像路径位置  mask960.png
	Mat maskBird = imread(filename, -1);//用于消除边界的掩膜MASK
	Mat element = getStructuringElement(MORPH_RECT, Size(21, 21));//获取自定义核
	Mat mask;
	cv::erode(maskBird, mask, element);//进行腐蚀
	if (binary.size() != mask.size())
	{
		cout << "掩膜MASK与图像大小不相等，请重新选择MASK" << endl;
		return -1;
	}
	binary = binary&mask;//mask与二值图相与
	namedWindow("BinaryShow3");//创建窗口，显示新的二值图
	imshow("BinaryShow3", binary);
	waitKey(0);


	//形态学处理，初步去除噪声
	cout << "形态学处理，初步去除噪声...";
	start = clock();
	Mat element0 = getStructuringElement(MORPH_RECT, Size(3, 3));//获取自定义核
	Mat outputMat;
	outputMat.create(binary.size(), binary.type());//形态学处理后的二值图
	cv::erode(binary, outputMat, element0);//进行腐蚀	
	dilate(outputMat, outputMat, element0);//进行膨胀
	element0 = getStructuringElement(MORPH_RECT, Size(5, 5));//获取自定义核
	dilate(outputMat, outputMat, element0);//进行膨胀
	erode(outputMat, outputMat, element0);//进行腐蚀
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("BinaryShow2");//创建窗口，显示形态学处理后的二值图
	imshow("BinaryShow2", outputMat);
	waitKey(0);


	//区域生长法去除面积小区域
	//cout << "区域生长法去除面积小区域及填充空洞..." ;
	//Mat RemoveBinary;
	//RemoveBinary.create(binary.size(), binary.type());//去除面积小区域后的二值图
	//start = clock();
	//RemoveSmallRegion(outputMat, RemoveBinary, 600, 1, 1);//去除小区域
	//RemoveSmallRegion(RemoveBinary, RemoveBinary, 100, 0, 0);//填充空洞
	//end = clock();
	//cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	//namedWindow("RemoveSmallRegion", CV_WINDOW_AUTOSIZE);
	//imshow("RemoveSmallRegion", RemoveBinary);
	//imwrite("binaryImg2.png", RemoveBinary);
	//waitKey(0);

	//另外一种去除小面积区域的方法：检测轮廓，删除面积较小的轮廓
	cout << "轮廓检测法去除面积小区域...";
	start = clock();
	Mat RemoveBinary(binary.size(), binary.type(), Scalar(0));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(outputMat,
		contours, // a vector of contours 
		CV_RETR_EXTERNAL, // 只检测外轮廓
		CHAIN_APPROX_NONE); // retrieve all pixels of each contours
	for (int i = 0; i<contours.size(); i++)
	{
		double tmparea = fabs(contourArea(contours[i]));
		if (contours[i].size()>60)
		{
			drawContours(RemoveBinary, contours, i, Scalar(255), CV_FILLED);//画出轮廓
		}
	}
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("RemoveSmallRegion2", CV_WINDOW_AUTOSIZE);
	imshow("RemoveSmallRegion2", RemoveBinary);
	waitKey(0);


	//图像细化
	cout << "图像细化...";
	Mat ThinImg;
	ThinImg.create(binary.size(), binary.type());//图像细化后的细化图像
	start = clock();
	ThinnerRosenfeld(RemoveBinary, ThinImg);//进行图像细化
	end = clock();
	cout << "Use Time:" << ((double)(end - start) / CLOCKS_PER_SEC) << "s" << endl;
	namedWindow("ThinnerRosenfeld", CV_WINDOW_AUTOSIZE);
	imshow("ThinnerRosenfeld", ThinImg);///显示细化图像
	waitKey(0);

	cout << "停车位生成及筛选..." << endl;
	start = clock();
	//先对 CenterPoints0进行聚类
	vector<Point2i> CenterPoints1;//用于存储聚类后的重心
	vector<int> labels0;//聚类标签
	randShuffle(CenterPoints0, 1);
	int count = partition(CenterPoints0, labels0, _EqPredicate1);//对所有交点进行聚类，属于一个区域的交点将求均值
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
		CenterPoints1.push_back(P2i);//将每个对象对应的方向点存起来
	}

	//进行“T”形结构的重心筛选和提取，并确定三个方向点
	vector<Point3i> CenterPoints;//所有"T"形结构的重心，第三维代表标签，用来表示是否是最终车位的“T”形结构
	vector<Point3f> vPoints;//每个匹配重心对应的三个值，第三维代表标签
	vector<Mat> channels;
	split(src, channels);//三通道分离
	for (int j = 0; j < CenterPoints1.size(); j += 1)
	{
		Point3i centerPoint;//访问每一个候选匹配重心
		centerPoint.x = CenterPoints1[j].x;
		centerPoint.y = CenterPoints1[j].y;

		//利用原图中匹配对象重心位置处的像素信息对匹配对象进行筛选
		Mat csAvg, csSdv;//均值和方差
		meanStdDev(src, csAvg, csSdv);//计算均值和标准差
		uchar B = src.at<Vec3b>(centerPoint.y, centerPoint.x)[0];
		uchar G = src.at<Vec3b>(centerPoint.y, centerPoint.x)[1];
		uchar R = src.at<Vec3b>(centerPoint.y, centerPoint.x)[2];
		uchar* pAvg, *pSdv;
		pAvg = csAvg.ptr<uchar>(0);
		pSdv = csAvg.ptr<uchar>(0);
		if (float(R)>(pAvg[0] * 0.2 + pSdv[0] * 0.1) && float(G)>(pAvg[1] * 0.2 + pSdv[1] * 0.1) && float(B)>(pAvg[2] * 0.2 + pSdv[2] * 0.1) && (abs(R - G)<30))
		{
			circle(src, cvPoint(centerPoint.x, centerPoint.y), 8, CV_RGB(0, 0, 255), 2);//画出匹配对象重心

			vector<Point> vectorPoints;//所有交点
			vector<int> labels;//聚类标签

			//检测每个匹配对象三个方向的点
			int startY, startX, ROIwidth, ROIheight;

			startY = centerPoint.y - 20;//计算左上角坐标  
			startX = centerPoint.x - 20;
			ROIwidth = 40;//计算ROI大小
			ROIheight = 40;
			startY = startY < 0 ? 0 : startY;//不允许越界  
			startX = startX < 0 ? 0 : startX;
			ROIwidth = (centerPoint.x + 20) < outputMat.cols ? 40 : (outputMat.cols - startX);
			ROIheight = (centerPoint.y + 20) < outputMat.rows ? 40 : (outputMat.rows - startY);
			Mat ROI = outputMat(Rect(startX, startY, ROIwidth, ROIheight));//设置ROI
			// rectangle函数参数： 图片， 左上角， 右下角， 颜色， 线条粗细， 线条类型，点类型    
			rectangle(src, cvPoint(startX, startY), cvPoint(startX + ROIwidth, startY + ROIheight), cvScalar(0, 255, 0), 3, 4, 0);//画ROI区域

			for (int i = 0; i < ROI.cols; i++){
				for (int j = 0; j < ROI.rows; j++){
					if (ThinImg.ptr<uchar>(startY + j)[startX + i] > 0){//先判断是否是图像骨架上的一点
						//加一个停止计算策略：如果当前点的横、纵坐标到重心的横纵坐标的差值大于sqrt(diantance2)，则计算下一个

						int distance2 = (centerPoint.x - startX - i)*(centerPoint.x - startX - i)
							+ (centerPoint.y - startY - j)*(centerPoint.y - startY - j);//到中心的距离(int)cvGet2D(ThinImg, startY + j, startX + i).val[0]
						if (distance2 <= 225 && distance2 >= 100){//距离10<=d<=15
							CvPoint vectorPoint;
							vectorPoint.x = startX + i;
							vectorPoint.y = startY + j;
							vectorPoints.push_back(vectorPoint);//将满足条件的方向点加入
							//cvCircle(src, vectorPoint, 8, CV_RGB(0, 0, 255), 2);
						}
					}
				}
			}//end 方向点筛选

			if (vectorPoints.size()>0)
			{
				randShuffle(vectorPoints, 1);
				int count = partition(vectorPoints, labels, _EqPredicate2);//对所有交点进行聚类，属于一个区域的交点将求均值
				if (count == 3){//只有满足“T”形结构的方向点才是正确的结构点
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
						circle(src, point_mean, 8, CV_RGB(0, 0, 255), 2);//画方向点

						Point3f P3f(x / n, y / n, 0);
						vPoints.push_back(P3f);//将每个对象对应的方向点存起来
					}
					centerPoint.z = 0;
					CenterPoints.push_back(centerPoint);//将筛选过的重心存起来
				}
			}

		}//end if
	}//end 重心筛选

	//确定“T”形结构主方向（计算之前确定出来的每个“T”形结构三个方向的三个方向点彼此之间的中点，
	//若中点处的二值图像素值大于0，说明这两个点是“T”形结构上面一横上的两个点）
	for (int n = 0; n < CenterPoints.size(); n++){
		for (int i = 0; i < 3; i++){
			for (int j = i + 1; j < 3; j++){
				Point2f P2f((vPoints[3 * n + i].x + vPoints[3 * n + j].x) / 2, (vPoints[3 * n + i].y + vPoints[3 * n + j].y) / 2);
				//circle(src, P2f, 8, CV_RGB(0, 255, 0), 2);//画每个对象三个方向点彼此之间的中点
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

	vector<Point3i> NormalVector;//所有匹配对象的代表主方向的点
	vector<Point3i> PerpendicularToNV;//除主方向之外的点
	for (int n = 0; n < CenterPoints.size(); n++){
		for (int i = 0; i < 3; i++){
			if (vPoints[3 * n + i].z == 0)
			{
				CenterPoints[n].z = 1;
				NormalVector.push_back(Point3i(vPoints[3 * n + i].x, vPoints[3 * n + i].y, n));//将主方向点存起来
				circle(src, Point2i(vPoints[3 * n + i].x, vPoints[3 * n + i].y), 8, CV_RGB(255, 0, 0), 2);
				if (i<2){
					PerpendicularToNV.push_back(Point3i(vPoints[3 * n + i + 1].x, vPoints[3 * n + i + 1].y, n));//除主方向之外的点存起来
				}
				else{
					PerpendicularToNV.push_back(Point3i(vPoints[3 * n + i - 1].x, vPoints[3 * n + i - 1].y, n));//除主方向之外的点存起来
				}
				break;
			}
			else if (i == 2)
			{
				NormalVector.push_back(Point3i(0, 0, 0));
				PerpendicularToNV.push_back(Point3i(0, 0, 0));
			}
		}
	}//end 匹配对象的代表主方向的点提取


	//确定最终停车位
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
					{//两个T形结构的距离范围100~161
						if (NormalVector[i].z == i&&NormalVector[j].z == j)
						{
							//根据数值向量判断两个T形结构的主方向是否一致，即两个向量近似平行：AB=x1*y2-x2*y1=0，或角度值相差0度或360度
							float OA = fastAtan2(NormalVector[i].y - CenterPoints[i].y, NormalVector[i].x - CenterPoints[i].x);//计算第一个的主方向角度,这个函数的作用是把方向向量转换为角度值
							float OB = fastAtan2(NormalVector[j].y - CenterPoints[j].y, NormalVector[j].x - CenterPoints[j].x);

							//int AB = (NormalVector[i].x - CenterPoints[i].x)*(NormalVector[j].y - CenterPoints[j].y)
							//	- (NormalVector[j].x - CenterPoints[j].x)*(NormalVector[i].y - CenterPoints[i].y);//这个是直接根据AB的值判断
							if (abs(OA - OB) <= 10 || (abs(OA - OB) <= 370 && abs(OA - OB) >= 350)){
								//if (abs(AB) <= 80){
								//进一步，两个T形结构重心连线与主方向近似垂直(CD1=x1*x2+y1*y2=0，或角度值接近90度)
								//(或第二种策略：重心连线的向量与不是主方向点的另外两个方向点的向量平行:CD2=x1*y2-x2*y1=0，或角度值相差0度或360度)
								int CD1 = (CenterPoints[j].x - CenterPoints[i].x)*(NormalVector[i].x - CenterPoints[i].x)
									+ (CenterPoints[j].y - CenterPoints[i].y)*(NormalVector[i].y - CenterPoints[i].y);
								//int CD2 = (CenterPoints[j].x - CenterPoints[i].x)*(PerpendicularToNV[j].y - PerpendicularToNV[i].y)
								//	- (PerpendicularToNV[j].x - PerpendicularToNV[i].x)*(CenterPoints[j].y - CenterPoints[i].y);
								OA = fastAtan2(CenterPoints[j].y - CenterPoints[i].y, CenterPoints[j].x - CenterPoints[i].x);//两个结构重心的连线
								OB = fastAtan2(PerpendicularToNV[j].y - PerpendicularToNV[i].y, PerpendicularToNV[j].x - PerpendicularToNV[i].x);
								if (abs(OA - OB) <= 10 || (abs(OA - OB) <= 370 && abs(OA - OB) >= 350)){
									if (abs(CD1) <= 300){

										line(src, cvPoint(CenterPoints[i].x, CenterPoints[i].y), cvPoint(NormalVector[i].x, NormalVector[i].y),
											CV_RGB(255, 0, 0), 2);
										line(src, cvPoint(CenterPoints[j].x, CenterPoints[j].y), cvPoint(NormalVector[j].x, NormalVector[j].y),
											CV_RGB(255, 0, 0), 2);
										line(src, cvPoint(CenterPoints[i].x, CenterPoints[i].y), cvPoint(CenterPoints[j].x, CenterPoints[j].y),
											CV_RGB(255, 0, 0), 2);
										cout << "车位入口在俯视图中的坐标位置是：" << endl;
										cout << "（" << CenterPoints[i].x << "，" << CenterPoints[i].y << "）" << "    "
											<< "（" << CenterPoints[j].x << "，" << CenterPoints[j].y << "）" << endl;
										cout << "车位入口在俯视图中的像素宽：" << endl;
										cout << sqrt(distance2) << "    个像素" << endl;
										//映射到俯视图之前的图像上
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
	namedWindow("ParkingSlotImg");//创建窗口，显示形态学处理后的二值图
	imshow("ParkingSlotImg", src);
	waitKey(0);

	return 0;
}

