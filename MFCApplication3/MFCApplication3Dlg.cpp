
// MFCApplication3Dlg.cpp : 實作檔
//

#include "stdafx.h"
#include "MFCApplication3.h"
#include "MFCApplication3Dlg.h"
#include "afxdialogex.h"
#include<string.h>
#include "opencv2\highgui\highgui.hpp"
#include "opencv\cv.hpp"
#include "opencv2\opencv.hpp"
#include <cmath>
# define M_PI 3.14159265358979323846/* pi */
using namespace cv;
//#include<stdio.h>
using namespace std;
#ifdef _DEBUG
#define new DEBUG_NEW
#endif
Mat img;
Mat mirror;
Mat output;
Point2f src[4];
int index;
// 對 App About 使用 CAboutDlg 對話方塊

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 對話方塊資料
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支援

// 程式碼實作
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CMFCApplication3Dlg 對話方塊



CMFCApplication3Dlg::CMFCApplication3Dlg(CWnd* pParent /*=NULL*/)
	: CDialogEx(IDD_MFCAPPLICATION3_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CMFCApplication3Dlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CMFCApplication3Dlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CMFCApplication3Dlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CMFCApplication3Dlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CMFCApplication3Dlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CMFCApplication3Dlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CMFCApplication3Dlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CMFCApplication3Dlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CMFCApplication3Dlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CMFCApplication3Dlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CMFCApplication3Dlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CMFCApplication3Dlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &CMFCApplication3Dlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &CMFCApplication3Dlg::OnBnClickedButton12)
	ON_EN_CHANGE(IDC_EDIT2, &CMFCApplication3Dlg::OnEnChangeEdit2)
END_MESSAGE_MAP()


// CMFCApplication3Dlg 訊息處理常式

BOOL CMFCApplication3Dlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 將 [關於...] 功能表加入系統功能表。

	// IDM_ABOUTBOX 必須在系統命令範圍之中。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != NULL)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 設定此對話方塊的圖示。當應用程式的主視窗不是對話方塊時，
	// 框架會自動從事此作業
	SetIcon(m_hIcon, TRUE);			// 設定大圖示
	SetIcon(m_hIcon, FALSE);		// 設定小圖示

	// TODO: 在此加入額外的初始設定
	//AllocConsole();
	//freopen("CONOUT$", "w", stdout);

	return TRUE;  // 傳回 TRUE，除非您對控制項設定焦點
}

void CMFCApplication3Dlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果將最小化按鈕加入您的對話方塊，您需要下列的程式碼，
// 以便繪製圖示。對於使用文件/檢視模式的 MFC 應用程式，
// 框架會自動完成此作業。

void CMFCApplication3Dlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 繪製的裝置內容

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 將圖示置中於用戶端矩形
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 描繪圖示
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 當使用者拖曳最小化視窗時，
// 系統呼叫這個功能取得游標顯示。
HCURSOR CMFCApplication3Dlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CMFCApplication3Dlg::OnBnClickedButton1()//l.1
{
	img = imread("./img/dog.bmp");
	AllocConsole();
	freopen("CONOUT$", "w", stdout);
	cout << "Height = " << img.rows << endl;
	cout << "Width = " <<img.cols << endl;
	namedWindow("1");
	imshow("1",img);
	waitKey(0);
	img.release();
	destroyWindow("1");
}

void RGB_trans(Mat* img) {
	int height = img->rows;
	int width = img->cols;
	int i, j;
	int R;
	int G;
	int B;
	for (i = 0; i < height; i++) {
		for ( j = 0; j < width; j++) {
			B = img->at<Vec3b>(i, j)[0];
			G = img->at<Vec3b>(i, j)[1];
			R = img->at<Vec3b>(i, j)[2];
			img->at<Vec3b>(i, j)[0] = G;
			img->at<Vec3b>(i, j)[1] = R;
			img->at<Vec3b>(i, j)[2] = B;
		}
	}
}

void CMFCApplication3Dlg::OnBnClickedButton2()//1.2
{
	// TODO: 在此加入控制項告知處理常式程式碼
	img = imread("./img/color.png");
	Mat copy = img.clone();
	RGB_trans(&copy);
	namedWindow("Mix Color");
	imshow("Mix Color",copy);
	waitKey(0);
	img.release();
	copy.release();
	destroyWindow("Mix Color");
}


void CMFCApplication3Dlg::OnBnClickedButton3()//1.3
{
	img = imread("./img/dog.bmp");
	Mat copy = img.clone();
	flip(img,copy,1);
	namedWindow("flip");
	imshow("flip", copy);
	waitKey(0);
	img.release();
	copy.release();
	destroyWindow("flip");
}

void text(int i, void*)
{
	double alpha = (double)i/ 100;
	double beta;
	beta = 1 - alpha;
	addWeighted(img, alpha, mirror, beta, 0.0, output);
	imshow("BLENDINGG", output);
}

void CMFCApplication3Dlg::OnBnClickedButton4()//1.4
{	
	int i = 0;
	img = imread("./img/dog.bmp");
	mirror = img.clone();
	output = img.clone();
	flip(img, mirror, 1);
	namedWindow("BLENDINGG");
	createTrackbar("BLEND ", "BLENDINGG", &i, 100, text);
	text(0, 0);	
	waitKey(0);
	img.release();
	mirror.release();
	output.release();
	destroyWindow("BLENDINGG");
}


void CMFCApplication3Dlg::OnBnClickedButton5()//2.1
{
	// TODO: 在此加入控制項告知處理常式程式碼
	img = imread("./img/QR.png");
	cvtColor(img,output,COLOR_BGR2GRAY);
	namedWindow("Oringinal image");
	namedWindow("Threshold image");
	imshow("Oringinal image", img);
	threshold(output, output, 80, 255,THRESH_BINARY);
	imshow("Threshold image", output);
	waitKey(0);
	img.release();
	output.release();
	destroyWindow("Oringinal image");
	destroyWindow("Threshold image");
}


void CMFCApplication3Dlg::OnBnClickedButton6()//2.2
{
	// TODO: 在此加入控制項告知處理常式程式碼

	img = imread("./img/QR.png");
	cvtColor(img, output, COLOR_BGR2GRAY);
	namedWindow("Oringinal image");
	imshow("Oringinal image", img);
	adaptiveThreshold(output,output,255, ADAPTIVE_THRESH_MEAN_C , THRESH_BINARY,19,-1);
	imshow("Adaptive threshold image", output);
	waitKey(0);
	img.release();
	output.release();
	destroyWindow("Oringinal image");
	destroyWindow("Adaptive threshold image");
}



void CMFCApplication3Dlg::OnBnClickedButton7()//3.1
{
	CString angle, scale, tx, ty;
	GetDlgItem(IDC_EDIT1) -> GetWindowTextW(angle);
	GetDlgItem(IDC_EDIT2) -> GetWindowTextW(scale);
	GetDlgItem(IDC_EDIT3) -> GetWindowTextW(tx);
	GetDlgItem(IDC_EDIT4) -> GetWindowTextW(ty);
	double angle_double = (double)_ttof(angle);
	double scale_double = (double)_ttof(scale);
	double tx_double = (double)_ttof(tx);
	double ty_double = (double)_ttof(ty);
	img = imread("./img/OriginalTransform.png");
	namedWindow("Oringinal Image");
	imshow("Oringinal Image",img);
	Mat t_mat = Mat::zeros(2, 3, CV_32FC1);
	t_mat.at<float>(0, 0) = 1;
	t_mat.at<float>(0, 2) = (float)tx_double;//水平平移量
	t_mat.at<float>(1, 1) = 1;
	t_mat.at<float>(1, 2) = (float)ty_double;
	warpAffine(img,output,t_mat,img.size());
	Point2f temp = Point2f(130 + (float)tx_double, 125 + (float)ty_double);
	t_mat = getRotationMatrix2D(temp, angle_double, scale_double);
	warpAffine(output, output, t_mat, output.size());
	namedWindow("Rotation + Scale + Translation Image");
	imshow("Rotation + Scale + Translation Image", output);
	waitKey(0);
	img.release();
	t_mat.release();
	output.release();
	destroyWindow("Rotation + Scale + Translation Image");
	destroyWindow("Oringinal Image");
}

void my_mouse_callback(int event,int x,int y,int flags,void* param) {
	if (event == EVENT_LBUTTONDOWN && index <4) {
		src[index] = Point2f((float)x, (float)y);
		index++;
	}
	else if (index == 4) {
		index = 0;
		output.rows = 450;
		output.cols = 450;
		Point2f des[4];
		des[0] = Point2f(20, 20);
		des[1] = Point2f(450, 20);
		des[2] = Point2f(450, 450);
		des[3] = Point2f(20, 450);
		Mat transmatrix;
		transmatrix = Mat::zeros(img.rows, img.cols, img.type());
		transmatrix = getPerspectiveTransform(src, des);
		warpPerspective(img, output, transmatrix, output.size());
		namedWindow("Perspective Result Image");
		imshow("Perspective Result Image", output);
		waitKey(0);
		transmatrix.release();
		output.release();
		destroyWindow("Perspective Result Image");
	}

}




void CMFCApplication3Dlg::OnBnClickedButton8()//3.2
{
	index = 0;
	img = imread("./img/OriginalPerspective.png");
	namedWindow("OriginalPerspective");
	imshow("OriginalPerspective", img);
	setMouseCallback("OriginalPerspective", my_mouse_callback, NULL);
	waitKey(0);
	img.release();
	destroyWindow("OriginalPerspective");
}


Mat RGBtoGRAY(Mat image) {
	int height = image.rows;
	int width = image.cols;
	int i, j;
	int R;
	int G;
	int B;
	output = Mat(height, width, CV_8U, Scalar(0));
	for (i = 0; i < height; i++) {
		for ( j = 0; j < width; j++) {
			B = image.at<Vec3b>(i, j)[0];
			G = image.at<Vec3b>(i, j)[1];
			R = image.at<Vec3b>(i, j)[2];
			output.at<uchar>(i,j) = (int)(0.299*R + 0.587*G + 0.114*B);
		}
	}
	return output;
}

Mat gaussian_smooth(Mat image) {
	double kernel[3][3]; 
	double sigma = 1.0;
	double r, s = 2.0 * sigma * sigma;
	double sum = 0.0;
	int x, y; 
	output = Mat(image.rows, image.cols, CV_8U, Scalar(0));
	for (x = 0; x < output.rows; x++) {
		for (y = 0; y < output.cols; y++) {
			output.at<uchar>(x, y) = 0;
		}
	}
	for ( x = -1; x <= 1; x++) {
		for ( y = -1; y <= 1; y++) {
			r = sqrt(x * x + y * y);
			kernel[x + 1][y + 1] = (exp(-(r * r) / s)) / ( M_PI * s);
			sum += kernel[x + 1][y + 1];
		}
	}
	for ( x = 0; x < 3; ++x)
		for (y = 0; y < 3; ++y) 
			kernel[x][y] /= sum;
		




	for (x = 1; x < output.rows-1; x++) {
		for (y = 1; y < output.cols-1; y++) {
			output.at<uchar>(x, y) = (int)(kernel[0][0]*image.at<UCHAR>(x-1,y-1) + kernel[0][1] * image.at<UCHAR>(x - 1, y) + kernel[0][2] * image.at<UCHAR>(x - 1, y + 1)+
											kernel[1][0] * image.at<UCHAR>(x, y - 1) + kernel[1][1] * image.at<UCHAR>(x, y) + kernel[1][2] * image.at<UCHAR>(x , y + 1)+
					kernel[2][0] * image.at<UCHAR>(x + 1, y - 1) + kernel[2][1] * image.at<UCHAR>(x + 1, y) + kernel[2][2] * image.at<UCHAR>(x + 1, y + 1));

		}
	}


	return output;
}


float calculateSD(float* data,float* mean,int total)
{
	float sum = 0.0,  standardDeviation = 0.0;
	*mean = 0;
	int i;
	for (i = 0; i < total; ++i)
	{
		sum += data[i];
	}
	*mean = sum / total;
	for (i = 0; i < total; ++i) {
		standardDeviation += pow(data[i] - (*mean), 2);
		//cout << data[i] << endl;
	}
	return sqrt(standardDeviation / total);
}


void CMFCApplication3Dlg::OnBnClickedButton9()//4.1
{
	// TODO: 在此加入控制項告知處理常式程式碼
	img = imread("./img/School.jpg");
	Mat temp = RGBtoGRAY(img);
	output = gaussian_smooth(temp);
	namedWindow("Gaussian");
	imshow("Gaussian", output);
	waitKey(0);
	temp.release();
	output.release();
	destroyWindow("Gaussian");
}

Mat sobel(Mat image,int x_y) {
	int kernel[3][3];
	if (x_y == 0) {
		kernel[0][0] = -1;
		kernel[0][1] =  0;
		kernel[0][2] = 1;
		kernel[1][0] = -2;
		kernel[1][1] =  0;
		kernel[1][2] =  2;
		kernel[2][0] = -1;
		kernel[2][1] =  0;
		kernel[2][2] = 1;

	}
	else {
		kernel[0][0] = -1;
		kernel[0][1] = -2;
		kernel[0][2] = -1;
		kernel[1][0] = 0;
		kernel[1][1] = 0;
		kernel[1][2] = 0;
		kernel[2][0] = 1;
		kernel[2][1] = 2;
		kernel[2][2] = 1;
	}
	int x, y;
	output = Mat(image.rows, image.cols, CV_8U, Scalar(0));
	float *temp_array = new float[image.rows*image.cols];

	for (x = 0; x < output.rows; x++) {
		for (y = 0; y < output.cols; y++) {
			output.at<uchar>(x, y) = 0;
			
		}
	}

	float mean;
	float sd;
	int total = 0;
	for (x = 1; x < output.rows - 1; x++) {
		for (y = 1; y < output.cols - 1; y++) {
			temp_array[total] = (float)(kernel[0][0] * image.at<UCHAR>(x - 1, y - 1) + kernel[0][1] * image.at<UCHAR>(x - 1, y) + kernel[0][2] * image.at<UCHAR>(x - 1, y + 1) +
				kernel[1][0] * image.at<UCHAR>(x, y - 1) + kernel[1][1] * image.at<UCHAR>(x, y) + kernel[1][2] * image.at<UCHAR>(x, y + 1) +
				kernel[2][0] * image.at<UCHAR>(x + 1, y - 1) + kernel[2][1] * image.at<UCHAR>(x + 1, y) + kernel[2][2] * image.at<UCHAR>(x + 1, y + 1));
			total++;
		}
	}
	sd = calculateSD(temp_array,&mean,total);
	total = 0;
	for (x = 1; x < image.rows-1; x++) {
		for (y = 1; y < image.cols-1; y++) {
			output.at<UCHAR>(x, y) = (int)((temp_array[total] - mean) / sd )* (255);
			total++;

		}
	}


	delete[] temp_array;
	


	return output;
}


void CMFCApplication3Dlg::OnBnClickedButton10()//sobel; x
{
	img = imread("./img/School.jpg");
	Mat temp = RGBtoGRAY(img);
	temp = gaussian_smooth(temp);
	output = sobel(temp,0);
	namedWindow("absX");
	imshow("absX", output);
	waitKey(0);
	temp.release();
	img.release();
	output.release();
	destroyWindow("absX");
}


void CMFCApplication3Dlg::OnBnClickedButton11()//sobel y
{
	img = imread("./img/School.jpg");
	Mat temp = RGBtoGRAY(img);
	temp = gaussian_smooth(temp);
	output = sobel(temp, 1);
	namedWindow("absY");
	imshow("absY", output);
	waitKey(0);
	temp.release();
	img.release();
	output.release();
	destroyWindow("absY");
}

Mat magnititude(Mat sobel_x,Mat sobel_y) {
	output = Mat(sobel_x.rows, sobel_x.cols, CV_8U, Scalar(0));
	float* temp_array = new float [sobel_x.rows*sobel_x.cols];
	int x, y;
	int total = 0;
	float mean;
	float sd;


	for (x = 1; x < output.rows-1; x++) {
		for (y = 1; y < output.cols-1; y++) {
			temp_array[total] = (float)((sobel_x.at<UCHAR>(x, y)*sobel_x.at<UCHAR>(x, y)) + sobel_y.at<UCHAR>(x, y)*sobel_y.at<UCHAR>(x, y));
			temp_array[total] = sqrt(temp_array[total]);
		
			total++;
			
		}
	}
	sd = calculateSD(temp_array,&mean,total);
	total = 0;

	for (x = 1; x < output.rows-1; x++) {
		for (y = 1; y < output.cols-1; y++) {
			output.at<UCHAR>(x, y) = (int)((temp_array[total] - mean) / sd)* (255);
			total++;
		}
	}

	delete[] temp_array;

	return output;
}

void CMFCApplication3Dlg::OnBnClickedButton12()//magnitude
{
	img = imread("./img/School.jpg");
	Mat temp = RGBtoGRAY(img);
	temp = gaussian_smooth(temp);
	Mat sobel_x = sobel(temp,0);
	sobel(temp, 1);
	Mat sobel_y = sobel(temp,1);
	output = magnititude(sobel_x, sobel_y);
	namedWindow("Result");
	imshow("Result", output);
	waitKey(0);
	temp.release();
	sobel_x.release();
	sobel_y.release();
	img.release();
	output.release();
	destroyWindow("Result");
}


void CMFCApplication3Dlg::OnEnChangeEdit2()
{
	
}
