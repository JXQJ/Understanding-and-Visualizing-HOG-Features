#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <math.h>
#include <vector>
#include <algorithm>
#define PI 3.145
#include <fstream>
#include <string>
using namespace cv;
using namespace std;

class HOG
{
public:
    vector<float> ComputeHistogramFeatures(Mat& img, Mat& filter, int cell_rows, int cell_cols, int orientations);
    void computeGradients(Mat& img, Mat& f_grad_x, Mat&  f_grad_y, Mat& filter);
    void getGradients(Mat& img, Mat& grad, Mat& filter);
    void computeMagnitudeAngle(Mat& grad_x, Mat& grad_y, Mat& mag, Mat& ang);
    void drawHOGImage( vector<vector<vector<float> >> orient_bin_histogram, int orientation, int cell_rows,  int cell_cols);
    vector<vector<vector<float> >> getHistograms_Cell(Mat& magnitude_cell, Mat& angle_cell,int cell_rows, int cell_cols, int orientations);
    float orient_hog(Mat& mag_cell_block, Mat& orient_cell_block, int orient_start, int orient_end);
    void normalize_histograms(int i_idx, int j_idx, vector<vector<vector<float> >>orient_bin_histogram, vector<vector<vector<float> >> norm_orient_bin_histogram);
    void check_hog(string img_file);
};

float HOG::orient_hog(Mat& mag_cell_block, Mat& orient_cell_block, int orient_start, int orient_end)
{
    int cell_rows = mag_cell_block.rows;
    int cell_cols = mag_cell_block.cols;
    float total = 0;

    float* mag;
    int* orient;
    for (int i = 0; i < cell_rows; i++)
    {
        mag = mag_cell_block.ptr<float>(i);
        orient = orient_cell_block.ptr<int>(i);
        for(int j = 0; j < cell_cols; j++)
        {
            if(orient[j]<orient_start || orient[j]>=orient_end)
            {
                continue;
            }
            else
            {
               total = total +  mag[j];
            }
        }
    }
    total  = total / (cell_rows * cell_cols);

    return total ;

}

vector<vector<vector<float> >> HOG::getHistograms_Cell(Mat& magnitude_cell, Mat& angle_cell,int cell_rows, int cell_cols, int orientations){

    int img_rows = magnitude_cell.rows;
    int img_cols = magnitude_cell.cols;
    int row_blocks = int (img_rows / cell_rows);
    int col_blocks = int (img_cols / cell_cols);
    Mat mag_cell_block;
    Mat orient_cell_block;
    int start_x;
    int start_y;
    int orient_per_180 = 180 / orientations;
    int orient_start,orient_end;
    vector<vector<vector<float> >>orient_bin_histogram(row_blocks,vector<vector<float> >(col_blocks,vector<float>(orientations,0)));


    for (int i = 0 ; i < row_blocks; i++)
    {
       for (int j = 0; j < col_blocks; j++)
       {
           start_x = j * cell_cols;
           start_y = i * cell_rows;
           mag_cell_block = Mat(magnitude_cell,Rect(start_x, start_y, cell_cols, cell_rows));
           orient_cell_block = Mat(angle_cell, Rect (start_x, start_y, cell_cols, cell_rows));

           for  (int k = 0 ; k < orientations; k++)
           {
               orient_start = k * orient_per_180;
               orient_end = (k + 1)  * orient_per_180;
               orient_bin_histogram[i][j][k] = orient_hog(mag_cell_block, orient_cell_block, orient_start, orient_end);

           }
       }
    }
    return orient_bin_histogram;
}


void HOG::normalize_histograms(int i_idx, int j_idx, vector<vector<vector<float> >>orient_bin_histogram, vector<vector<vector<float> >> norm_orient_bin_histogram)
{

    int orientations = orient_bin_histogram[0][0].size();
    float norm = 0;
    for (int k = 0 ; k < orientations; k++)
    {
        norm  = norm + pow(orient_bin_histogram[i_idx][j_idx][k],2) + pow(orient_bin_histogram[i_idx][j_idx+1][k],2) + pow(orient_bin_histogram[i_idx+1][j_idx][k],2) + pow(orient_bin_histogram[i_idx+1][j_idx+1][k],2);

    }
    norm = sqrt(norm);


    for (int k = 0; k < orientations; k++)
    {
        norm_orient_bin_histogram[i_idx][j_idx][k] = orient_bin_histogram[i_idx][j_idx][k] / norm;
        norm_orient_bin_histogram[i_idx][j_idx][9 + k] = orient_bin_histogram[i_idx][j_idx + 1][k] / norm;
        norm_orient_bin_histogram[i_idx][j_idx][18 + k] = orient_bin_histogram[i_idx + 1][j_idx][k] / norm;
        norm_orient_bin_histogram[i_idx][j_idx][27 + k] = orient_bin_histogram[i_idx + 1][j_idx + 1][k] / norm;
    }

}


vector<float> HOG::ComputeHistogramFeatures(Mat& img, Mat& filter, int cell_rows, int cell_cols, int orientations)
{


    Mat f_grad_x = Mat(img.rows,img.cols,CV_32SC1,Scalar(0));
    Mat f_grad_y = Mat(img.rows,img.cols,CV_32SC1,Scalar(0));

    computeGradients(img, f_grad_x, f_grad_y, filter);

    Mat magnitude = Mat(img.rows,img.cols,CV_32FC1,Scalar(0));
    Mat angle = Mat(img.rows,img.cols,CV_32SC1,Scalar(0));
    computeMagnitudeAngle(f_grad_x, f_grad_y, magnitude,  angle);

    vector<vector<vector<float> >> orient_bin_histogram = getHistograms_Cell(magnitude, angle, cell_rows, cell_cols, orientations);
    drawHOGImage(orient_bin_histogram, orientations, cell_rows,  cell_cols);

    int row_blocks = magnitude.rows / cell_rows;
    int col_blocks = magnitude.cols / cell_cols;
    int norm_row_blocks = row_blocks - 1;
    int norm_col_blocks = col_blocks - 1;

    vector<vector<vector<float> >> norm_orient_bin_histogram(norm_row_blocks,vector<vector<float> >(norm_col_blocks,vector<float>(orientations*4,0)));  // cuz 4 8*8 blocks, each having 9 histograms

    for (int i = 0; i < norm_row_blocks; i++)
    {
        for(int j = 0 ; j < norm_col_blocks; j++)
        {
           normalize_histograms(i, j, orient_bin_histogram, norm_orient_bin_histogram);
        }
    }

    // get one vector representation as features for the image
    vector<float>hist_features;
    for (int i = 0; i < norm_orient_bin_histogram.size(); i++)
    {
        for(int j = 0; j < norm_orient_bin_histogram[0].size(); j++)
        {
            for (int k = 0; k < norm_orient_bin_histogram[0][0].size(); k++)
            {
                hist_features.push_back(norm_orient_bin_histogram[i][j][k]);
            }
        }
    }

    return hist_features;
}

void HOG::computeGradients(Mat& img, Mat& f_grad_x, Mat&  f_grad_y, Mat& filter)
{

    int rows = img.rows;
    int cols = img.cols;
    int c = img.channels();

    Mat grad_x = Mat(rows,cols,CV_32SC1,Scalar(0));
    Mat grad_y = Mat(rows,cols,CV_32SC1,Scalar(0));
    Mat temp_c1 = Mat(rows,cols,CV_32SC1,Scalar(0));
    Mat temp_c2 = Mat(rows,cols,CV_32SC1,Scalar(0));
    Mat temp_c3 = Mat(rows,cols,CV_32SC1,Scalar(0));

    vector<Mat>three_channels;
    split(img,three_channels);

    Mat grad_y_t;
    Mat temp_t;
    Mat grad_y_result;
    Mat grad_x_c1;
    Mat grad_x_c2;
    Mat grad_x_c3;

    Mat grad_y_c1;
    Mat grad_y_c2;
    Mat grad_y_c3;

    for (int j = 0; j < rows; j++)
        {
            int* p1 = temp_c1.ptr<int>(j);
            int* p2 = temp_c2.ptr<int>(j);
            int* p3 = temp_c3.ptr<int>(j);
            for(int k = 0; k < cols; k++)
            {

                p1[k] = three_channels[0].at<int>(j,k);
                p2[k] = three_channels[1].at<int>(j,k);
                p3[k] = three_channels[2].at<int>(j,k);

            }
        }

            //channel 1
        getGradients(temp_c1,grad_x,filter);
        transpose(grad_y,grad_y_t);
        transpose(temp_c1,temp_t);
        getGradients(temp_t,grad_y_t,filter);
        transpose(grad_y_t,grad_y_result);
        grad_x_c1 = grad_x.clone();
        grad_y_c1 = grad_y_result.clone();

        getGradients(temp_c2,grad_x,filter);
        transpose(grad_y,grad_y_t);
        transpose(temp_c2,temp_t);
        getGradients(temp_t,grad_y_t,filter);
        transpose(grad_y_t,grad_y_result);
        grad_x_c2 = grad_x.clone();
        grad_y_c2 = grad_y_result.clone();

        getGradients(temp_c3,grad_x,filter);
        transpose(grad_y,grad_y_t);
        transpose(temp_c3,temp_t);
        getGradients(temp_t, grad_y_t,filter);
        transpose(grad_y_t,grad_y_result);
        grad_x_c3 = grad_x.clone();
        grad_y_c3 = grad_y_result.clone();

        // across x dimension
        max(grad_x_c1,grad_x_c2,f_grad_x);
        max(grad_x_c3,f_grad_x,f_grad_x);

        // across y dimension
        max(grad_y_c1,grad_y_c2,f_grad_y);
        max(grad_y_c3,f_grad_y,f_grad_y);
}

void HOG::getGradients(Mat& img, Mat& grad, Mat& filter){
    int i,j;
    int nRows = img.rows;
    int nCols = img.cols;
    for (i = 0; i < nRows; i++){
        int* p_img = img.ptr<int>(i);
        int* p_grad = grad.ptr<int>(i);
        int* p_filter = filter.ptr<int>(0);
        for (j = 0; j < nCols; j++){
            if (j==0)
            {
                p_grad[j]=p_img[j+1];
            }
            else if(j==nCols-1){
                p_grad[j] = -1 * p_img[j-1];
            }
            else{
                p_grad[j] = (p_img[j-1] * p_filter[0] + p_img[j] * p_filter[1] + p_img[j+1] * p_filter[2]);
            }
        }
    }
}


void HOG::computeMagnitudeAngle(Mat& grad_x, Mat& grad_y, Mat& mag, Mat& ang){

    int nRows = grad_x.rows;
    int nCols = grad_x.cols;

    if (grad_x.isContinuous() && grad_y.isContinuous())
    {

        nCols = nRows * nCols;
        nRows = 1;
    }

    int i,j;

    for (i=0; i < nRows; i++){
        int* gx = grad_x.ptr<int>(i);
        int* gy = grad_y.ptr<int>(i);
        float* mg = mag.ptr<float>(i);
        int* theta = ang.ptr<int>(i);

        for (j = 0 ; j<nCols; j++)
        {
            //cout<<j<<endl;
            mg[j] = sqrt(pow(gx[j],2)+pow(gy[j],2));
            theta[j] = int (atan2(gy[j],gx[j]) * 180/M_PI);

            if (theta[j]<0)
            {
                theta[j] = theta[j] + 180;  // if negative then add 90...(tan inverse domain is from -90<theta<90)
            }
        }
    }
}

void HOG::drawHOGImage( vector<vector<vector<float> >> orient_bin_histogram, int orientation, int cell_rows,  int cell_cols)
{
    int row_blocks = orient_bin_histogram.size();
    int col_blocks = orient_bin_histogram[0].size();
    float dx,dy;
    float radius = cell_rows/2 - 1;
    int cx  = cell_cols /2;
    int cy = cell_rows / 2;
    int center_x;
    int center_y;
    float val;
    Mat hog_display = Mat(row_blocks * cell_rows, col_blocks * cell_cols,CV_32FC1,Scalar(0));
    float norm =0;
    for (int i = 0; i < row_blocks; i++)
    {
        for(int j = 0; j < col_blocks; j++)
        {
            int l = 0;
            norm = 0;
            while (l<9)
            {
                norm = norm + orient_bin_histogram[i][j][l];
                l++;
            }

            for (int k = 0; k < orientation; k++)
            {
                dx = radius * cos(k/float(orientation) * M_PI); // basically we are getting x and y coordinates
                dy = radius * sin(k/float(orientation) * M_PI);
                center_y = i * cell_rows + cy;
                center_x = j * cell_cols + cx;
                val = orient_bin_histogram[i][j][k] * 64;
                //val = (val/norm) * 255;
                line(hog_display,Point(center_x - dx, center_y + dy),Point(center_x + dx, center_y - dy), Scalar(val),1);

            }
        }
    }
    Mat hog_display_uchar;
    hog_display.convertTo(hog_display_uchar,CV_8UC3);
    imshow("Hog display", hog_display_uchar);
    waitKey(0);
}




void HOG::check_hog(string img_file)
{
    Mat img_load = imread(img_file,CV_LOAD_IMAGE_COLOR);
    Mat img;
    int rows = img_load.rows;
    int cols = img_load.cols;
    int cell_rows = 8;
    int cell_cols = 8;
    int orientations = 9;
    int r_rows,r_cols;

    if (rows % cell_rows >0 || cols % cell_cols>0)
    {
        if (rows % cell_rows >0)
        {
            //r_rows = rows - (rows % cell_rows);
            //r_rows = rows % cell_rows;
            r_rows = rows/cell_rows + 1;
            rows = r_rows * cell_rows;
        }
        if (cols % cell_cols > 0)
        {
            //r_cols = cols - (cols % cell_cols);
            r_cols = cols/cell_cols + 1;
            cols = r_cols * cell_cols;
        }

        Mat dst;

        resize(img_load,dst,Size(cols,rows),CV_INTER_LINEAR);

        img_load = dst;
    }

    img_load.convertTo(img,CV_32SC3);
    Mat filter = Mat(1,3,CV_32SC1,Scalar(0));
    filter.at<int>(0,0) = -1;
    filter.at<int>(0,1) = 0;
    filter.at<int>(0,2) = 1;


    ComputeHistogramFeatures(img, filter, cell_rows, cell_cols, orientations);

}


int main(){

    string img_file = "/home/suhaspillai/Suhas/Pedistrian_Detection/INRIAPerson/test_64x128_H96/pos/crop001633d.png";
    img_file = "/home/suhaspillai/Suhas/Practice/HOG_implemntation/Images/car.jpg";
    HOG h_obj;
    h_obj.check_hog(img_file);
}
