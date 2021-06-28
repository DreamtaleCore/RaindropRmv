#include <opencv2/opencv.hpp>
#include <iostream>
#include <glob.h>
#include <regex>
#include <boost/format.hpp>

#define HEIGHT 256
#define WIDTH 512

using namespace std;
using boost::format;
using boost::str;


void getFiles(const string &pattern, vector<string> &filePath);
void getEdge(cv::Mat &edge, cv::Mat &img);
cv::Vec3f absdiff(cv::Vec3b &a, cv::Vec3b &b);

int main() {
    vector<string> imgPath;
    getFiles("repo/dataset/rain_val_with_sem/*_B.png", imgPath);

    int totalIndex = imgPath.size();
    cout << totalIndex << endl;

    cv::Mat edge(HEIGHT, WIDTH, CV_8UC1);
    string savePath;
    for(int i{0}; i < totalIndex; ++i) {
        cv::Mat img = cv::imread(imgPath[i]);
        getEdge(edge, img);
        savePath = regex_replace(imgPath[i], regex(R"(_B)"), "_E");
        //savePath = regex_replace(imgPath[i], regex(R"(rain.png$)"), "E.png");
        cv::imwrite(savePath, edge);
        if(i % 100 == 0) {
            cout << int(float(i) / totalIndex * 100) << "%" << endl;
        }
    }
    return 0;
}

void getFiles(const string &pattern, vector<string> &filePath) {
    glob_t globBuf;

    glob(pattern.c_str(), GLOB_TILDE, NULL, &globBuf);
    
    for(unsigned i{0}; i < globBuf.gl_pathc; i++) {
        filePath.push_back(globBuf.gl_pathv[i]);
    }

    if(globBuf.gl_pathc > 0) {
        globfree(&globBuf);
    }
}

void getEdge(cv::Mat &edge, cv::Mat &img) {
    for(int row = 0; row < img.rows; ++row) {
        for(int col = 0; col < img.cols; ++col) {
            cv::Vec3f pixel{0, 0 , 0};
            if(row > 0)
                pixel += absdiff(img.at<cv::Vec3b>(row-1, col), img.at<cv::Vec3b>(row, col));
            if(row < img.rows - 1)
                pixel += absdiff(img.at<cv::Vec3b>(row+1, col), img.at<cv::Vec3b>(row, col));
            if(col > 0)
                pixel += absdiff(img.at<cv::Vec3b>(row, col-1), img.at<cv::Vec3b>(row, col));
            if(col < img.cols - 1)
                pixel += absdiff(img.at<cv::Vec3b>(row, col+1), img.at<cv::Vec3b>(row, col));
            edge.at<char>(row, col) = cv::sum(pixel)[0] / 4.0;
        }
    }
}

cv::Vec3f absdiff(cv::Vec3b &a, cv::Vec3b &b) {
    cv::Vec3f c{a - b};
    for(int i = 0; i < 3; i++) {
        c[i] = abs(c[i]);
    }
    return c;
}
