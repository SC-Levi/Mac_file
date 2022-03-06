#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int ImgPorcess(char *imgPath) {
    cout << "ss" << endl;
    Mat src, dst;
    char *window_name = "copyMakeBorder Demo";

    src = imread(imgPath);
    if (!src.data) {
        cerr << "Can't open the image!" << endl;
        return EXIT_FAILURE;
    }

    namedWindow(window_name);
    cout << src.rows << endl;
    cout << src.cols << endl;
    dst = src;


    while (true) {
        waitKey(500);
        copyMakeBorder(src, dst, 0, 10, 0, 20, BORDER_CONSTANT, 0);
        imshow(window_name, dst);
    }

}