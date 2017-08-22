#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv.hpp>
#include <QFileDialog>
#include <qfiledialog.h>
#include <QMessageBox>
#include <QDebug>
#include <QImage>
#include <QVector>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private slots:
    void on_pushButton_open_clicked();

    QImage Mat2QImage(const cv::Mat mat_original);

    void showImage(const cv::Mat &mat_original);

    void on_pushButton_gray_clicked();

    void on_horizontalSlider_valueChanged(int value);

    void on_pushButton_Binary_clicked();

    void on_pushButton_Clear_clicked();

    void on_pushButton_Invert_clicked();

    void on_pushButton_Blur_clicked();

    void on_pushButton_Brightness_clicked();

    void on_pushButton_Color_clicked();

    void on_pushButton_Save_clicked();

    void cal_percentage();

    void on_pushButton_Grabcut_clicked();

    void on_pushButton_Cluster_clicked();

    void on_pushButton_Dithering_BW_clicked();

    void on_pushButton_Dithering_color_clicked();

    void on_pushButton_Sharpening_clicked();

private:
    Ui::MainWindow *ui;
    cv::Mat Ori;
    cv::Mat Processed;
    cv::Mat Gray;
    cv::Mat Final_img;
    bool GrayYet;
    int color;
    int status;
    QImage *Image;
    QImage *Image_save;
    QPixmap pixmap;
    QImage *a;
};

#endif // MAINWINDOW_H
