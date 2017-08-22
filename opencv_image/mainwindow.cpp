#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace cv;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    this->setWindowTitle("Image Processing using opencv and Qt");
    color=0;
    status=0;
    GrayYet = false;
}

MainWindow::~MainWindow()
{
    delete ui;
}

//-----------------basic settings---------------------

void MainWindow::on_pushButton_open_clicked()
{
    QString fileName = QFileDialog::getOpenFileName();
    if(fileName==NULL){
        QMessageBox::information(0,"Error!","Need image file!");
        return;
    }
    else{
//        Image->load(fileName);
//        pixmap = QPixmap::fromImage(*Image);
        a = new QImage;
//        QString FilePath = QFileDialog::getOpenFileName(this,tr("Open File"),"/",tr("Images (*.png *.jpg)"));
        a->load(fileName);
        if(a->isNull()) {QMessageBox::information(this,"Warning!","Need a image file!");}
            if(!a->isNull()){
                Image = new QImage;
                Image=a;
                pixmap = QPixmap::fromImage(*Image);
            }
        Mat mat_original = imread(fileName.toStdString());
        int width = mat_original.cols;
        int height = mat_original.rows;
        //
        cv::resize(mat_original,mat_original,Size(width/2,height/2));
        this->Ori = mat_original.clone();
        //imshow("img", mat_original);
        showImage(mat_original);
        GrayYet = false;
    }
    ui->label_img_processed->clear();
}

QImage MainWindow::Mat2QImage(const Mat mat_original)
{
    if(mat_original.type()==CV_8UC1)
    {
        QVector<QRgb> colorTable;
        for (int i = 0; i < 256; i++)
            colorTable.push_back(qRgb(i,i,i));
        const uchar *qImageBuffer = (const uchar*)mat_original.data;
        QImage img(qImageBuffer, mat_original.cols, mat_original.rows, mat_original.step, QImage::Format_Indexed8);
        img.setColorTable(colorTable);
        return img;
    }
    if(mat_original.type()== CV_8UC3)
    {
        const uchar *qImageBuffer = (const uchar*)mat_original.data;
        QImage img(qImageBuffer, mat_original.cols, mat_original.rows, mat_original.step, QImage::Format_RGB888);
        return img.rgbSwapped();
    }
    else
    {
        qDebug() << "Error! Can't convert image type.";
        return QImage();
    }
}

void MainWindow::showImage(const Mat &mat_original)
{
    //A function for opening image on label 1
    Mat mat_processed;
    int width = ui->label_img_original->width();
    int height = ui->label_img_original->height();
    double ratio = (double)width / (double)height;
    double imgRatio = mat_original.cols / mat_original.rows;

    if (ratio<imgRatio)
    {
       cv::resize(mat_original,mat_processed,Size(width,(mat_original.rows*width)/mat_original.cols));
    }
    else
    {
       cv::resize(mat_original,mat_processed,Size((mat_original.cols*height)/mat_original.rows,height));
    }

    ui->label_img_original->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_processed))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    this->Processed = mat_processed.clone();
    this->Final_img = mat_processed.clone();
}

void MainWindow::on_pushButton_Save_clicked()
{
    //Save processed image
    Mat mat_final = this->Final_img;
    if(status == 0){
        QMessageBox::information(0,"Error!","Need processed image!");
        return;
    }
    //opencv mat
    else if(status <= 8){
        QString fileName = QFileDialog::getSaveFileName(this, tr("Save Image"), "/", tr("image (*.bmp)"));
        if(fileName==NULL){
            QMessageBox::information(0,"Error!","Need image name!");
            return;
        }
        else{
            imwrite(fileName.toStdString(),mat_final);
        }
    }
    //QImage pixmap
    else{
        QString ImagePath = QFileDialog::getSaveFileName(this, tr("Save Image"), "/", tr("JPG (*.jpg);;PNG (*png)"));
        *Image_save = pixmap.toImage();
        Image_save->save(ImagePath);
    }

}

void MainWindow::on_pushButton_Clear_clicked()
{
    //Clear all image
    ui->label_img_original->clear();
    ui->label_img_processed->clear();
}

//--------------image processing-----------------

void MainWindow::on_pushButton_gray_clicked()
{
    //grayscale #1
    Mat mat_processed = this->Processed;
    Mat mat_gray;
    mat_gray.create(Size(mat_processed.cols,mat_processed.rows),CV_8UC1);
    //easy way
//    cvtColor(mat_processed, mat_gray, CV_BGR2GRAY);
    //do it by myself
    for(int r=0; r<mat_processed.rows; r++){
        for(int c=0; c<mat_processed.cols; c++){
            mat_gray.at<uchar>(r,c)
                    = (mat_processed.at<Vec3b>(r,c)[0]
                    + mat_processed.at<Vec3b>(r,c)[1]
                    + mat_processed.at<Vec3b>(r,c)[2])/3;
//            mat_gray.at<uchar>(r,c)
//                    = (0.299 * mat_processed.at<Vec3b>(r,c)[0]
//                    + 0.587 * mat_processed.at<Vec3b>(r,c)[1]
//                    + 0.114 * mat_processed.at<Vec3b>(r,c)[2]);
        }
    }
    this->Gray = mat_gray.clone();
    this->Final_img = mat_gray.clone();
    GrayYet = true;
    ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_gray))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: Grayscale");
    //change status
    status = 1;
    ui->horizontalSlider->setValue(0);
}

void MainWindow::on_pushButton_Binary_clicked()
{
    //Binary #2
    //change status
    status = 2;
    ui->horizontalSlider->setSliderPosition(125);
}

void MainWindow::on_pushButton_Invert_clicked()
{
    //Invert #3
    Mat mat_processed = this->Processed;
    Mat mat_invert;
    mat_invert = mat_processed.clone();
    //easy way
//    mat_invert =  Scalar::all(255) - mat_processed;
//    or-> bitwise_not ( mat_processed, mat_invert );
//    can't work-> invert(mat_processed,mat_invert);
    //do it by myself
    for(int r=0; r<mat_invert.rows; r++){
        for(int c=0; c<mat_invert.cols; c++){
            for(int n=0; n<mat_invert.channels(); n++){
                mat_invert.at<Vec3b>(r,c)[n] = 255 - mat_processed.at<Vec3b>(r,c)[n];
            }
        }
    }
    ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_invert))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: Invert");
    this->Final_img = mat_invert.clone();
    //change status
    status = 3;
    ui->horizontalSlider->setValue(0);
}

void MainWindow::on_pushButton_Brightness_clicked()
{
    //Brightness #4
    //change status
    status = 4;
    ui->horizontalSlider->setSliderPosition(150);
}

void MainWindow::on_pushButton_Color_clicked()
{
    //Color #5
    Mat mat_processed = this->Processed;
    Mat mat_color;
    mat_color = mat_processed.clone();
    QVector<int> value(3);
    if(color==0){
        value[0] = 50; value[1] = 0; value[2] = 0;
    }
    else if(color==1){
        value[0] = 0;
        value[1] = 50;
        value[2] = 0;
    }
    else if(color==2){
        value[0] = 0;
        value[1] = 0;
        value[2] = 50;
    }
    for(int r=0; r<mat_color.rows; r++){
        for(int c=0; c<mat_color.cols; c++){
            for(int k=0; k<mat_color.channels(); k++){
                if(mat_color.at<Vec3b>(r,c)[k]+value[k] >255){
                    mat_color.at<Vec3b>(r,c)[k]=255;
                }
                else if(mat_color.at<Vec3b>(r,c)[k]+value[k] <0){
                    mat_color.at<Vec3b>(r,c)[k]=0;
                }
                else{
                    mat_color.at<Vec3b>(r,c)[k] =
                            mat_color.at<Vec3b>(r,c)[k]+value[k];
                }
            }
        }
    }
    color+=1;
    if(color==3) color=0;
    ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_color))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: Color changed");
    this->Final_img = mat_color.clone();
    //change status
    status = 5;
    ui->horizontalSlider->setValue(0);
}


void MainWindow::on_pushButton_Blur_clicked()
{
    //Blur #6
    //change status
    status = 6;
    ui->horizontalSlider->setSliderPosition(6);
}

void MainWindow::on_pushButton_Cluster_clicked()
{
    //Cluster #7
    // #7
    Mat mat_processed = this->Processed;
    Mat mat_grabcut;
    mat_grabcut = mat_processed.clone();
    pyrMeanShiftFiltering( mat_processed, mat_grabcut, 20, 45, 3);
    ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_grabcut))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: MeanShiftFiltering");
    this->Final_img = mat_grabcut.clone();
    //change status
    status = 7;
    ui->horizontalSlider->setValue(0);
}

void MainWindow::on_pushButton_Grabcut_clicked()
{
    //Grabcut #8
    Mat mat_processed = this->Processed;
    Mat mat_grabcut;
    mat_grabcut = mat_processed.clone();
    //pyrMeanShiftFiltering( mat_processed, mat_grabcut, 20, 45, 3);
    int border = 1;
    Rect rect(border,border,mat_processed.cols-2*border,mat_processed.rows-2*border);
    Mat bgdModel,fgdModel;
    grabCut(mat_processed, mat_grabcut, rect, bgdModel, fgdModel, 1, GC_INIT_WITH_RECT);
    // Get the pixels marked as likely foreground
    compare(mat_grabcut,GC_PR_FGD,mat_grabcut,CMP_EQ);
    // Generate output image
    Mat foreground(mat_processed.size(),CV_8UC3,Scalar(255,255,255));
    mat_processed.copyTo(foreground,mat_grabcut); // bg pixels not copied
    ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(foreground))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: Grabcut");
    this->Final_img = foreground.clone();
    //change status
    status = 8;
    ui->horizontalSlider->setValue(0);
}



//新加功能~~~

void MainWindow::on_pushButton_Dithering_BW_clicked()
{
    QImage grayscale = *Image;
    QRgb val;
    QColor oldColor;
    for(int x=0;x<grayscale.width();x++){
        for(int y=0;y<grayscale.height();y++){
            oldColor = QColor(grayscale.pixel(x,y));
            int ave = (oldColor.red()+oldColor.green()+oldColor.blue())/3;
            val = qRgb(ave,ave,ave);
            grayscale.setPixel(x,y,val);
        }
    }
    QImage dithering = grayscale;
    QRgb val1;
    QColor oldColor1;
    int shade, err= 0;
    for (int y=0;y<dithering.height();y++){
        for(int x=0;x<dithering.width()-1;x++)
        {
            oldColor1 = QColor(dithering.pixel(x,y));
            shade = oldColor1.red() + err;
            if(shade<127){
                err = shade;
                shade = 0;
            }
            else{
                err = shade - 255;
                shade = 255;
            }
            val1 = qRgb(shade,shade,shade);
            dithering.setPixel(x,y,val1);
        }
    }
    pixmap = QPixmap::fromImage(dithering);
    ui->label_img_processed->setPixmap(pixmap.scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: Dithering BW");
    //change status
    status = 9;
    ui->horizontalSlider->setValue(0);
}


void MainWindow::on_pushButton_Dithering_color_clicked()
{
    QImage dithering = *Image;
    QRgb val1;
    QColor oldColor1;
    int shader, shadeg, shadeb, errr = 0, errg = 0, errb = 0;
    for (int y=0;y<dithering.height();y++){
        for(int x=0;x<dithering.width()-1;x++)
        {
            oldColor1 = QColor(dithering.pixel(x,y));
            shader = oldColor1.red() + errr;
            shadeg = oldColor1.green() + errg;
            shadeb = oldColor1.blue() + errb;
            if(shader<127){
                errr = shader;
                shader = 0;
            }
            else{
                errr = shader - 255;
                shader = 255;
            }
            if(shadeg<127){
                errg = shadeg;
                shadeg = 0;
            }
            else{
                errg = shadeg - 255;
                shadeg = 255;
            }
            if(shadeb<127){
                errb = shadeb;
                shadeb = 0;
            }
            else{
                errb = shadeb - 255;
                shadeb = 255;
            }
            val1 = qRgb(shader,shadeg,shadeb);
            dithering.setPixel(x,y,val1);
        }
    }
    pixmap = QPixmap::fromImage(dithering);
    ui->label_img_processed->setPixmap(pixmap.scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
    ui->label_information->setText("Status: Dithering color");
    //change status
    status = 10;
    ui->horizontalSlider->setValue(0);
}


void MainWindow::on_pushButton_Sharpening_clicked()
{
    //change status
    status = 11;
    ui->horizontalSlider->setValue(90);
}

//------------------slider------------------

void MainWindow::on_horizontalSlider_valueChanged(int value)
{
    //action depends on status
        //only works when Binary #2 / Brightness #4 / Blur #6 / Sharpen #11 checked
        if(status == 2) {  //binary
            if(!GrayYet){
                Mat mat_processed = this->Processed;
                Mat mat_gray;
                mat_gray.create(Size(mat_processed.cols,mat_processed.rows),CV_8UC1);
                cvtColor(mat_processed, mat_gray, CV_BGR2GRAY);
                this->Gray = mat_gray.clone();
                GrayYet = true;
            }
            Mat mat_gray = this->Gray;
            Mat mat_cut;
            mat_cut.create(Size(mat_gray.cols,mat_gray.rows),CV_8UC1);
            threshold(mat_gray, mat_cut, value, 255, THRESH_BINARY);
            ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_cut))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
            this->Final_img = mat_cut.clone();
            //calculate black percentage
            cal_percentage();
        }
        else if(status == 4) {  //Brightness
            Mat mat_processed = this->Processed;
            Mat mat_bright;
            mat_bright = mat_processed.clone();
//            mat_bright.create(Size(mat_processed.cols,mat_processed.rows),CV_8UC1);
            for(int r=0; r<mat_bright.rows; r++){
                for(int c=0; c<mat_bright.cols; c++){
                    for(int k=0; k<mat_bright.channels(); k++){
                        if(mat_bright.at<Vec3b>(r,c)[k]+(value-125)*2 >255){
                            mat_bright.at<Vec3b>(r,c)[k]=255;
                        }
                        else if(mat_bright.at<Vec3b>(r,c)[k]+(value-125)*2 <0){
                            mat_bright.at<Vec3b>(r,c)[k]=0;
                        }
                        else{
                            mat_bright.at<Vec3b>(r,c)[k] =
                                    mat_bright.at<Vec3b>(r,c)[k]+(value-125)*2;
                        }
                    }
                }
            }
            ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_bright))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
            if((value-125)*2 > 0){ui->label_information->setText("Status: Brightness +" + QString::number((value-125)*2));}
            else{ui->label_information->setText("Status: Brightness " + QString::number((value-125)*2));}
            this->Final_img = mat_bright.clone();
        }
        else if(status == 11){  //Sharpen
            QImage sharpen = *Image;
            QRgb val;
            QColor oldColor;
            int Lap[9] = {-1, -1, -1, -1, (value/10), -1, -1, -1, -1};
            for(int x=1;x<sharpen.width()-1;x++){
                for(int y=1;y<sharpen.height()-1;y++){
                    int r=0, g=0, b=0;
                    int Index=0;
                    for (int col = -1; col <= 1; col++){
                        for (int row = -1; row <= 1; row++){
                            oldColor = QColor(sharpen.pixel(x+row,y+col));
                            r += oldColor.red() * Lap[Index];
                            g += oldColor.green() * Lap[Index];
                            b += oldColor.blue() * Lap[Index];
                            Index++;
                        }
                    }
                    if(r<0) r=0; if(r>255) r=255;
                    if(g<0) g=0; if(g>255) g=255;
                    if(b<0) b=0; if(b>255) b=255;
                    val = qRgb(r,g,b);
                    sharpen.setPixel(x-1,y-1,val);
                }
            }
            pixmap = QPixmap::fromImage(sharpen);
            ui->label_img_processed->setPixmap(pixmap.scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
            ui->label_information->setText("Status: Sharpening " + QString::number(value/10));
    }
        else if(status == 6) {  //Blur
        if(value>1){
            Mat mat_processed = this->Processed;
            Mat mat_blur;
            mat_blur = mat_processed.clone();
            //easy way
            blur(mat_processed,mat_blur,Size(value,value));
            //do it by myself
//            for(int r=1; r<mat_processed.rows-1; r++){
//                for(int c=1; c<mat_processed.cols-1; c++){
//                    for(int n=0; n<mat_processed.channels(); n++){
//                        mat_blur.at<Vec3b>(r,c)[n] =
//                                (mat_processed.at<Vec3b>(r-1,c-1)[n] + mat_processed.at<Vec3b>(r-1,c)[n] + mat_processed.at<Vec3b>(r-1,c+1)[n]
//                                 + mat_processed.at<Vec3b>(r,c-1)[n] + mat_processed.at<Vec3b>(r,c)[n] + mat_processed.at<Vec3b>(r,c+1)[n]
//                                 + mat_processed.at<Vec3b>(r+1,c-1)[n] + mat_processed.at<Vec3b>(r+1,c)[n] + mat_processed.at<Vec3b>(r+1,c+1)[n])/9;
//                    }
//                }
//            }
            ui->label_img_processed->setPixmap((QPixmap::fromImage(this->Mat2QImage(mat_blur))).scaled(ui->label_img_processed->width(),ui->label_img_processed->height(),Qt::KeepAspectRatio));
            ui->label_information->setText("Status: Blur 3");
            this->Final_img = mat_blur.clone();
        }
    }
}

//-------------------other functions----------------------

void MainWindow::cal_percentage()
{
    //calculate percentage of black pixel
    Mat mat_cut = this->Final_img;
    double count = 0 ;
    for(int r=0; r<mat_cut.rows; r++){
        for(int c=0; c<mat_cut.cols; c++){
            if(mat_cut.at<uchar>(r,c) == 0)
                count ++;
        }
    }
    double total = mat_cut.rows * mat_cut.cols;
    double ratio = count * 100 /total;
    ui->label_information->setText("Status: Binary, black percentage " + QString::number(ratio) + " %");
}



