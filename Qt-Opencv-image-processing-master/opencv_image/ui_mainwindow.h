/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QLabel *label_img_original;
    QPushButton *pushButton_open;
    QLabel *label_original;
    QLabel *label_img_processed;
    QLabel *label_processed;
    QPushButton *pushButton_gray;
    QPushButton *pushButton_Binary;
    QSlider *horizontalSlider;
    QLabel *label_slider;
    QPushButton *pushButton_Clear;
    QPushButton *pushButton_Brightness;
    QPushButton *pushButton_Color;
    QPushButton *pushButton_Blur;
    QPushButton *pushButton_Save;
    QPushButton *pushButton_Invert;
    QLabel *label_information;
    QMenuBar *menuBar;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(900, 500);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        label_img_original = new QLabel(centralWidget);
        label_img_original->setObjectName(QStringLiteral("label_img_original"));
        label_img_original->setGeometry(QRect(10, 140, 400, 300));
        pushButton_open = new QPushButton(centralWidget);
        pushButton_open->setObjectName(QStringLiteral("pushButton_open"));
        pushButton_open->setGeometry(QRect(10, 10, 75, 23));
        label_original = new QLabel(centralWidget);
        label_original->setObjectName(QStringLiteral("label_original"));
        label_original->setGeometry(QRect(10, 100, 91, 31));
        label_img_processed = new QLabel(centralWidget);
        label_img_processed->setObjectName(QStringLiteral("label_img_processed"));
        label_img_processed->setGeometry(QRect(430, 140, 400, 300));
        label_processed = new QLabel(centralWidget);
        label_processed->setObjectName(QStringLiteral("label_processed"));
        label_processed->setGeometry(QRect(430, 90, 101, 41));
        pushButton_gray = new QPushButton(centralWidget);
        pushButton_gray->setObjectName(QStringLiteral("pushButton_gray"));
        pushButton_gray->setGeometry(QRect(100, 10, 75, 23));
        pushButton_gray->setCheckable(false);
        pushButton_Binary = new QPushButton(centralWidget);
        pushButton_Binary->setObjectName(QStringLiteral("pushButton_Binary"));
        pushButton_Binary->setGeometry(QRect(190, 10, 75, 23));
        pushButton_Binary->setCheckable(false);
        horizontalSlider = new QSlider(centralWidget);
        horizontalSlider->setObjectName(QStringLiteral("horizontalSlider"));
        horizontalSlider->setGeometry(QRect(570, 30, 281, 22));
        horizontalSlider->setMaximum(255);
        horizontalSlider->setOrientation(Qt::Horizontal);
        label_slider = new QLabel(centralWidget);
        label_slider->setObjectName(QStringLiteral("label_slider"));
        label_slider->setGeometry(QRect(430, 20, 131, 41));
        label_slider->setWordWrap(true);
        pushButton_Clear = new QPushButton(centralWidget);
        pushButton_Clear->setObjectName(QStringLiteral("pushButton_Clear"));
        pushButton_Clear->setGeometry(QRect(10, 70, 75, 23));
        pushButton_Brightness = new QPushButton(centralWidget);
        pushButton_Brightness->setObjectName(QStringLiteral("pushButton_Brightness"));
        pushButton_Brightness->setGeometry(QRect(100, 40, 75, 23));
        pushButton_Brightness->setCheckable(false);
        pushButton_Color = new QPushButton(centralWidget);
        pushButton_Color->setObjectName(QStringLiteral("pushButton_Color"));
        pushButton_Color->setGeometry(QRect(190, 40, 75, 23));
        pushButton_Color->setCheckable(false);
        pushButton_Blur = new QPushButton(centralWidget);
        pushButton_Blur->setObjectName(QStringLiteral("pushButton_Blur"));
        pushButton_Blur->setGeometry(QRect(280, 40, 75, 23));
        pushButton_Blur->setCheckable(false);
        pushButton_Save = new QPushButton(centralWidget);
        pushButton_Save->setObjectName(QStringLiteral("pushButton_Save"));
        pushButton_Save->setGeometry(QRect(10, 40, 75, 23));
        pushButton_Save->setCheckable(false);
        pushButton_Invert = new QPushButton(centralWidget);
        pushButton_Invert->setObjectName(QStringLiteral("pushButton_Invert"));
        pushButton_Invert->setGeometry(QRect(280, 10, 75, 23));
        label_information = new QLabel(centralWidget);
        label_information->setObjectName(QStringLiteral("label_information"));
        label_information->setGeometry(QRect(430, 70, 421, 16));
        MainWindow->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(MainWindow);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 900, 21));
        MainWindow->setMenuBar(menuBar);
        mainToolBar = new QToolBar(MainWindow);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        MainWindow->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(MainWindow);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        MainWindow->setStatusBar(statusBar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", Q_NULLPTR));
        label_img_original->setText(QString());
        pushButton_open->setText(QApplication::translate("MainWindow", "Open Image", Q_NULLPTR));
        label_original->setText(QApplication::translate("MainWindow", "Original Image", Q_NULLPTR));
        label_img_processed->setText(QString());
        label_processed->setText(QApplication::translate("MainWindow", "Processed Image", Q_NULLPTR));
        pushButton_gray->setText(QApplication::translate("MainWindow", "Grayscale", Q_NULLPTR));
        pushButton_Binary->setText(QApplication::translate("MainWindow", "Binary", Q_NULLPTR));
        label_slider->setText(QApplication::translate("MainWindow", "Slider for Binary / Brightness / Color / Blur", Q_NULLPTR));
        pushButton_Clear->setText(QApplication::translate("MainWindow", "Clear Image", Q_NULLPTR));
        pushButton_Brightness->setText(QApplication::translate("MainWindow", "Brightness", Q_NULLPTR));
        pushButton_Color->setText(QApplication::translate("MainWindow", "Change Color", Q_NULLPTR));
        pushButton_Blur->setText(QApplication::translate("MainWindow", "Blur", Q_NULLPTR));
        pushButton_Save->setText(QApplication::translate("MainWindow", "Save Image", Q_NULLPTR));
        pushButton_Invert->setText(QApplication::translate("MainWindow", "Invert", Q_NULLPTR));
        label_information->setText(QApplication::translate("MainWindow", "Information: ", Q_NULLPTR));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
