/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.9.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../opencv_image/mainwindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.9.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[18];
    char stringdata0[363];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 26), // "on_pushButton_open_clicked"
QT_MOC_LITERAL(2, 38, 0), // ""
QT_MOC_LITERAL(3, 39, 10), // "Mat2QImage"
QT_MOC_LITERAL(4, 50, 7), // "cv::Mat"
QT_MOC_LITERAL(5, 58, 12), // "mat_original"
QT_MOC_LITERAL(6, 71, 9), // "showImage"
QT_MOC_LITERAL(7, 81, 26), // "on_pushButton_gray_clicked"
QT_MOC_LITERAL(8, 108, 32), // "on_horizontalSlider_valueChanged"
QT_MOC_LITERAL(9, 141, 5), // "value"
QT_MOC_LITERAL(10, 147, 28), // "on_pushButton_Binary_clicked"
QT_MOC_LITERAL(11, 176, 27), // "on_pushButton_Clear_clicked"
QT_MOC_LITERAL(12, 204, 28), // "on_pushButton_Invert_clicked"
QT_MOC_LITERAL(13, 233, 26), // "on_pushButton_Blur_clicked"
QT_MOC_LITERAL(14, 260, 32), // "on_pushButton_Brightness_clicked"
QT_MOC_LITERAL(15, 293, 27), // "on_pushButton_Color_clicked"
QT_MOC_LITERAL(16, 321, 26), // "on_pushButton_Save_clicked"
QT_MOC_LITERAL(17, 348, 14) // "cal_percentage"

    },
    "MainWindow\0on_pushButton_open_clicked\0"
    "\0Mat2QImage\0cv::Mat\0mat_original\0"
    "showImage\0on_pushButton_gray_clicked\0"
    "on_horizontalSlider_valueChanged\0value\0"
    "on_pushButton_Binary_clicked\0"
    "on_pushButton_Clear_clicked\0"
    "on_pushButton_Invert_clicked\0"
    "on_pushButton_Blur_clicked\0"
    "on_pushButton_Brightness_clicked\0"
    "on_pushButton_Color_clicked\0"
    "on_pushButton_Save_clicked\0cal_percentage"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       7,       // revision
       0,       // classname
       0,    0, // classinfo
      13,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,   79,    2, 0x08 /* Private */,
       3,    1,   80,    2, 0x08 /* Private */,
       6,    1,   83,    2, 0x08 /* Private */,
       7,    0,   86,    2, 0x08 /* Private */,
       8,    1,   87,    2, 0x08 /* Private */,
      10,    0,   90,    2, 0x08 /* Private */,
      11,    0,   91,    2, 0x08 /* Private */,
      12,    0,   92,    2, 0x08 /* Private */,
      13,    0,   93,    2, 0x08 /* Private */,
      14,    0,   94,    2, 0x08 /* Private */,
      15,    0,   95,    2, 0x08 /* Private */,
      16,    0,   96,    2, 0x08 /* Private */,
      17,    0,   97,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::QImage, 0x80000000 | 4,    5,
    QMetaType::Void, 0x80000000 | 4,    5,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    9,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        MainWindow *_t = static_cast<MainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->on_pushButton_open_clicked(); break;
        case 1: { QImage _r = _t->Mat2QImage((*reinterpret_cast< const cv::Mat(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QImage*>(_a[0]) = std::move(_r); }  break;
        case 2: _t->showImage((*reinterpret_cast< const cv::Mat(*)>(_a[1]))); break;
        case 3: _t->on_pushButton_gray_clicked(); break;
        case 4: _t->on_horizontalSlider_valueChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->on_pushButton_Binary_clicked(); break;
        case 6: _t->on_pushButton_Clear_clicked(); break;
        case 7: _t->on_pushButton_Invert_clicked(); break;
        case 8: _t->on_pushButton_Blur_clicked(); break;
        case 9: _t->on_pushButton_Brightness_clicked(); break;
        case 10: _t->on_pushButton_Color_clicked(); break;
        case 11: _t->on_pushButton_Save_clicked(); break;
        case 12: _t->cal_percentage(); break;
        default: ;
        }
    }
}

const QMetaObject MainWindow::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_MainWindow.data,
      qt_meta_data_MainWindow,  qt_static_metacall, nullptr, nullptr}
};


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(const_cast< MainWindow*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 13)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 13;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 13)
            *reinterpret_cast<int*>(_a[0]) = -1;
        _id -= 13;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
