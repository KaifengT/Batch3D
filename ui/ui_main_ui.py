# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 6.6.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QHeaderView,
    QMainWindow, QSizePolicy, QSpacerItem, QTableWidgetItem,
    QTextBrowser, QTextEdit, QVBoxLayout, QWidget)

from glw.GLWidget import GLWidget
from qfluentwidgets import (DoubleSpinBox, DropDownToolButton, PushButton, StrongBodyLabel,
    SwitchButton, TableWidget)
from ui.addon import GLAddon_circling

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(795, 463)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_3 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(-1, -1, 0, -1)
        self.tool = QWidget(self.centralwidget)
        self.tool.setObjectName(u"tool")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tool.sizePolicy().hasHeightForWidth())
        self.tool.setSizePolicy(sizePolicy)
        self.tool.setMaximumSize(QSize(370, 16777215))
        self.verticalLayout = QVBoxLayout(self.tool)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(12, 12, 12, 12)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton_openfolder = PushButton(self.tool)
        self.pushButton_openfolder.setObjectName(u"pushButton_openfolder")
        sizePolicy1 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton_openfolder.sizePolicy().hasHeightForWidth())
        self.pushButton_openfolder.setSizePolicy(sizePolicy1)
        self.pushButton_openfolder.setMinimumSize(QSize(100, 0))

        self.horizontalLayout.addWidget(self.pushButton_openfolder)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.pushButton_openremotefolder = PushButton(self.tool)
        self.pushButton_openremotefolder.setObjectName(u"pushButton_openremotefolder")
        sizePolicy1.setHeightForWidth(self.pushButton_openremotefolder.sizePolicy().hasHeightForWidth())
        self.pushButton_openremotefolder.setSizePolicy(sizePolicy1)
        self.pushButton_openremotefolder.setMinimumSize(QSize(100, 0))

        self.verticalLayout.addWidget(self.pushButton_openremotefolder)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(-1, 0, -1, -1)
        self.label = StrongBodyLabel(self.tool)
        self.label.setObjectName(u"label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        self.label.setStyleSheet(u"color: rgb(238, 238, 238);")

        self.horizontalLayout_6.addWidget(self.label)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_6.addItem(self.horizontalSpacer_4)

        self.checkBox = SwitchButton(self.tool)
        self.checkBox.setObjectName(u"checkBox")
        sizePolicy1.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy1)
        self.checkBox.setChecked(True)

        self.horizontalLayout_6.addWidget(self.checkBox)


        self.verticalLayout.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setContentsMargins(-1, 0, -1, -1)
        self.label_4 = StrongBodyLabel(self.tool)
        self.label_4.setObjectName(u"label_4")
        sizePolicy2.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy2)
        self.label_4.setStyleSheet(u"color: rgb(238, 238, 238);")

        self.horizontalLayout_8.addWidget(self.label_4)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_8.addItem(self.horizontalSpacer_6)

        self.checkBox_axis = SwitchButton(self.tool)
        self.checkBox_axis.setObjectName(u"checkBox_axis")
        sizePolicy1.setHeightForWidth(self.checkBox_axis.sizePolicy().hasHeightForWidth())
        self.checkBox_axis.setSizePolicy(sizePolicy1)
        self.checkBox_axis.setChecked(True)

        self.horizontalLayout_8.addWidget(self.checkBox_axis)


        self.verticalLayout.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(-1, 0, -1, -1)
        self.label_5 = StrongBodyLabel(self.tool)
        self.label_5.setObjectName(u"label_5")
        sizePolicy2.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy2)
        self.label_5.setStyleSheet(u"color: rgb(238, 238, 238);")

        self.horizontalLayout_9.addWidget(self.label_5)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_7)

        self.checkBox_arrow = SwitchButton(self.tool)
        self.checkBox_arrow.setObjectName(u"checkBox_arrow")
        sizePolicy1.setHeightForWidth(self.checkBox_arrow.sizePolicy().hasHeightForWidth())
        self.checkBox_arrow.setSizePolicy(sizePolicy1)
        self.checkBox_arrow.setChecked(False)

        self.horizontalLayout_9.addWidget(self.checkBox_arrow)


        self.verticalLayout.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.label_2 = StrongBodyLabel(self.tool)
        self.label_2.setObjectName(u"label_2")
        sizePolicy2.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy2)
        self.label_2.setStyleSheet(u"color: rgb(238, 238, 238);")

        self.horizontalLayout_2.addWidget(self.label_2)

        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_2.addItem(self.horizontalSpacer)

        self.doubleSpinBox = DoubleSpinBox(self.tool)
        self.doubleSpinBox.setObjectName(u"doubleSpinBox")
        self.doubleSpinBox.setDecimals(1)
        self.doubleSpinBox.setMinimum(0.000000000000000)
        self.doubleSpinBox.setValue(1.000000000000000)

        self.horizontalLayout_2.addWidget(self.doubleSpinBox)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton_openscript = PushButton(self.tool)
        self.pushButton_openscript.setObjectName(u"pushButton_openscript")
        sizePolicy1.setHeightForWidth(self.pushButton_openscript.sizePolicy().hasHeightForWidth())
        self.pushButton_openscript.setSizePolicy(sizePolicy1)
        self.pushButton_openscript.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_3.addWidget(self.pushButton_openscript)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_3.addItem(self.horizontalSpacer_2)

        self.label_script = StrongBodyLabel(self.tool)
        self.label_script.setObjectName(u"label_script")
        sizePolicy2.setHeightForWidth(self.label_script.sizePolicy().hasHeightForWidth())
        self.label_script.setSizePolicy(sizePolicy2)
        self.label_script.setStyleSheet(u"color: rgb(238, 238, 238);\n"
"font: 9pt;")
        self.label_script.setWordWrap(True)

        self.horizontalLayout_3.addWidget(self.label_script)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.pushButton_runscript = PushButton(self.tool)
        self.pushButton_runscript.setObjectName(u"pushButton_runscript")
        sizePolicy1.setHeightForWidth(self.pushButton_runscript.sizePolicy().hasHeightForWidth())
        self.pushButton_runscript.setSizePolicy(sizePolicy1)
        self.pushButton_runscript.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_4.addWidget(self.pushButton_runscript)

        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer_3)

        self.widget_circle = GLAddon_circling(self.tool)
        self.widget_circle.setObjectName(u"widget_circle")
        self.widget_circle.setMinimumSize(QSize(24, 24))

        self.horizontalLayout_4.addWidget(self.widget_circle)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.tableWidget = TableWidget(self.tool)
        if (self.tableWidget.columnCount() < 1):
            self.tableWidget.setColumnCount(1)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setMaximumSize(QSize(16777215, 16777215))
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(80)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)

        self.verticalLayout.addWidget(self.tableWidget)

        self.label_info = QTextBrowser(self.tool)
        self.label_info.setObjectName(u"label_info")
        self.label_info.setStyleSheet(u"QTextBrowser\n"
"{\n"
"    background-color: #00000000;\n"
"    \n"
"    border-radius: 6px;\n"
"    border: 0px;\n"
"	font: 1000 8pt;\n"
"	\n"
"	color: rgb(238, 238, 238);\n"
"}\n"
"QScrollBar:vertical \n"
"{\n"
"    background-color: #00000000;\n"
"    \n"
"    width: 12px;\n"
"border-radius:6px;\n"
"}\n"
"QScrollBar::handle:vertical \n"
"{\n"
"    background-color: #00000000; \n"
"    width: 12px;\n"
"    border-radius:6px;\n"
"}\n"
"\n"
"QScrollBar::add-line:vertical\n"
"{\n"
"    width: 0px;\n"
"}\n"
"QScrollBar::sub-line:vertical\n"
"{\n"
"    width: 0px;\n"
"}\n"
"QScrollBar::add-page:vertical\n"
"{\n"
"    background-color: #00000000;\n"
"}\n"
"QScrollBar::sub-page:vertical\n"
"{\n"
"    background-color: #00000000;\n"
"}\n"
"\n"
"QScrollBar:horizontal \n"
"{\n"
"    background-color: #00000000;\n"
"    \n"
"    width: 12px;\n"
"border-radius:6px;\n"
"}\n"
"\n"
"QScrollBar::handle:horizontal \n"
"{\n"
"    background-color: #00000000; \n"
"    width: 12px;\n"
"    border-radius:6px;\n"
"}\n"
"\n"
"QScr"
                        "ollBar::add-line:horizontal\n"
"{\n"
"    width: 0px;\n"
"}\n"
"QScrollBar::sub-line:horizontal\n"
"{\n"
"    width: 0px;\n"
"}\n"
"QScrollBar::add-page:horizontal\n"
"{\n"
"    background-color: #00000000;\n"
"}\n"
"QScrollBar::sub-page:horizontal\n"
"{\n"
"    background-color: #00000000;\n"
"}")
        self.label_info.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.label_info.setLineWrapMode(QTextEdit.NoWrap)

        self.verticalLayout.addWidget(self.label_info)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_7.addItem(self.horizontalSpacer_5)

        self.toolButton_theme = DropDownToolButton(self.tool)
        self.toolButton_theme.setObjectName(u"toolButton_theme")

        self.horizontalLayout_7.addWidget(self.toolButton_theme)


        self.verticalLayout.addLayout(self.horizontalLayout_7)


        self.horizontalLayout_5.addWidget(self.tool)

        self.openGLWidget = GLWidget(self.centralwidget)
        self.openGLWidget.setObjectName(u"openGLWidget")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.openGLWidget.sizePolicy().hasHeightForWidth())
        self.openGLWidget.setSizePolicy(sizePolicy3)
        self.openGLWidget.setMinimumSize(QSize(300, 0))

        self.horizontalLayout_5.addWidget(self.openGLWidget)


        self.verticalLayout_3.addLayout(self.horizontalLayout_5)

        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.pushButton_openfolder.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u672c\u5730\u6587\u4ef6\u5939", None))
        self.pushButton_openremotefolder.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u8fdc\u7a0b\u6587\u4ef6\u5939", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"\u8ddf\u968f\u7269\u4f53", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"CheckBox", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"\u7f51\u683c", None))
        self.checkBox_axis.setText(QCoreApplication.translate("MainWindow", u"CheckBox", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"\u7bad\u5934", None))
        self.checkBox_arrow.setText(QCoreApplication.translate("MainWindow", u"CheckBox", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"\u5c3a\u5ea6", None))
        self.doubleSpinBox.setPrefix(QCoreApplication.translate("MainWindow", u"\u653e\u5927", None))
        self.doubleSpinBox.setSuffix(QCoreApplication.translate("MainWindow", u"\u500d", None))
        self.pushButton_openscript.setText(QCoreApplication.translate("MainWindow", u"\u6253\u5f00\u811a\u672c", None))
        self.label_script.setText(QCoreApplication.translate("MainWindow", u"...", None))
        self.pushButton_runscript.setText(QCoreApplication.translate("MainWindow", u"\u8fd0\u884c\u811a\u672c", None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"\u6587\u4ef6", None));
        self.toolButton_theme.setText("")
    # retranslateUi

