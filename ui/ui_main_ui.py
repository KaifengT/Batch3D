# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_main.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
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
from qfluentwidgets import (DropDownToolButton, PrimaryPushButton, PushButton, SpinBox,
    StrongBodyLabel, SwitchButton, TableWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(986, 659)
        MainWindow.setMinimumSize(QSize(200, 200))
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
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tool.sizePolicy().hasHeightForWidth())
        self.tool.setSizePolicy(sizePolicy)
        self.tool.setMaximumSize(QSize(370, 16777215))
        self.tool.setStyleSheet(u"\n"
"background-color: rgb(32, 30, 28);\n"
"border-radius:10px;\n"
"")
        self.verticalLayout = QVBoxLayout(self.tool)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton_openfolder = PrimaryPushButton(self.tool)
        self.pushButton_openfolder.setObjectName(u"pushButton_openfolder")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.pushButton_openfolder.sizePolicy().hasHeightForWidth())
        self.pushButton_openfolder.setSizePolicy(sizePolicy1)
        self.pushButton_openfolder.setMinimumSize(QSize(20, 0))

        self.horizontalLayout.addWidget(self.pushButton_openfolder)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.pushButton_openremotefolder = PrimaryPushButton(self.tool)
        self.pushButton_openremotefolder.setObjectName(u"pushButton_openremotefolder")
        sizePolicy1.setHeightForWidth(self.pushButton_openremotefolder.sizePolicy().hasHeightForWidth())
        self.pushButton_openremotefolder.setSizePolicy(sizePolicy1)
        self.pushButton_openremotefolder.setMinimumSize(QSize(20, 0))

        self.verticalLayout.addWidget(self.pushButton_openremotefolder)

        self.tableWidget = TableWidget(self.tool)
        if (self.tableWidget.columnCount() < 2):
            self.tableWidget.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setMaximumSize(QSize(16777215, 16777215))
        self.tableWidget.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget.horizontalHeader().setVisible(True)
        self.tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(80)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)

        self.verticalLayout.addWidget(self.tableWidget)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setContentsMargins(-1, 12, -1, -1)
        self.label_5 = StrongBodyLabel(self.tool)
        self.label_5.setObjectName(u"label_5")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy2)
        self.label_5.setStyleSheet(u"color: rgb(238, 238, 238);")

        self.horizontalLayout_9.addWidget(self.label_5)

        self.horizontalSpacer_7 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_9.addItem(self.horizontalSpacer_7)

        self.checkBox_arrow = SwitchButton(self.tool)
        self.checkBox_arrow.setObjectName(u"checkBox_arrow")
        sizePolicy1.setHeightForWidth(self.checkBox_arrow.sizePolicy().hasHeightForWidth())
        self.checkBox_arrow.setSizePolicy(sizePolicy1)
        self.checkBox_arrow.setChecked(False)

        self.horizontalLayout_9.addWidget(self.checkBox_arrow)


        self.verticalLayout.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.label_3 = StrongBodyLabel(self.tool)
        self.label_3.setObjectName(u"label_3")
        sizePolicy2.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy2)
        self.label_3.setStyleSheet(u"color: rgb(238, 238, 238);")

        self.horizontalLayout_10.addWidget(self.label_3)

        self.horizontalSpacer_8 = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.horizontalLayout_10.addItem(self.horizontalSpacer_8)

        self.spinBox = SpinBox(self.tool)
        self.spinBox.setObjectName(u"spinBox")
        self.spinBox.setEnabled(False)
        self.spinBox.setMinimum(-1)
        self.spinBox.setMaximum(-1)
        self.spinBox.setValue(-1)

        self.horizontalLayout_10.addWidget(self.spinBox)


        self.verticalLayout.addLayout(self.horizontalLayout_10)

        self.tableWidget_obj = TableWidget(self.tool)
        if (self.tableWidget_obj.columnCount() < 3):
            self.tableWidget_obj.setColumnCount(3)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.tableWidget_obj.setHorizontalHeaderItem(0, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.tableWidget_obj.setHorizontalHeaderItem(1, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.tableWidget_obj.setHorizontalHeaderItem(2, __qtablewidgetitem4)
        self.tableWidget_obj.setObjectName(u"tableWidget_obj")
        self.tableWidget_obj.setMaximumSize(QSize(16777215, 16777215))
        self.tableWidget_obj.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget_obj.setRowCount(0)
        self.tableWidget_obj.setColumnCount(3)
        self.tableWidget_obj.horizontalHeader().setVisible(True)
        self.tableWidget_obj.horizontalHeader().setCascadingSectionResizes(True)
        self.tableWidget_obj.horizontalHeader().setMinimumSectionSize(30)
        self.tableWidget_obj.horizontalHeader().setDefaultSectionSize(30)
        self.tableWidget_obj.horizontalHeader().setStretchLastSection(True)
        self.tableWidget_obj.verticalHeader().setVisible(False)

        self.verticalLayout.addWidget(self.tableWidget_obj)

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
        self.label_info.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.label_info.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.label_info.setAutoFormatting(QTextEdit.AutoFormattingFlag.AutoAll)
        self.label_info.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)

        self.verticalLayout.addWidget(self.label_info)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton_openscript = PushButton(self.tool)
        self.pushButton_openscript.setObjectName(u"pushButton_openscript")
        sizePolicy1.setHeightForWidth(self.pushButton_openscript.sizePolicy().hasHeightForWidth())
        self.pushButton_openscript.setSizePolicy(sizePolicy1)
        self.pushButton_openscript.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_3.addWidget(self.pushButton_openscript)


        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.pushButton_runscript = PushButton(self.tool)
        self.pushButton_runscript.setObjectName(u"pushButton_runscript")
        sizePolicy1.setHeightForWidth(self.pushButton_runscript.sizePolicy().hasHeightForWidth())
        self.pushButton_runscript.setSizePolicy(sizePolicy1)
        self.pushButton_runscript.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_4.addWidget(self.pushButton_runscript)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.pushButton_openconsole = PushButton(self.tool)
        self.pushButton_openconsole.setObjectName(u"pushButton_openconsole")
        sizePolicy1.setHeightForWidth(self.pushButton_openconsole.sizePolicy().hasHeightForWidth())
        self.pushButton_openconsole.setSizePolicy(sizePolicy1)
        self.pushButton_openconsole.setMinimumSize(QSize(20, 0))

        self.horizontalLayout_7.addWidget(self.pushButton_openconsole)

        self.pushButton_opendetail = PushButton(self.tool)
        self.pushButton_opendetail.setObjectName(u"pushButton_opendetail")
        sizePolicy1.setHeightForWidth(self.pushButton_opendetail.sizePolicy().hasHeightForWidth())
        self.pushButton_opendetail.setSizePolicy(sizePolicy1)
        self.pushButton_opendetail.setMinimumSize(QSize(20, 0))

        self.horizontalLayout_7.addWidget(self.pushButton_opendetail)

        self.toolButton_theme = DropDownToolButton(self.tool)
        self.toolButton_theme.setObjectName(u"toolButton_theme")

        self.horizontalLayout_7.addWidget(self.toolButton_theme)


        self.verticalLayout.addLayout(self.horizontalLayout_7)


        self.horizontalLayout_5.addWidget(self.tool)

        self.openGLWidget = GLWidget(self.centralwidget)
        self.openGLWidget.setObjectName(u"openGLWidget")
        sizePolicy3 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        self.pushButton_openfolder.setText(QCoreApplication.translate("MainWindow", u"Local Folder", None))
        self.pushButton_openremotefolder.setText(QCoreApplication.translate("MainWindow", u"Remote Folder", None))
        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"File", None));
        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Modify Time", None));
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Arrow", None))
        self.checkBox_arrow.setText(QCoreApplication.translate("MainWindow", u"CheckBox", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Slice", None))
        ___qtablewidgetitem2 = self.tableWidget_obj.horizontalHeaderItem(0)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Vis", None));
        ___qtablewidgetitem3 = self.tableWidget_obj.horizontalHeaderItem(1)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"ID", None));
        ___qtablewidgetitem4 = self.tableWidget_obj.horizontalHeaderItem(2)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Size", None));
        self.pushButton_openscript.setText(QCoreApplication.translate("MainWindow", u"Open Script", None))
        self.pushButton_runscript.setText(QCoreApplication.translate("MainWindow", u"Exec Script ", None))
        self.pushButton_openconsole.setText(QCoreApplication.translate("MainWindow", u"console", None))
        self.pushButton_opendetail.setText(QCoreApplication.translate("MainWindow", u"Info", None))
        self.toolButton_theme.setText("")
    # retranslateUi

