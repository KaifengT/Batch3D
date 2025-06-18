# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui_remote.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
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
    QLineEdit, QSizePolicy, QSpacerItem, QTableWidgetItem,
    QVBoxLayout, QWidget)

from qfluentwidgets import (ComboBox, LineEdit, PushButton, TableWidget)

class Ui_RemoteWidget(object):
    def setupUi(self, RemoteWidget):
        if not RemoteWidget.objectName():
            RemoteWidget.setObjectName(u"RemoteWidget")
        RemoteWidget.setWindowModality(Qt.ApplicationModal)
        RemoteWidget.resize(497, 480)
        RemoteWidget.setStyleSheet(u"\n"
"border-top-left-radius: 2px;\n"
"border-top-right-radius: 2px;\n"
"border-bottom-left-radius: 2px;\n"
"border-bottom-right-radius: 2px;")
        self.verticalLayout = QVBoxLayout(RemoteWidget)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_host = LineEdit(RemoteWidget)
        self.lineEdit_host.setObjectName(u"lineEdit_host")

        self.horizontalLayout.addWidget(self.lineEdit_host)

        self.lineEdit_port = LineEdit(RemoteWidget)
        self.lineEdit_port.setObjectName(u"lineEdit_port")

        self.horizontalLayout.addWidget(self.lineEdit_port)


        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.lineEdit_username = LineEdit(RemoteWidget)
        self.lineEdit_username.setObjectName(u"lineEdit_username")

        self.horizontalLayout_2.addWidget(self.lineEdit_username)

        self.lineEdit_passwd = LineEdit(RemoteWidget)
        self.lineEdit_passwd.setObjectName(u"lineEdit_passwd")
        self.lineEdit_passwd.setEchoMode(QLineEdit.Password)

        self.horizontalLayout_2.addWidget(self.lineEdit_passwd)


        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.pushButton_connect = PushButton(RemoteWidget)
        self.pushButton_connect.setObjectName(u"pushButton_connect")

        self.verticalLayout.addWidget(self.pushButton_connect)

        self.lineEdit_dir = LineEdit(RemoteWidget)
        self.lineEdit_dir.setObjectName(u"lineEdit_dir")

        self.verticalLayout.addWidget(self.lineEdit_dir)

        self.tableWidget = TableWidget(RemoteWidget)
        if (self.tableWidget.columnCount() < 1):
            self.tableWidget.setColumnCount(1)
        self.tableWidget.setObjectName(u"tableWidget")
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.setColumnCount(1)
        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.verticalHeader().setVisible(False)

        self.verticalLayout.addWidget(self.tableWidget)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout_4.addItem(self.horizontalSpacer)

        self.comboBox = ComboBox(RemoteWidget)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")

        self.horizontalLayout_4.addWidget(self.comboBox)


        self.verticalLayout.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.pushButton_cancel = PushButton(RemoteWidget)
        self.pushButton_cancel.setObjectName(u"pushButton_cancel")

        self.horizontalLayout_3.addWidget(self.pushButton_cancel)

        self.pushButton_openfolder = PushButton(RemoteWidget)
        self.pushButton_openfolder.setObjectName(u"pushButton_openfolder")

        self.horizontalLayout_3.addWidget(self.pushButton_openfolder)


        self.verticalLayout.addLayout(self.horizontalLayout_3)


        self.retranslateUi(RemoteWidget)

        QMetaObject.connectSlotsByName(RemoteWidget)
    # setupUi

    def retranslateUi(self, RemoteWidget):
        RemoteWidget.setWindowTitle(QCoreApplication.translate("RemoteWidget", u"Remote Folder", None))
        self.lineEdit_host.setText("")
        self.lineEdit_port.setInputMask(QCoreApplication.translate("RemoteWidget", u"00000", None))
        self.lineEdit_port.setText(QCoreApplication.translate("RemoteWidget", u"22", None))
        self.lineEdit_username.setText(QCoreApplication.translate("RemoteWidget", u"root", None))
        self.lineEdit_passwd.setInputMask("")
        self.lineEdit_passwd.setText("")
        self.pushButton_connect.setText(QCoreApplication.translate("RemoteWidget", u"Connect Server", None))
        self.lineEdit_dir.setText(QCoreApplication.translate("RemoteWidget", u"/root", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("RemoteWidget", u"Current Dir", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("RemoteWidget", u"Recursive Open", None))

        self.comboBox.setCurrentText(QCoreApplication.translate("RemoteWidget", u"Current Dir", None))
        self.pushButton_cancel.setText(QCoreApplication.translate("RemoteWidget", u"Cancel", None))
        self.pushButton_openfolder.setText(QCoreApplication.translate("RemoteWidget", u"Accept", None))
    # retranslateUi

