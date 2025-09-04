# save_pcd_window_trimesh_multi.py

import sys
import numpy as np
from pathlib import Path

import trimesh
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
)
from PySide6.QtCore import Qt
from qfluentwidgets import (
    BodyLabel,
    PushButton,
    PrimaryPushButton,
    InfoBar,
    InfoBarPosition,
    ListWidget,
)

from b3d import b3d

class PCDExportWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.current_data = {}  # 缓存 workspace 数据 {name: (N,3) numpy array}
        self.init_ui()
        self.load_workspace_from_b3d(b3d.getWorkspaceObj())  # 初始化加载一次

        # 连接 b3d 的 workspace 更新信号
        try:
            b3d.workspaceUpdatedSignal.connect(self.load_workspace_from_b3d)
        except Exception as e:
            print(f"无法连接 workspaceUpdatedSignal: {e}")

    def init_ui(self):
        self.setWindowTitle("点云批量导出工具")
        self.resize(500, 400)

        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title = BodyLabel("点云导出为 .ply 文件")
        layout.addWidget(title)

        # 描述
        desc = BodyLabel("从列表中选择一个或多个点云对象。\n"
                        "支持 Ctrl + 点击（多选）、Shift + 点击（连续选择）。")
        desc.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(desc)

        # 多选列表
        self.list_label = BodyLabel("选择点云对象：")
        layout.addWidget(self.list_label)

        self.pcd_listwidget = ListWidget()
        self.pcd_listwidget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)  # 启用多选
        layout.addWidget(self.pcd_listwidget)

        # 按钮布局
        btn_layout = QHBoxLayout()
        self.refresh_btn = PushButton("刷新列表")
        self.export_btn = PrimaryPushButton("导出选中的点云")

        self.refresh_btn.clicked.connect(self.on_refresh)
        self.export_btn.clicked.connect(self.on_export)

        btn_layout.addWidget(self.refresh_btn)
        btn_layout.addWidget(self.export_btn)
        layout.addLayout(btn_layout)

    def load_workspace_from_b3d(self, data: dict):
        """接收 b3d workspace 更新，更新列表"""
        if not isinstance(data, dict):
            return

        self.current_data = data
        self.pcd_listwidget.clear()

        # 添加所有对象名称
        for name in data.keys():
            item = QListWidgetItem(name)
            self.pcd_listwidget.addItem(item)

    def on_refresh(self):
        """手动刷新对象列表"""
        try:
            data = b3d.getWorkspaceObj()
            self.load_workspace_from_b3d(data)
            InfoBar.success(
                title="刷新成功",
                content="点云列表已更新。",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
        except Exception as e:
            InfoBar.error(
                title="刷新失败",
                content=f"错误: {str(e)}",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=-1,
                parent=self
            )

    def on_export(self):
        """导出选中的多个点云，合并为一个 .ply 文件"""
        selected_items = self.pcd_listwidget.selectedItems()
        if not selected_items:
            InfoBar.warning(
                title="未选择",
                content="请至少选择一个点云对象。",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
            return

        # 获取选中的对象名称
        selected_names = [item.text() for item in selected_items]
        points_list = []
        used_names = []

        for name in selected_names:
            if name not in self.current_data:
                continue
            points = self.current_data[name].reshape(-1, 3)
            if points.ndim != 2 or points.shape[1] != 3:
                InfoBar.error(
                    title="数据格式错误",
                    content=f"'{name}' 不是有效的 (N,3) 点云数据。",
                    orient="vertical",
                    isClosable=True,
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=2000,
                    parent=self
                )
                return
            points_list.append(points)
            used_names.append(name)

        if len(points_list) == 0:
            InfoBar.error(
                title="无有效数据",
                content="没有可用的点云数据。",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
            return

        # 合并点云
        try:
            merged_points = np.vstack(points_list)
        except Exception as e:
            InfoBar.error(
                title="合并失败",
                content=f"点云合并失败: {str(e)}",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=-1,
                parent=self
            )
            return

        if merged_points.shape[0] == 0:
            InfoBar.error(
                title="空点云",
                content="合并后的点云为空，无法保存。",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self
            )
            return

        # 构建 trimesh 点云
        try:
            trimesh_pcd = trimesh.PointCloud(vertices=merged_points)
        except Exception as e:
            InfoBar.error(
                title="构建点云失败",
                content=f"无法构建 trimesh 点云: {str(e)}",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=-1,
                parent=self
            )
            return

        # 选择保存路径
        default_name = "_".join(used_names[:3]) + ("_etc" if len(used_names) > 3 else "")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存合并的点云文件",
            f"{default_name}.ply",
            "PLY 文件 (*.ply);;所有文件 (*)"
        )

        if not file_path:
            return  # 用户取消

        file_path = Path(file_path)
        if file_path.suffix.lower() != ".ply":
            file_path = file_path.with_suffix(".ply")

        # 保存
        try:
            trimesh_pcd.export(str(file_path))
            InfoBar.success(
                title="保存成功",
                content=f"已合并 {len(used_names)} 个点云并保存至:\n{file_path}",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=4000,
                parent=self
            )
        except Exception as e:
            InfoBar.error(
                title="保存失败",
                content=f"保存失败: {str(e)}",
                orient="vertical",
                isClosable=True,
                position=InfoBarPosition.TOP_RIGHT,
                duration=-1,
                parent=self
            )


# ========================
# 显示窗口（实际插件环境中使用）
# ========================

window = PCDExportWidget()
window.show()