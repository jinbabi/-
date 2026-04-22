# -*- coding: utf-8 -*-
# 导入Python标准库和PySide6相关模块
import sys          # 系统相关的模块，例如获取命令行参数
import time         # 时间处理模块，用于日志中的时间戳
import cv2          # OpenCV库，用于图像处理和视频显示
from PySide6.QtWidgets import (
    # 窗口和布局组件
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QStackedWidget,
    # 按钮和标签组件
    QPushButton, QLabel,
    # 表格组件
    QTableWidget, QTableWidgetItem,
    # 框架和分组组件
    QFrame, QGroupBox,
    # 对话框组件
    QMessageBox, QTextEdit, QDialog,
    # 标签页组件
    QTabWidget,
    # 表单布局组件
    QFormLayout,
    # 输入框组件
    QLineEdit, QCheckBox, QComboBox,
    # 输入对话框
    QInputDialog,
    # 滚动区域组件
    QScrollArea  #  新增：导入 QScrollArea
)
from PySide6.QtGui import (
    # 绘图相关
    QColor, QPainter, QBrush, QFont, QPixmap, QPen, QLinearGradient, QPainterPath
)
from PySide6.QtCore import Qt, QRectF, Signal, QTimer # Qt核心常量、信号
from functools import partial # 用于创建偏函数，方便传递参数给信号连接
from vision_robot_controller import VisionRobotController

# ========== 自定义 Logo 组件（安全绘制） ==========
class LogoWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 60)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 24, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, "D")

# ========== 网格单元格按钮 ==========
class GridCellButton(QPushButton):
    cell_clicked = Signal(str)

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        self.setFixedSize(30, 30)
        self.setText(label)
        self.setFont(QFont("Arial", 6))
        self.setStyleSheet(self.get_style(False))
        self.clicked.connect(lambda: self.cell_clicked.emit(self.label))

    def set_filled(self, filled: bool):
        self.setStyleSheet(self.get_style(filled))

    def get_style(self, filled: bool):
        if filled:
            return """
            QPushButton {
                background-color: #00ff55;
                color: black;
                border: 1px solid #00aa33;
                border-radius: 15px;
                font-size: 8px;
            }
            QPushButton:pressed {
                background-color: #00dd44;
            }
            """
        else:
            return """
            QPushButton {
                background-color: #222233;
                color: #aaaaaa;
                border: 1px solid #444455;
                border-radius: 15px;
                font-size: 8px;
            }
            QPushButton:pressed {
                background-color: #333344;
            }
            """

# ========== 网格区域 ==========
class GridWidget(QWidget):
    def __init__(self, rows=12, cols=8, label="", parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.label_text = label
        self.cells = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(2)

        col_labels = [chr(ord('A') + i) for i in range(self.cols)]
        for r in range(self.rows):
            logical_row = self.rows - r
            for c in range(self.cols):
                cell_label = f"{logical_row}{col_labels[c]}"
                btn = GridCellButton(cell_label)
                self.cells[cell_label] = btn
                grid_layout.addWidget(btn, r, c)

        layout.addLayout(grid_layout)

        label = QLabel(self.label_text)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(label)

    def set_cell_filled(self, label: str, filled: bool):
        if label in self.cells:
            self.cells[label].set_filled(filled)

    def clear_all(self):
        for btn in self.cells.values():
            btn.set_filled(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#1a1a2e")))
        painter.drawRect(self.rect())
        super().paintEvent(event)

# ========== 图标按钮 ==========
class RoundedIconWidget(QPushButton):
    def __init__(self, icon_pixmap=None, text="", parent=None):
        super().__init__(parent)
        self.icon_pixmap = icon_pixmap
        self.text = text
        self.setFixedSize(60, 70)
        self.setStyleSheet("background: transparent; border: none;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(5, 5, -5, -5)
        gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        gradient.setColorAt(0, QColor("#00aa7f"))
        gradient.setColorAt(1, QColor("#00cc99"))
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 20, 20)

        if self.icon_pixmap:
            icon_rect = QRectF(15, 10, 30, 30)
            painter.drawPixmap(icon_rect.toRect(), self.icon_pixmap.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.text:
            painter.setPen(QColor("white"))
            painter.setFont(QFont("Arial", 8, QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter | Qt.AlignBottom, self.text)
        painter.end()

# ========== 主窗口 ==========
class NewMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowTitle("自动化挑管系统")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #1e1e2e; color: white;")
        self.controller = VisionRobotController()
        self.manual_pick_active = False
        self.init_ui()
        # ========== ✅ 新增：连接比对结果信号 ==========
        self.controller.comparison_result_ready.connect(self.handle_comparison_result)
        # ========== ✅ 新增：连接新任务接收信号 ==========
        self.controller.new_task_received.connect(self.add_new_task_to_table)
        self._last_comparison = None  # 临时存储比对结果

        # ========== 信号连接 ==========
        self.controller.image_updated.connect(self.update_image)
        self.controller.log_message.connect(self.update_log)
        self.controller.tube_coords_updated.connect(self.on_tube_selected)
        self.controller.robot_state_updated.connect(self.update_robot_state_ui)
        self.controller.gripper_state_updated.connect(self.update_gripper_state_ui)
        self.controller.tube_detection_updated.connect(self.update_tube_detection_status)
        self.controller.barcode_data_received.connect(self.show_barcode_scan_results)
        self.controller.start_robot_state_thread()
        self.on_box_type_changed(self.controller.selected_box_type)
        self.on_square_type_changed(self.controller.selected_square_type)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        sidebar = self.create_sidebar()
        main_layout.addWidget(sidebar)

        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget, 4)

        self.create_main_page()
        self.create_settings_page()
        self.create_camera_page()
        self.create_alert_page()
        self.create_chart_page()
        self.stacked_widget.setCurrentIndex(0)

    # ========== ✅ 新增：处理比对结果 ==========
    def handle_comparison_result(self, mapping, box_code, found_dict, missing_list):
        self._last_comparison = {
            'found': found_dict,
            'missing': missing_list
        }

    # ========== ✅ 新增：向任务表添加新行 ==========
    def add_new_task_to_table(self, task_id, task_type, task_code, task_status, receive_time):
        """将接收到的新任务信息添加到任务列表表格中。"""
        row_count = self.task_table.rowCount()
        self.task_table.insertRow(row_count)
        self.task_table.setItem(row_count, 0, QTableWidgetItem(task_id))
        self.task_table.setItem(row_count, 1, QTableWidgetItem(task_type))
        self.task_table.setItem(row_count, 2, QTableWidgetItem(task_code))
        self.task_table.setItem(row_count, 3, QTableWidgetItem(task_status))
        self.task_table.setItem(row_count, 4, QTableWidgetItem(receive_time))
        self.task_table.scrollToItem(self.task_table.item(row_count, 0), QTableWidget.PositionAtBottom)

    # ========== ✅ 修改：扫码结果显示比对结果 ==========
    def show_barcode_scan_results(self, mapping, box_code):
        dialog = QDialog(self)
        dialog.setWindowTitle("扫码器结果")
        dialog.setModal(False)  # ✅ 改为非模态
        dialog.setFixedSize(800, 700)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # 显示盒码
        box_label = QLabel(f"<b>盒码:</b> {box_code}")
        box_label.setStyleSheet("color: #00ffaa; font-size: 16px;")
        layout.addWidget(box_label)

        # 比对结果标签（动态更新）
        self._comparison_label = QLabel("<b>比对结果:</b> <span style='color: gray;'>正在比对...</span>")
        self._comparison_label.setWordWrap(True)
        self._comparison_label.setStyleSheet("font-size: 13px; padding: 8px; background-color: #2a2a3a; border-radius: 5px;")
        layout.addWidget(self._comparison_label)

        # 刷新比对按钮
        refresh_btn = QPushButton("刷新比对")
        refresh_btn.clicked.connect(lambda: self._refresh_comparison_result(mapping, box_code))
        layout.addWidget(refresh_btn)

        # 管码表格（8x6）
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)

        rows = list(range(8, 0, -1))
        cols = ['A', 'B', 'C', 'D', 'E', 'F']
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                pos = f"{row}{col}"
                code = mapping.get(pos, "NoRead")
                container = QWidget()
                container.setStyleSheet("background-color: #222233; border: 1px solid #444455; padding: 5px;")
                container_layout = QVBoxLayout(container)
                container_layout.setContentsMargins(5, 5, 5, 5)
                pos_label = QLabel(pos)
                pos_label.setAlignment(Qt.AlignCenter)
                pos_label.setStyleSheet("color: white; font-weight: bold;")
                code_label = QLabel(code)
                code_label.setAlignment(Qt.AlignCenter)
                code_label.setStyleSheet("color: #00ffaa;")
                container_layout.addWidget(pos_label)
                container_layout.addWidget(code_label)
                grid_layout.addWidget(container, row_idx, col_idx)

        scroll_layout.addLayout(grid_layout)
        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)
        dialog.show()  # 非模态用 show()

        # 首次比对
        self._refresh_comparison_result(mapping, box_code)

    # ========== ✅ 新增：刷新比对结果 ==========
    def _refresh_comparison_result(self, mapping, box_code):
        """根据当前目标刷新比对结果"""
        found_tubes = {}
        missing_tubes = []
        target_box = self.controller.target_box_code
        target_tubes = self.controller.target_tube_codes

        if not target_tubes:
            self._comparison_label.setText("<b>比对结果:</b> <span style='color: orange;'>请先在主界面选择目标试管</span>")
            return

        if target_box != box_code:
            self._comparison_label.setText(f"<b>比对结果:</b> <span style='color: red;'>盒码不匹配! 目标:{target_box}</span>")
            return

        for target in target_tubes:
            found = False
            for pos, code in mapping.items():
                if code == target:
                    found_tubes[pos] = code
                    found = True
                    break
            if not found:
                missing_tubes.append(target)

        parts = []
        if missing_tubes:
            parts.append(f"<span style='color: orange;'>未找到: {', '.join(missing_tubes)}</span>")
        if found_tubes:
            found_str = ", ".join([f"{pos}: {code}" for pos, code in found_tubes.items()])
            parts.append(f"<span style='color: green;'>已找到: {found_str}</span>")
        if not parts:
            parts.append("<span style='color: gray;'>无匹配管码</span>")

        comparison_text = "<b>比对结果:</b> " + " | ".join(parts)
        self._comparison_label.setText(comparison_text)

    # ========== 页面（保持不变）==========
    def create_main_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        type_layout = QHBoxLayout()
        type_layout.setContentsMargins(10, 0, 10, 10)

        self.box_type_combo = QComboBox()
        tube_types = list(self.controller.tube_config.keys()) if self.controller.tube_config else [
            "8*6试管盒", "12*8试管盒(2ml)", "12*8试管盒(1ml)", "4*4试管盒"
        ]
        self.box_type_combo.addItems(tube_types)
        self.box_type_combo.setCurrentText(self.controller.selected_box_type)
        self.box_type_combo.currentTextChanged.connect(self.on_box_type_changed)

        self.square_type_combo = QComboBox()
        square_types = list(self.controller.square_grid_config.keys()) if self.controller.square_grid_config else [
            "8*6空盒", "12*8空盒(2ml)", "12*8试管盒(1ml)", "4*4空盒"
        ]
        self.square_type_combo.addItems(square_types)
        self.square_type_combo.setCurrentText(self.controller.selected_square_type)
        self.square_type_combo.currentTextChanged.connect(self.on_square_type_changed)

        type_layout.addWidget(QLabel("试管盒:"))
        type_layout.addWidget(self.box_type_combo)
        type_layout.addWidget(QLabel("空盒:"))
        type_layout.addWidget(self.square_type_combo)
        type_layout.addStretch()
        layout.addLayout(type_layout)

        top_row = QHBoxLayout()
        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels(["任务编号", "任务类型", "任务编码", "任务状态", "接收时间"])
        self.task_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a2e; color: #00ffaa; gridline-color: #333344; border: 1px solid #444455; }
            QHeaderView::section { background-color: #1a1a2e; color: #00ffaa; padding: 5px; border: 1px solid #444455; }
        """)
        self.task_table.horizontalHeader().setStretchLastSection(True)
        self.task_table.verticalHeader().setVisible(False)
        for i, w in enumerate([120, 100, 100, 100, 150]):
            self.task_table.setColumnWidth(i, w)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(300, 200)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid #444455;")
        top_row.addWidget(self.task_table, 4)
        top_row.addWidget(self.video_label, 1)

        middle_row = QHBoxLayout()
        grid_and_button_layout = QGridLayout()
        grid_and_button_layout.setSpacing(10)
        grid_and_button_layout.setContentsMargins(5, 5, 5, 5)

        self.grid_A = GridWidget(rows=12, cols=8, label="挑选区域A（抓取）")
        self.grid_A.setStyleSheet("border: 1px solid #444455; padding: 5px;")
        for btn in self.grid_A.cells.values():
            btn.cell_clicked.connect(self.on_tube_cell_clicked)

        self.grid_B = GridWidget(rows=12, cols=8, label="放置区域B（空盒）")
        self.grid_B.setStyleSheet("border: 1px solid #444455; padding: 5px;")
        for btn in self.grid_B.cells.values():
            btn.cell_clicked.connect(self.on_place_cell_clicked)

        grid_and_button_layout.addWidget(self.grid_A, 0, 0, 1, 3)
        grid_and_button_layout.addWidget(self.grid_B, 0, 3, 1, 3)

        button_panel = QWidget()
        button_layout = QGridLayout(button_panel)
        button_layout.setSpacing(10)
        button_layout.setContentsMargins(10, 10, 10, 10)

        buttons = [
            ("一键功能", self.one_click_pick),
            ("手动挑管", self.manual_pick),
            ("放置样本", self.show_sample_actions),
            ("手动装盖", self.show_cover_install_actions),
            ("手动取盖", self.show_cover_actions),
            ("手动扫码", self.show_scan_actions),
            ("手臂归位", self.show_home_actions),
            ("清空空盒", self.clear_place_positions),
            ("启动相机", self.toggle_camera),
            ("发送数据", self.send_data),
        ]

        for i, (text, func) in enumerate(buttons):
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #00aa7f;
                    color: black;
                    border: 2px solid #333344;
                    border-radius: 8px;
                    padding: 10px;
                    font-weight: bold;
                    font-size: 14px;
                }
                QPushButton:hover { background-color: #00cc99; }
            """)
            btn.clicked.connect(func)
            row = i % 5
            col = i // 5
            button_layout.addWidget(btn, row, col)

        grid_and_button_layout.addWidget(button_panel, 0, 6, 1, 2)
        middle_row.addLayout(grid_and_button_layout, 1)

        layout.addLayout(top_row)
        layout.addLayout(middle_row)
        layout.addStretch()
        self.stacked_widget.addWidget(page)
        self.enable_grid_buttons(False)

    # ========== 其他页面保持不变（省略以节省篇幅）==========
    def create_settings_page(self):
        # ...（与原始文件完全一致，此处省略重复代码）
        page = QWidget()
        layout = QVBoxLayout(page)
        tab_widget = QTabWidget()

        conn_tab = QWidget()
        conn_layout = QFormLayout(conn_tab)
        self.ip_edit = QLineEdit(self.controller.ip)
        self.port_edit = QLineEdit(str(self.controller.port))
        conn_layout.addRow("IP地址:", self.ip_edit)
        conn_layout.addRow("端口:", self.port_edit)
        tab_widget.addTab(conn_tab, "连接参数")

        offset_tab = QWidget()
        offset_layout = QFormLayout(offset_tab)
        self.offset_x_edit = QLineEdit(str(self.controller.offset_x))
        self.offset_y_edit = QLineEdit(str(self.controller.offset_y))
        offset_layout.addRow("X偏移量:", self.offset_x_edit)
        offset_layout.addRow("Y偏移量:", self.offset_y_edit)
        tab_widget.addTab(offset_tab, "偏移参数")

        roi_tab = QWidget()
        roi_layout = QVBoxLayout(roi_tab)
        self.roi_checkbox = QCheckBox("启用ROI")
        self.roi_checkbox.setChecked(self.controller.use_roi)
        roi_form = QFormLayout()
        self.roi_x_edit = QLineEdit(str(self.controller.roi_x))
        self.roi_y_edit = QLineEdit(str(self.controller.roi_y))
        self.roi_width_edit = QLineEdit(str(self.controller.roi_width))
        self.roi_height_edit = QLineEdit(str(self.controller.roi_height))
        roi_form.addRow("ROI X:", self.roi_x_edit)
        roi_form.addRow("ROI Y:", self.roi_y_edit)
        roi_form.addRow("宽度:", self.roi_width_edit)
        roi_form.addRow("高度:", self.roi_height_edit)
        roi_layout.addWidget(self.roi_checkbox)
        roi_layout.addLayout(roi_form)
        tab_widget.addTab(roi_tab, "ROI设置")

        circle_tab = QWidget()
        circle_layout = QFormLayout(circle_tab)
        self.dp_edit = QLineEdit(str(self.controller.dp))
        self.min_dist_edit = QLineEdit(str(self.controller.min_dist))
        self.param1_edit = QLineEdit(str(self.controller.param1))
        self.param2_edit = QLineEdit(str(self.controller.param2))
        self.min_radius_edit = QLineEdit(str(self.controller.min_radius))
        self.max_radius_edit = QLineEdit(str(self.controller.max_radius))
        self.template_path_edit = QLineEdit(self.controller.template_path)
        self.match_threshold_edit = QLineEdit(str(self.controller.match_threshold))
        circle_layout.addRow("DP值:", self.dp_edit)
        circle_layout.addRow("最小距离:", self.min_dist_edit)
        circle_layout.addRow("Param1:", self.param1_edit)
        circle_layout.addRow("Param2:", self.param2_edit)
        circle_layout.addRow("最小半径:", self.min_radius_edit)
        circle_layout.addRow("最大半径:", self.max_radius_edit)
        circle_layout.addRow("模板路径:", self.template_path_edit)
        circle_layout.addRow("匹配阈值:", self.match_threshold_edit)
        tab_widget.addTab(circle_tab, "圆形检测")

        correction_tab = QWidget()
        correction_layout = QFormLayout(correction_tab)
        self.rotation_edit = QLineEdit(str(self.controller.rotation_angle))
        correction_layout.addRow("旋转角度:", self.rotation_edit)
        self.matrix_edit = []
        for i in range(3):
            row_layout = QHBoxLayout()
            for j in range(3):
                edit = QLineEdit(str(self.controller.H_vision[i][j]))
                edit.setFixedWidth(80)
                self.matrix_edit.append(edit)
                row_layout.addWidget(edit)
            correction_layout.addRow(f"矩阵行 {i + 1}:", row_layout)
        tab_widget.addTab(correction_tab, "矫正参数")

        square_tab = QWidget()
        square_layout = QFormLayout(square_tab)
        self.square_rows_edit = QLineEdit(str(self.controller.square_rows))
        self.square_cols_edit = QLineEdit(str(self.controller.square_cols))
        self.square_bottom_left_x_edit = QLineEdit(str(self.controller.square_bottom_left_x))
        self.square_bottom_left_y_edit = QLineEdit(str(self.controller.square_bottom_left_y))
        self.square_distance_edit = QLineEdit(str(self.controller.square_distance))
        self.square_offset_x_edit = QLineEdit(str(self.controller.square_offset_x))
        self.square_offset_y_edit = QLineEdit(str(self.controller.square_offset_y))
        square_layout.addRow("网格行数:", self.square_rows_edit)
        square_layout.addRow("网格列数:", self.square_cols_edit)
        square_layout.addRow("左下角X:", self.square_bottom_left_x_edit)
        square_layout.addRow("左下角Y:", self.square_bottom_left_y_edit)
        square_layout.addRow("中心距离:", self.square_distance_edit)
        square_layout.addRow("X偏移:", self.square_offset_x_edit)
        square_layout.addRow("Y偏移:", self.square_offset_y_edit)
        tab_widget.addTab(square_tab, "正方形网格")

        fixed_tab = QWidget()
        fixed_layout = QFormLayout(fixed_tab)
        self.z_edit = QLineEdit(str(self.controller.z))
        self.rx_edit = QLineEdit(str(self.controller.rx))
        self.ry_edit = QLineEdit(str(self.controller.ry))
        self.rz_edit = QLineEdit(str(self.controller.rz))
        fixed_layout.addRow("Z:", self.z_edit)
        fixed_layout.addRow("RX:", self.rx_edit)
        fixed_layout.addRow("RY:", self.ry_edit)
        fixed_layout.addRow("RZ:", self.rz_edit)
        tab_widget.addTab(fixed_tab, "固定参数")

        task_tab = QWidget()
        task_layout = QFormLayout(task_tab)
        self.task_edit = QLineEdit(str(self.controller.task_number))
        self.height_edit = QLineEdit(str(self.controller.height_value))
        task_layout.addRow("任务号:", self.task_edit)
        task_layout.addRow("高度参数:", self.height_edit)
        tab_widget.addTab(task_tab, "任务参数")

        robot_state_tab = QWidget()
        robot_state_layout = QFormLayout(robot_state_tab)
        self.auto_exit_label = QLabel("未读取")
        self.ready_label = QLabel("未读取")
        self.paused_label = QLabel("未读取")
        self.running_label = QLabel("未读取")
        self.alarm_label = QLabel("未读取")
        self.data_valid_label = QLabel("未读取")
        self.digital_in_label = QLabel("未读取")
        self.digital_out_label = QLabel("未读取")
        state_labels = ["自动退出:", "已准备好:", "暂停中:", "运行状态:", "报警状态:", "数据有效性:", "数字输入:", "数字输出:"]
        state_values = [self.auto_exit_label, self.ready_label, self.paused_label, self.running_label,
                        self.alarm_label, self.data_valid_label, self.digital_in_label, self.digital_out_label]
        for lbl_text, widget in zip(state_labels, state_values):
            robot_state_layout.addRow(QLabel(lbl_text), widget)
        self.refresh_robot_state_btn = QPushButton("刷新机器人状态")
        self.refresh_robot_state_btn.clicked.connect(self.read_robot_state)
        robot_state_layout.addRow(self.refresh_robot_state_btn)
        tab_widget.addTab(robot_state_tab, "机器人状态")

        gripper_tab = QWidget()
        gripper_layout = QFormLayout(gripper_tab)
        self.gripper_31005_label = QLabel("未读取")
        gripper_layout.addRow(QLabel("31005（夹爪状态）:"), self.gripper_31005_label)
        self.refresh_gripper_btn = QPushButton("刷新夹爪状态")
        self.refresh_gripper_btn.clicked.connect(self.read_robot_state)
        gripper_layout.addRow(self.refresh_gripper_btn)
        tab_widget.addTab(gripper_tab, "夹爪状态")

        layout.addWidget(tab_widget)
        self.stacked_widget.addWidget(page)

    def create_camera_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        log_group = QGroupBox("日志信息")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #1a1a2e; color: #00ffaa;")
        log_layout.addWidget(self.log_text)
        layout.addWidget(self.image_label)
        layout.addWidget(log_group)
        self.stacked_widget.addWidget(page)

    def create_alert_page(self):
        page = self.create_placeholder_page("警告与报警")
        self.stacked_widget.addWidget(page)

    def create_chart_page(self):
        page = self.create_placeholder_page("运行图表")
        self.stacked_widget.addWidget(page)

    def create_placeholder_page(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        label = QLabel(title)
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("color: white; font-size: 24px; font-weight: bold;")
        layout.addWidget(label)
        layout.addStretch()
        return page

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setFixedWidth(80)
        sidebar.setStyleSheet("background-color: #00aa7f; border-right: 2px solid #333344;")
        layout = QVBoxLayout(sidebar)
        layout.setSpacing(10)
        layout.setContentsMargins(5, 10, 5, 10)

        logo_widget = LogoWidget()
        layout.addWidget(logo_widget)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("background-color: #008855;")
        layout.addWidget(line)

        icons_with_text = [
            ("file", "主界面"),
            ("settings", "系统设置"),
            ("camera", "实时画面"),
            ("alert", "警报中心"),
            ("chart", "运行图表"),
        ]
        for icon_type, text in icons_with_text:
            btn = RoundedIconWidget(icon_pixmap=self.create_icon_pixmap(icon_type), text=text)
            btn.clicked.connect(partial(self.switch_page, icons_with_text.index((icon_type, text))))
            layout.addWidget(btn)

        power_icon = self.create_icon_pixmap("power")
        power_btn = RoundedIconWidget(icon_pixmap=power_icon, text="关机")
        power_btn.clicked.connect(self.confirm_exit)
        layout.addStretch()
        layout.addWidget(power_btn)
        return sidebar

    def create_icon_pixmap(self, icon_type):
        pixmap = QPixmap(30, 30)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor("white"), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        if icon_type == "file":
            painter.drawRect(5, 5, 20, 20)
            painter.drawLine(25, 5, 25, 25)
            painter.drawLine(5, 25, 25, 25)
        elif icon_type == "settings":
            painter.drawEllipse(5, 5, 20, 20)
            for i in range(8):
                x = 15 + (8 if i % 2 == 0 else 0)
                y = 15 + (8 if i % 2 == 1 else 0)
                painter.drawLine(15, 15, x, y)
        elif icon_type == "camera":
            painter.drawRect(5, 10, 20, 10)
            painter.drawEllipse(10, 5, 10, 10)
        elif icon_type == "alert":
            path = QPainterPath()
            path.moveTo(15, 5)
            path.lineTo(5, 25)
            path.lineTo(25, 25)
            path.closeSubpath()
            painter.drawPath(path)
            painter.drawText(QRectF(10, 10, 10, 15), Qt.AlignCenter, "!")
        elif icon_type == "chart":
            painter.drawLine(5, 25, 5, 5)
            painter.drawLine(5, 25, 25, 25)
            painter.drawLine(5, 20, 12, 12)
            painter.drawLine(12, 12, 20, 18)
            painter.drawLine(20, 18, 25, 10)
        elif icon_type == "power":
            painter.drawArc(5, 5, 20, 20, 45 * 16, 270 * 16)
            painter.drawLine(15, 15, 15, 5)
        painter.end()
        return pixmap

    def switch_page(self, index):
        self.stacked_widget.setCurrentIndex(index)

    def confirm_exit(self):
        reply = QMessageBox.question(self, "确认关机", "确定要关闭系统吗？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.controller.shutdown()
            QApplication.quit()

    def toggle_camera(self):
        if self.controller.camera_thread and self.controller.camera_thread.isRunning():
            self.controller.stop_camera()
        else:
            self._apply_params_to_controller()
            self.controller.start_camera()

    def manual_pick(self):
        if not (self.controller.camera_thread and self.controller.camera_thread.isRunning()):
            QMessageBox.warning(self, "警告", "请先启动相机")
            return
        self.manual_pick_active = True
        self.enable_grid_buttons(True)
        QMessageBox.information(self, "手动挑管", "手动挑管模式已激活，请选择抓取和放置位置。")

    # ========== 实现一键功能 ==========
    def one_click_pick(self):
        """实现一键功能，弹出对话框提供四个子选项：一键取盖、一键扫码、一键装盖、一键挑管。"""
        dialog = QDialog(self)
        dialog.setWindowTitle("一键功能")
        dialog.setModal(True)
        dialog.setFixedSize(300, 240)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        btn_take_cover = QPushButton("一键取盖")
        btn_scan = QPushButton("一键扫码")
        btn_put_cover = QPushButton("一键装盖")
        btn_pick_tubes = QPushButton("一键挑管")  # ← 新增

        btn_take_cover.clicked.connect(lambda: self.one_click_take_cover(dialog))
        btn_scan.clicked.connect(lambda: self.one_click_scan(dialog))
        btn_put_cover.clicked.connect(lambda: self.one_click_put_cover(dialog))
        btn_pick_tubes.clicked.connect(lambda: self.one_click_auto_pick(dialog))  # ← 新增

        layout.addWidget(btn_take_cover)
        layout.addWidget(btn_scan)
        layout.addWidget(btn_put_cover)
        layout.addWidget(btn_pick_tubes)  # ← 新增

        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        dialog.setLayout(layout)
        dialog.exec()

    # ========== ✅ 新增：一键挑管 ==========
    def one_click_auto_pick(self, parent_dialog):
        """执行一键挑管：自动抓取扫码匹配到的试管，并按1A~8F顺序放置。"""
        if not hasattr(self, '_last_comparison') or not self._last_comparison:
            QMessageBox.warning(parent_dialog, "无数据", "请先执行扫码并完成比对！")
            return
        found_dict = self._last_comparison.get('found', {})
        if not found_dict:
            QMessageBox.warning(parent_dialog, "无匹配", "当前扫码结果中没有匹配到目标试管！")
            return

        pick_positions = list(found_dict.keys())
        pick_positions.sort(key=lambda x: (int(x[:-1]), x[-1]))

        # 生成放置顺序 1A~8F
        place_sequence = []
        rows = list(range(1, 9))
        cols = ['A', 'B', 'C', 'D', 'E', 'F']
        for r in rows:
            for c in cols:
                place_sequence.append(f"{r}{c}")

        if len(pick_positions) > len(place_sequence):
            QMessageBox.critical(parent_dialog, "容量超限", "目标试管数量超过空盒容量（48个）！")
            return

        self._auto_pick_queue = list(zip(pick_positions, place_sequence[:len(pick_positions)]))
        self._auto_pick_index = 0
        self._parent_dialog_for_pick = parent_dialog
        self.log_message(f"🔄 一键挑管：共 {len(self._auto_pick_queue)} 个试管待处理")
        QTimer.singleShot(100, self._execute_next_auto_pick)

    def _execute_next_auto_pick(self):
        if self._auto_pick_index >= len(self._auto_pick_queue):
            self.log_message("🎉 一键挑管流程全部完成！")
            QMessageBox.information(self._parent_dialog_for_pick, "完成", "一键挑管已全部执行完毕！")
            return

        pick_label, place_label = self._auto_pick_queue[self._auto_pick_index]
        self._auto_pick_index += 1

        try:
            rows = int(self.square_rows_edit.text())
            cols = int(self.square_cols_edit.text())
            x = float(self.square_bottom_left_x_edit.text())
            y = float(self.square_bottom_left_y_edit.text())
            d = float(self.square_distance_edit.text())
            ox = float(self.square_offset_x_edit.text())
            oy = float(self.square_offset_y_edit.text())
            self.controller.set_square_params(rows, cols, x, y, d, ox, oy)

            self.controller.select_tube(pick_label)
            self.controller.select_place_position(place_label)
            if self.controller.selected_place_label != place_label:
                raise Exception(f"放置位置 {place_label} 无效或已被占用")

            self.controller.set_task_params(6, 108)
            z = float(self.z_edit.text())
            rx = float(self.rx_edit.text())
            ry = float(self.ry_edit.text())
            rz = float(self.rz_edit.text())
            self.controller.set_fixed_pose(z, rx, ry, rz)
            self.controller.send_data()

            self.log_message(f"✅ 已发送任务6：抓取 {pick_label} → 放置 {place_label}")
            self.grid_B.set_cell_filled(place_label, True)
            QTimer.singleShot(3000, self._execute_next_auto_pick)
        except Exception as e:
            error_msg = f"❌ 一键挑管失败（第{self._auto_pick_index}个）: {str(e)}"
            self.log_message(error_msg)
            QMessageBox.critical(self._parent_dialog_for_pick, "执行失败", error_msg)

    # ========== ✅ 重构：一键取盖（非阻塞 QTimer 版） ==========
    def one_click_take_cover(self, parent_dialog):
        self.log_message(" 开始执行一键取盖流程...")
        def step1():
            try:
                self.controller.set_task_params(2, 0)
                self.controller.send_data()
                self.log_message(" 已发送任务2：取A盖。")
                QTimer.singleShot(5000, step2)
            except Exception as e:
                self.log_message(f" 一键取盖流程失败 (步骤1): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"一键取盖流程失败: {str(e)}")
        def step2():
            try:
                self.controller.set_task_params(7, 0)
                self.controller.send_data()
                self.log_message("✅ 已发送任务7：取B盖。")
                self.log_message("🎉 一键取盖流程执行完毕！")
                QMessageBox.information(parent_dialog, "操作成功", "一键取盖流程已成功执行。")
            except Exception as e:
                self.log_message(f"❌ 一键取盖流程失败 (步骤2): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"一键取盖流程失败: {str(e)}")
        QTimer.singleShot(0, step1)

    # ========== ✅ 修改：一键扫码（含任务13） ==========
    def one_click_scan(self, parent_dialog):
        self.log_message("🔄 开始执行一键扫码流程...")
        def step1():
            try:
                self.controller.set_task_params(9, 0)
                self.controller.send_data()
                self.log_message("✅ 已发送任务9：B盒到扫码区。")
                QTimer.singleShot(3000, step2)
            except Exception as e:
                self.log_message(f"❌ 一键扫码失败 (步骤1): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"一键扫码失败: {str(e)}")
        def step2():
            try:
                self.controller.start_barcode_scanning()
                self.log_message("✅ 已启动扫码器。")
                QTimer.singleShot(10000, step3)
            except Exception as e:
                self.log_message(f"❌ 一键扫码失败 (步骤2): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"启动扫码器失败: {str(e)}")
        def step3():
            try:
                self.controller.set_task_params(10, 0)
                self.controller.send_data()
                self.log_message("✅ 已发送任务10：B盒归位。")
                QTimer.singleShot(5000, step4)  # 👈 延时5秒后执行任务13
            except Exception as e:
                self.log_message(f"❌ 一键扫码失败 (步骤3): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"一键扫码失败: {str(e)}")
        def step4():  # 👈 新增步骤：任务13
            try:
                self.controller.set_task_params(13, 0)
                self.controller.send_data()
                self.log_message("✅ 已发送任务13：大夹爪转小夹爪。")
                self.log_message("🎉 一键扫码流程执行完毕！")
                QMessageBox.information(parent_dialog, "操作成功", "一键扫码流程已成功执行。")
            except Exception as e:
                self.log_message(f"❌ 一键扫码失败 (步骤4): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"切换夹爪失败: {str(e)}")
        QTimer.singleShot(0, step1)

    # ========== ✅ 重构：一键装盖（非阻塞 QTimer 版） ==========
    def one_click_put_cover(self, parent_dialog):
        self.log_message("🔄 开始执行一键装盖流程...")
        def step1():
            try:
                self.controller.set_task_params(8, 0)
                self.controller.send_data()
                self.log_message("✅ 已发送任务8：装B盖。")
                QTimer.singleShot(5000, step2)
            except Exception as e:
                self.log_message(f"❌ 一键装盖失败 (步骤1): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"一键装盖失败: {str(e)}")
        def step2():
            try:
                self.controller.set_task_params(3, 0)
                self.controller.send_data()
                self.log_message("✅ 已发送任务3：装A盖。")
                self.log_message("🎉 一键装盖流程执行完毕！")
                QMessageBox.information(parent_dialog, "操作成功", "一键装盖流程已成功执行。")
            except Exception as e:
                self.log_message(f"❌ 一键装盖失败 (步骤2): {str(e)}")
                QMessageBox.critical(parent_dialog, "操作失败", f"一键装盖失败: {str(e)}")
        QTimer.singleShot(0, step1)

    def send_task_1(self):
        self.controller.set_task_params(1, 0)
        self.controller.send_data()

    def send_task_2(self):
        self.controller.set_task_params(2, 0)
        self.controller.send_data()

    def send_task_3(self):
        self.controller.set_task_params(3, 0)
        self.controller.send_data()

    def send_task_4(self):
        self.controller.set_task_params(4, 0)
        self.controller.send_data()

    def send_task_5(self):
        self.controller.set_task_params(5, 0)
        self.controller.send_data()

    def send_task_7(self):
        self.controller.set_task_params(7, 0)
        self.controller.send_data()

    def send_task_8(self):
        self.controller.set_task_params(8, 0)
        self.controller.send_data()

    def send_task_9(self):
        self.controller.set_task_params(9, 0)
        self.controller.send_data()

    def send_task_10(self):
        self.controller.set_task_params(10, 0)
        self.controller.send_data()

    def send_task_11(self):
        self.controller.set_task_params(11, 0)
        self.controller.send_data()

    def send_task_12(self):
        self.controller.set_task_params(12, 0)
        self.controller.send_data()

    def send_task_13(self):
        self.controller.set_task_params(13, 0)
        self.controller.send_data()

    def send_task_14(self):
        self.controller.set_task_params(14, 0)
        self.controller.send_data()

    def show_cover_actions(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("手动取盖")
        dialog.setModal(True)
        dialog.setFixedSize(200, 150)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        btn_a = QPushButton("取A盖（首先）")
        btn_b = QPushButton("取B盖")
        btn_a.clicked.connect(lambda: self.send_task_2())
        btn_b.clicked.connect(lambda: self.send_task_7())
        layout.addWidget(btn_a)
        layout.addWidget(btn_b)
        dialog.setLayout(layout)
        dialog.exec()

    def show_cover_install_actions(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("手动装盖")
        dialog.setModal(True)
        dialog.setFixedSize(200, 150)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        btn_a = QPushButton("装A盖")
        btn_b = QPushButton("装B盖（首先）")
        btn_a.clicked.connect(lambda: self.send_task_3())
        btn_b.clicked.connect(lambda: self.send_task_8())
        layout.addWidget(btn_a)
        layout.addWidget(btn_b)
        dialog.setLayout(layout)
        dialog.exec()

    def show_sample_actions(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("放置样本操作")
        dialog.setModal(True)
        dialog.setFixedSize(200, 120)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        btn_a_return = QPushButton("A盒归位")
        btn_b_return = QPushButton("B盒归位")
        btn_a_return.clicked.connect(lambda: self.send_task_5())
        btn_b_return.clicked.connect(lambda: self.send_task_10())
        layout.addWidget(btn_a_return)
        layout.addWidget(btn_b_return)
        dialog.setLayout(layout)
        dialog.exec()

    def show_home_actions(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("归位操作")
        dialog.setModal(True)
        dialog.setFixedSize(280, 240)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        btn1 = QPushButton("大夹爪归位（大夹爪在下）")
        btn2 = QPushButton("小夹爪归位（小夹爪在下）")
        btn3 = QPushButton("到拍摄点（小夹爪先归位）")
        btn4 = QPushButton("大夹爪转小夹爪（大夹爪在下）")
        btn5 = QPushButton("小夹爪转大夹爪（小夹爪在下）")
        btn1.clicked.connect(lambda: self.send_task_1())
        btn2.clicked.connect(lambda: self.send_task_11())
        btn3.clicked.connect(lambda: self.send_task_12())
        btn4.clicked.connect(lambda: self.send_task_13())
        btn5.clicked.connect(lambda: self.send_task_14())
        for btn in [btn1, btn2, btn3, btn4, btn5]:
            layout.addWidget(btn)
        dialog.setLayout(layout)
        dialog.exec()

    def show_scan_actions(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("扫码操作")
        dialog.setModal(True)
        dialog.setFixedSize(220, 160)
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        btn_a_scan = QPushButton("A盒到扫码区")
        btn_b_scan = QPushButton("B盒到扫码区")
        btn_start_scan = QPushButton("开始扫码")
        btn_a_scan.clicked.connect(lambda: self.send_task_4())
        btn_b_scan.clicked.connect(lambda: self.send_task_9())
        btn_start_scan.clicked.connect(lambda: [
            self.controller.start_barcode_scanning(),
            dialog.accept()
        ])
        layout.addWidget(btn_a_scan)
        layout.addWidget(btn_b_scan)
        layout.addWidget(btn_start_scan)
        dialog.setLayout(layout)
        dialog.exec()

    def clear_place_positions(self):
        self.controller.clear_place_positions()
        self.grid_B.clear_all()
        self.log_message("🔄 空盒状态已清空")
        self.manual_pick_active = False
        self.enable_grid_buttons(False)

    def send_data(self):
        try:
            task_text, ok1 = QInputDialog.getText(
                self, "输入任务号", "任务号（整数）:", text=str(self.controller.task_number)
            )
            if not ok1 or not task_text.strip():
                return
            task = int(task_text.strip())
            height_text, ok2 = QInputDialog.getText(
                self, "输入高度参数", "高度（整数）:", text=str(self.controller.height_value)
            )
            if not ok2 or not height_text.strip():
                return
            height = int(height_text.strip())

            if task == 6:
                if not self.controller.tube_coords:
                    QMessageBox.warning(self, "错误", "请先选择抓取试管！")
                    return
                if not self.controller.place_coords:
                    QMessageBox.warning(self, "错误", "请先选择放置位置！")
                    return

                try:
                    z = float(self.z_edit.text())
                    rx = float(self.rx_edit.text())
                    ry = float(self.ry_edit.text())
                    rz = float(self.rz_edit.text())
                except ValueError:
                    QMessageBox.critical(self, "输入错误", "Z/RX/RY/RZ 必须为有效数字！")
                    return

                self.controller.set_fixed_pose(z, rx, ry, rz)

            self.controller.set_task_params(task, height)
            self.controller.send_data()
        except ValueError:
            QMessageBox.critical(self, "输入错误", "任务号和高度必须为整数！")
        except Exception as e:
            self.log_message(f"❌ 发送数据失败: {str(e)}")

    def enable_grid_buttons(self, enabled: bool):
        for btn in self.grid_A.cells.values():
            btn.setEnabled(enabled)
        for btn in self.grid_B.cells.values():
            btn.setEnabled(enabled)

    def on_tube_cell_clicked(self, label: str):
        if not self.manual_pick_active:
            QMessageBox.warning(self, "警告", "请先点击“手动挑管”按钮激活此模式")
            return
        if not (self.controller.camera_thread and self.controller.camera_thread.isRunning()):
            QMessageBox.warning(self, "警告", "请先启动相机")
            return
        self.controller.select_tube(label)

    def on_place_cell_clicked(self, label: str):
        if not self.manual_pick_active:
            QMessageBox.warning(self, "警告", "请先点击“手动挑管”按钮激活此模式")
            return
        try:
            rows = int(self.square_rows_edit.text())
            cols = int(self.square_cols_edit.text())
            x = float(self.square_bottom_left_x_edit.text())
            y = float(self.square_bottom_left_y_edit.text())
            d = float(self.square_distance_edit.text())
            ox = float(self.square_offset_x_edit.text())
            oy = float(self.square_offset_y_edit.text())
            self.controller.set_square_params(rows, cols, x, y, d, ox, oy)
            self.controller.select_place_position(label)
            if self.controller.selected_place_label == label:
                self.grid_B.set_cell_filled(label, True)
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"请检查网格参数: {str(e)}")

    def update_tube_detection_status(self, label: str, is_detected: bool):
        self.grid_A.set_cell_filled(label, is_detected)

    def _apply_params_to_controller(self):
        self.controller.set_ip(self.ip_edit.text())
        self.controller.set_port(int(self.port_edit.text()))
        self.controller.set_offset_x(float(self.offset_x_edit.text()))
        self.controller.set_offset_y(float(self.offset_y_edit.text()))
        self.controller.set_use_roi(self.roi_checkbox.isChecked())
        self.controller.set_roi_params(
            int(self.roi_x_edit.text()), int(self.roi_y_edit.text()),
            int(self.roi_width_edit.text()), int(self.roi_height_edit.text())
        )
        self.controller.set_circle_params(
            float(self.dp_edit.text()), int(self.min_dist_edit.text()),
            int(self.param1_edit.text()), int(self.param2_edit.text()),
            int(self.min_radius_edit.text()), int(self.max_radius_edit.text())
        )
        self.controller.set_template_params(
            self.template_path_edit.text().strip(),
            float(self.match_threshold_edit.text())
        )
        self.controller.set_rotation_angle(float(self.rotation_edit.text()))
        matrix_vals = [float(edit.text()) for edit in self.matrix_edit]
        self.controller.set_matrix_values(matrix_vals)
        self.controller.set_fixed_pose(
            float(self.z_edit.text()), float(self.rx_edit.text()),
            float(self.ry_edit.text()), float(self.rz_edit.text())
        )

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage, QPixmap
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_log(self, message):
        self.log_text.append(f"{time.strftime('%H:%M:%S')} - {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def on_tube_selected(self, coords):
        pass

    def update_robot_state_ui(self, state_dict):
        self.auto_exit_label.setText(str(state_dict.get("auto_exit", "N/A")))
        self.ready_label.setText(str(state_dict.get("ready", "N/A")))
        self.paused_label.setText(str(state_dict.get("paused", "N/A")))
        self.running_label.setText(str(state_dict.get("running", "N/A")))
        self.alarm_label.setText(str(state_dict.get("alarm", "N/A")))
        self.data_valid_label.setText(str(state_dict.get("data_valid", "N/A")))
        self.digital_in_label.setText(str(state_dict.get("digital_in", "N/A")))
        self.digital_out_label.setText(str(state_dict.get("digital_out", "N/A")))

    def update_gripper_state_ui(self, state_31005):
        self.gripper_31005_label.setText(state_31005)

    def log_message(self, msg):
        self.controller.log_message.emit(msg)

    def on_box_type_changed(self, text):
        self.controller.on_box_type_changed(text)
        if text in self.controller.tube_config:
            p = self.controller.tube_config[text]
            self.dp_edit.setText(str(p["dp"]))
            self.min_radius_edit.setText(str(p["min_radius"]))
            self.max_radius_edit.setText(str(p["max_radius"]))
            self.height_edit.setText(str(p["height_value"]))
            self.offset_x_edit.setText(str(p["offset_x"]))
            self.offset_y_edit.setText(str(p["offset_y"]))
            self.min_dist_edit.setText(str(p.get("minDist", 50)))
            self.param1_edit.setText(str(p.get("param1", 50)))
            self.param2_edit.setText(str(p.get("param2", 23)))
            self.template_path_edit.setText(p.get("template_path", "template.jpg"))
            self.match_threshold_edit.setText(str(p.get("match_threshold", 0.4)))

    def on_square_type_changed(self, text):
        self.controller.on_square_type_changed(text)
        if text in self.controller.square_grid_config:
            p = self.controller.square_grid_config[text]
            self.square_rows_edit.setText(str(p["rows"]))
            self.square_cols_edit.setText(str(p["cols"]))
            self.square_bottom_left_x_edit.setText(str(p["bottom_left_x"]))
            self.square_bottom_left_y_edit.setText(str(p["bottom_left_y"]))
            self.square_distance_edit.setText(str(p["distance"]))
            self.square_offset_x_edit.setText(str(p.get("offset_x", 0.0)))
            self.square_offset_y_edit.setText(str(p.get("offset_y", 0.0)))

    def read_robot_state(self):
        self.controller.read_robot_state_once()

    def closeEvent(self, event):
        self.controller.shutdown()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NewMainWindow()
    window.show()
    sys.exit(app.exec())