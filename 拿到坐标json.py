import struct
import cv2
import numpy as np
import time
import json
from pymodbus.client import ModbusTcpClient
from MvCameraControl_class import *
from CameraParams_header import *
from ctypes import *
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox,
                               QGroupBox, QFormLayout, QMessageBox, QInputDialog, QTextEdit,
                               QComboBox, QFileDialog)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap

# ========== 新增：夹爪状态解析函数 ==========
def parse_gripper_31009(reg_value):
    if reg_value == 0:
        return "无动作"
    BOX_CLAMP = 1  # 盒夹紧
    BOX_RELEASE = 2  # 盒松开
    TUBE_CLAMP = 4  # 管夹紧
    TUBE_RELEASE = 8  # 管松开
    box_conflict = (reg_value & BOX_CLAMP) and (reg_value & BOX_RELEASE)
    tube_conflict = (reg_value & TUBE_CLAMP) and (reg_value & TUBE_RELEASE)
    if box_conflict and tube_conflict:
        return "❌ 错误：盒夹爪与管夹爪均同时夹紧与松开"
    elif box_conflict:
        return "❌ 错误：盒夹爪同时夹紧与松开"
    elif tube_conflict:
        return "❌ 错误：管夹爪同时夹紧与松开"
    parts = []
    if reg_value & BOX_CLAMP:
        parts.append("盒夹爪夹紧")
    if reg_value & BOX_RELEASE:
        parts.append("盒夹爪松开")
    if reg_value & TUBE_CLAMP:
        parts.append("管夹爪夹紧")
    if reg_value & TUBE_RELEASE:
        parts.append("管夹爪松开")
    return " + ".join(parts) if parts else f"未知状态({reg_value})"

# ========== 新增：机器人状态轮询线程 ==========
class RobotStateThread(QThread):
    update_robot_state_signal = Signal(dict)
    update_gripper_state_signal = Signal(str, str)
    log_signal = Signal(str)

    def __init__(self, ip, port, interval=1.0):
        super().__init__()
        self.ip = ip
        self.port = port
        self.interval = interval
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            try:
                client = ModbusTcpClient(self.ip, port=self.port)
                if not client.connect():
                    self.log_signal.emit("❌ 机器人状态线程：连接失败")
                    time.sleep(self.interval)
                    continue
                discrete_result = client.read_discrete_inputs(address=0, count=5)
                input_result = client.read_input_registers(address=999, count=13)

                robot_state = {}
                gripper_31005_state = "未读取"
                gripper_31009_state = "未读取"

                if not discrete_result.isError():
                    bits = discrete_result.bits[:5]
                    robot_state.update({
                        "auto_exit": bits[0],
                        "ready": bits[1],
                        "paused": bits[2],
                        "running": bits[3],
                        "alarm": bits[4]
                    })
                else:
                    self.log_signal.emit(f"❌ 离散输入读取失败: {discrete_result}")

                if not input_result.isError():
                    regs = input_result.registers
                    robot_state.update({
                        "data_valid": regs[0],
                        "digital_in": (regs[5] << 48) | (regs[6] << 32) | (regs[7] << 16) | regs[8],
                        "digital_out": (regs[9] << 48) | (regs[10] << 32) | (regs[11] << 16) | regs[12]
                    })
                    reg_31005 = regs[5]
                    reg_31009 = regs[9]

                    state_31005_map = {
                        17: "盒夹爪夹紧到位",
                        18: "盒夹爪松开到位",
                        19: "管夹爪夹紧到位",
                        20: "管夹爪松开到位"
                    }
                    gripper_31005_state = state_31005_map.get(reg_31005, f"未知状态({reg_31005})")
                    gripper_31009_state = parse_gripper_31009(reg_31009)

                self.update_robot_state_signal.emit(robot_state)
                self.update_gripper_state_signal.emit(gripper_31005_state, gripper_31009_state)
                client.close()
            except Exception as e:
                self.log_signal.emit(f"❌ 机器人状态线程异常: {str(e)}")
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        self.wait()

# ----------------------- 模板匹配相关函数（内嵌） -----------------------
def template_match_on_image(img, fixed_template_gray, threshold):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    result = cv2.matchTemplate(img_gray, fixed_template_gray, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    detections = []
    for pt in zip(*locations[::-1]):
        score = result[pt[1], pt[0]]
        detections.append((pt[0], pt[1], score))
    detections.sort(key=lambda x: x[2], reverse=True)
    final_detections = []
    w, h = fixed_template_gray.shape[1], fixed_template_gray.shape[0]
    for x, y, score in detections:
        overlap = False
        for fx, fy, _ in final_detections:
            if np.sqrt((x - fx) ** 2 + (y - fy) ** 2) < max(w, h) / 2:
                overlap = True
                break
        if not overlap:
            final_detections.append((x, y, score))
    return final_detections

def is_circle_in_template(circle_x, circle_y, tx, ty, tw, th):
    return tx <= circle_x <= tx + tw and ty <= circle_y <= ty + th

class CameraThread(QThread):
    update_signal = Signal(np.ndarray)
    coords_signal = Signal(tuple)
    log_signal = Signal(str)
    detection_complete_signal = Signal(dict)  # 新增：检测完成信号

    def __init__(self, rotation_angle, H_vision, offset_x, offset_y, use_roi, roi_params,
                 circle_params, template_path, match_threshold):
        super().__init__()
        self.running = False
        self.rotation_angle = rotation_angle
        self.H_vision = H_vision
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.USE_ROI = use_roi
        self.roi_x, self.roi_y, self.roi_width, self.roi_height = roi_params
        self.dp, self.min_dist, self.param1, self.param2, self.min_radius, self.max_radius = circle_params
        self.template_path = template_path
        self.match_threshold = match_threshold
        self.fixed_template = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        if self.fixed_template is None:
            raise FileNotFoundError(f"无法加载模板图像: {self.template_path}")
        self.TEMPLATE_W, self.TEMPLATE_H = self.fixed_template.shape[1], self.fixed_template.shape[0]
        self.detected_circles = {}
        self.selected_label = None
        self.max_circle_count = 0
        self.orig_to_corr_matrix = None  # 用于坐标转换

    def run(self):
        self.running = True
        ret = MvCamera.MV_CC_Initialize()
        if ret != 0:
            self.log_signal.emit(f"初始化SDK失败，错误码: {ret}")
            return

        deviceList = MV_CC_DEVICE_INFO_LIST()
        n_layer_type = MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(n_layer_type, deviceList)
        if ret != 0 or deviceList.nDeviceNum == 0:
            self.log_signal.emit("枚举设备失败或未找到设备")
            return

        # --- 修改开始：查找指定IP的相机 ---
        target_ip = "192.168.5.100" # 硬编码指定相机IP
        selected_device_index = -1
        camera = MvCamera() # 在循环前创建相机对象，避免重复创建

        for i in range(deviceList.nDeviceNum):
            stDeviceList = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                try:
                    # 获取IP地址
                    ip_parts = []
                    ip_addr = stDeviceList.SpecialInfo.stGigEInfo.nCurrentIp
                    for j in range(4):
                        ip_part = (ip_addr >> (8 * j)) & 0xFF
                        ip_parts.append(str(ip_part))
                    strIP = ".".join(reversed(ip_parts))
                    self.log_signal.emit(f"发现设备 {i}，IP地址: {strIP}")

                    if strIP == target_ip:
                        selected_device_index = i
                        self.log_signal.emit(f"✅ 找到目标相机 {target_ip}！")
                        # 在这里创建句柄，因为后续需要获取参数
                        ret = camera.MV_CC_CreateHandle(stDeviceList)
                        if ret != 0:
                            self.log_signal.emit(f"为设备 {i} 创建句柄失败，错误码: {ret}")
                            selected_device_index = -1 # 标记失败
                            continue # 尝试下一个设备
                        break
                except Exception as e:
                    self.log_signal.emit(f"获取设备 {i} 信息失败: {e}")

        if selected_device_index == -1:
            self.log_signal.emit(f"❌ 未找到IP为 {target_ip} 的相机")
            return # 直接退出线程
        # --- 修改结束 ---

        # stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
        # camera = MvCamera()
        # ret = camera.MV_CC_CreateHandle(stDeviceList)
        # if ret != 0:
        #     self.log_signal.emit(f"创建句柄失败，错误码: {ret}")
        #     return

        ret = camera.MV_CC_OpenDevice()
        if ret != 0:
            self.log_signal.emit(f"打开设备失败，错误码: {ret}")
            return

        stParam = MVCC_INTVALUE()
        ret = camera.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            self.log_signal.emit(f"获取PayloadSize失败，错误码: {ret}")
            return
        payload_size = stParam.nCurValue

        exposure_time = 15000
        camera.MV_CC_SetFloatValue("ExposureTime", exposure_time)
        ret = camera.MV_CC_StartGrabbing()
        if ret != 0:
            self.log_signal.emit(f"开始抓图失败，错误码: {ret}")
            return

        data_buf = (c_ubyte * payload_size)()
        stFrameInfo = MV_FRAME_OUT_INFO_EX()

        angle_calibrated = False
        transform_matrix = None
        inv_transform_matrix = None

        try:
            self.log_signal.emit("正在检测试管（模板匹配+圆形检测），请稍候...")
            self.log_signal.emit("提示: 按's'键选择试管，按'q'键退出相机模式")
            while self.running:
                data_buf = (c_ubyte * payload_size)()
                ret = camera.MV_CC_GetOneFrameTimeout(
                    byref(data_buf),
                    payload_size,
                    stFrameInfo,
                    1000
                )

                if ret == 0:
                    frame = np.frombuffer(data_buf, dtype=np.uint8)
                    actual_width = stFrameInfo.nWidth
                    actual_height = stFrameInfo.nHeight
                    expected_mono = actual_width * actual_height
                    if len(frame) != expected_mono:
                        continue

                    frame = frame.reshape((actual_height, actual_width))

                    if not angle_calibrated:
                        _, transform_matrix = self.rotate_image(frame, self.rotation_angle)
                        inv_transform_matrix = cv2.invertAffineTransform(transform_matrix)
                        self.orig_to_corr_matrix = self.get_original_to_corrected_matrix(inv_transform_matrix)
                        angle_calibrated = True

                    corrected_frame, _ = self.rotate_image(frame, self.rotation_angle)
                    frame_result, _, _ = self.opencv_action(
                        corrected_frame, transform_matrix, inv_transform_matrix)
                    self.update_signal.emit(frame_result)

                    # 发出检测结果
                    world_dict = {}
                    for label, (orig_x, orig_y) in self.detected_circles.items():
                        world_x, world_y = self.original_to_world(
                            orig_x, orig_y, self.orig_to_corr_matrix, self.H_vision)
                        world_dict[label] = {
                            "original_pixel": [orig_x, orig_y],
                            "world_coords_mm": [round(world_x, 4), round(world_y, 4)]
                        }
                    self.detection_complete_signal.emit(world_dict)

                    if self.selected_label is not None:
                        if self.selected_label in self.detected_circles:
                            orig_x, orig_y = self.detected_circles[self.selected_label]
                            world_x, world_y = self.original_to_world(
                                orig_x, orig_y, self.orig_to_corr_matrix, self.H_vision)
                            self.log_signal.emit(
                                f"试管 {self.selected_label} 的世界坐标为: ({world_x:.4f}, {world_y:.4f}) mm")
                            self.coords_signal.emit((world_x, world_y))
                            self.selected_label = None
                        else:
                            self.log_signal.emit(f"未找到标签为 {self.selected_label} 的试管")
                            self.selected_label = None

                else:
                    self.log_signal.emit(f"获取图像失败，错误码: {ret}")
                    time.sleep(0.5)

        finally:
            camera.MV_CC_StopGrabbing()
            camera.MV_CC_CloseDevice()
            camera.MV_CC_DestroyHandle()
            self.running = False

    def stop(self):
        self.running = False
        self.wait(1000)

    def select_tube(self, label):
        self.selected_label = label

    def rotate_image(self, img, angle):
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated, M

    def get_original_coords(self, corrected_x, corrected_y, M_inv):
        homogeneous = np.array([corrected_x, corrected_y, 1], dtype=np.float32)
        original = M_inv @ homogeneous
        return int(round(original[0])), int(round(original[1]))

    def get_original_to_corrected_matrix(self, inv_transform_matrix):
        inv_affine_homo = np.vstack([inv_transform_matrix, [0, 0, 1]])
        M_homo = np.linalg.inv(inv_affine_homo)
        return M_homo[:2, :]

    def original_to_corrected(self, u_orig, v_orig, M):
        orig_homo = np.array([u_orig, v_orig, 1], dtype=np.float32)
        corr_homo = M @ orig_homo
        return corr_homo[0], corr_homo[1]

    def corrected_pixel_to_vision_world(self, u_corr, v_corr, H_vision):
        pixel_homo = np.array([[u_corr, v_corr, 1]], dtype=np.float32).T
        world_homo = H_vision @ pixel_homo
        vision_x = world_homo[0, 0] / world_homo[2, 0]
        vision_y = world_homo[1, 0] / world_homo[2, 0]
        corrected_x = vision_x + self.offset_x
        corrected_y = vision_y + self.offset_y
        return corrected_x, corrected_y

    def original_to_world(self, u_orig, v_orig, M, H_vision):
        u_corr, v_corr = self.original_to_corrected(u_orig, v_orig, M)
        return self.corrected_pixel_to_vision_world(u_corr, v_corr, H_vision)

    def opencv_action(self, img, M, M_inv):
        self.detected_circles.clear()
        original_img = img.copy()

        roi_x = self.roi_x
        roi_y = self.roi_y
        roi_width = self.roi_width
        roi_height = self.roi_height

        img_height, img_width = img.shape[:2]
        roi_x = max(0, min(roi_x, img_width - 1))
        roi_y = max(0, min(roi_y, img_height - 1))
        roi_width = max(1, min(roi_width, img_width - roi_x))
        roi_height = max(1, min(roi_height, img_height - roi_y))

        if self.USE_ROI:
            roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
        else:
            roi = img.copy()
            roi_x, roi_y = 0, 0
            roi_width, roi_height = img_width, img_height

        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)

        # ========== 修复：HoughCircles 参数名和调用方式 ==========
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            self.dp,           # dp (位置参数)
            self.min_dist,     # minDist (位置参数)
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,  # 注意：不是 min_radius
            maxRadius=self.max_radius   # 注意：不是 max_radius
        )

        if len(roi.shape) == 2:
            output_roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        else:
            output_roi = roi.copy()

        template_detections = []
        if self.USE_ROI:
            roi_for_match = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR) if len(roi.shape) == 2 else roi.copy()
            local_detections = template_match_on_image(roi_for_match, self.fixed_template, self.match_threshold)
            for x, y, score in local_detections:
                template_detections.append((x + roi_x, y + roi_y, score))
        else:
            img_for_match = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
            template_detections = template_match_on_image(img_for_match, self.fixed_template, self.match_threshold)

        valid_circles = []
        matched_template_indices = set()
        if circles is not None and template_detections:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                global_x = x + roi_x if self.USE_ROI else x
                global_y = y + roi_y if self.USE_ROI else y
                for idx, (tx, ty, _) in enumerate(template_detections):
                    if is_circle_in_template(global_x, global_y, tx, ty, self.TEMPLATE_W, self.TEMPLATE_H):
                        valid_circles.append((global_x, global_y, r))
                        matched_template_indices.add(idx)
                        break

        current_count = len(valid_circles)
        if current_count > self.max_circle_count:
            self.max_circle_count = current_count

        if len(original_img.shape) == 2:
            output = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            output = original_img.copy()

        if self.USE_ROI:
            output[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = output_roi
            cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
        else:
            output = output_roi

        full_height, full_width = output.shape[:2]
        cv2.rectangle(output, (full_width - 260, 20), (full_width - 20, 130), (0, 0, 0), -1)
        cv2.putText(output, f"Valid Count: {current_count}", (full_width - 250, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(output, f"Max Valid: {self.max_circle_count}", (full_width - 250, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(output, f"Matched Templates: {len(matched_template_indices)}", (full_width - 250, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(output, f"ROI: {'ON' if self.USE_ROI else 'OFF'}",
                    (full_width - 130, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.USE_ROI else (0, 0, 255), 2)

        if valid_circles:
            centers = [(x, y, r) for x, y, r in valid_circles]

            # 行从下到上：先按 Y 降序排序
            centers.sort(key=lambda p: -p[1])

            rows = []
            if centers:
                current_row = [centers[0]]
                row_y = centers[0][1]
                for i in range(1, len(centers)):
                    if abs(centers[i][1] - row_y) < 50:
                        current_row.append(centers[i])
                    else:
                        rows.append(current_row)
                        current_row = [centers[i]]
                        row_y = centers[i][1]
                if current_row:
                    rows.append(current_row)

            # 列从左到右：每行内按 X 升序排序（去掉 reverse=True）
            for row in rows:
                row.sort(key=lambda p: p[0])

            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for r_idx, row in enumerate(rows):
                row_number = r_idx + 1
                for c_idx, (x, y, r) in enumerate(row):
                    col_letter = letters[c_idx] if c_idx < len(letters) else f"_{c_idx}"
                    label = f"{row_number}{col_letter}"
                    local_x = x - roi_x if self.USE_ROI else x
                    local_y = y - roi_y if self.USE_ROI else y
                    if 0 <= local_x < output_roi.shape[1] and 0 <= local_y < output_roi.shape[0]:
                        cv2.putText(output_roi, label, (local_x - 15, local_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        cv2.circle(output_roi, (local_x, local_y), 2, (0, 0, 255), 3)
                        cv2.circle(output_roi, (local_x, local_y), r, (0, 0, 255), 2)
                    orig_x, orig_y = self.get_original_coords(x, y, M_inv)
                    self.detected_circles[label] = (orig_x, orig_y)

            if self.USE_ROI:
                output[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width] = output_roi

        for idx, (x, y, score) in enumerate(template_detections):
            if idx in matched_template_indices:
                cv2.rectangle(output, (x, y), (x + self.TEMPLATE_W, y + self.TEMPLATE_H), (0, 255, 0), 2)
                cv2.putText(output, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return output, gray, blurred

class SquareGridCalculator:
    @staticmethod
    def calculate_labeled_square_centers(x, y, d, rows, cols, offset_x=0, offset_y=0):
        coordinates = {}
        col_labels = [chr(ord('A') + i) for i in range(cols)]
        for row_idx in range(rows):
            row_label = row_idx + 1
            for col_idx, col_label in enumerate(col_labels):
                center_x = x + col_idx * d + offset_x
                center_y = y + row_idx * d + offset_y
                cell_label = f"{row_label}{col_label}"
                coordinates[cell_label] = (center_x, center_y)
        return coordinates

    @staticmethod
    def print_labeled_coordinates(coordinates, rows, cols):
        row_labels = sorted(set(int(label[:-1]) for label in coordinates.keys()))
        result = []
        for row in row_labels[:rows]:
            row_str = f"行 {row}: "
            col_labels = [chr(ord('A') + i) for i in range(cols)]
            for col in col_labels:
                cell = f"{row}{col}"
                if cell in coordinates:
                    x, y = coordinates[cell]
                    row_str += f"{cell}:({x:.1f}, {y:.1f})  "
            result.append(row_str)
        return "\n".join(result)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("试管检测控制系统（模板匹配 + 机器人状态监控）")
        self.setGeometry(100, 100, 1200, 800)
        self.tube_config = self.load_tube_config()
        self.square_grid_config = self.load_square_grid_config()
        self.init_params()
        self.create_ui()
        self.camera_thread = None
        self.tube_coords = None
        self.place_coords = None
        self.latest_detection = {}  # 缓存最新检测结果
        # ========== 新增：机器人状态线程 ==========
        self.robot_state_thread = None
        self.start_robot_state_thread()

    def load_tube_config(self):
        try:
            with open("tube_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            QMessageBox.warning(self, "配置文件缺失", "未找到tube_config.json，请确保文件存在")
            return {}
        except json.JSONDecodeError:
            QMessageBox.warning(self, "配置文件错误", "tube_config.json格式错误")
            return {}

    def load_square_grid_config(self):
        try:
            with open("square_grid_config.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            QMessageBox.warning(self, "配置文件缺失", "未找到square_grid_config.json，请确保文件存在")
            return {}
        except json.JSONDecodeError:
            QMessageBox.warning(self, "配置文件错误", "square_grid_config.json格式错误")
            return {}

    def init_params(self):
        # 修改：默认IP为机械手IP
        self.ip = "192.168.5.1"
        self.port = 502
        self.offset_x = 0
        self.offset_y = 0
        self.use_roi = True
        self.roi_x = 772
        self.roi_y = 480
        self.roi_width = 702
        self.roi_height = 1030
        self.dp = 1.3
        self.min_dist = 55
        self.param1 = 45
        self.param2 = 20
        self.min_radius = 37
        self.max_radius = 40
        self.rotation_angle = 187.0
        self.H_vision = np.array([
            [-1.08752846e-01, 9.99295051e-03, 3.80777969e+02],
            [1.61922938e-02, 9.44374635e-02, -4.73632486e+02],
            [-2.98764047e-05, 1.49547878e-05, 1.00000000e+00]
        ])
        self.square_rows = 5
        self.square_cols = 5
        self.square_bottom_left_x = 100.0
        self.square_bottom_left_y = 100.0
        self.square_distance = 30.0
        self.square_offset_x = 0.0
        self.square_offset_y = 1.0
        self.z = 470.2899
        self.rx = 3.3021
        self.ry = -0.2048
        self.rz = 134.9677
        self.task_number = 0
        self.height_value = 0
        self.selected_box_type = "8*6试管盒"
        self.selected_square_type = "8*6空盒"
        self.template_path = "template_8x6.jpg"
        self.match_threshold = 0.3
        # 机器人状态初始值
        self.robot_state = {
            "auto_exit": False,
            "ready": False,
            "paused": False,
            "running": False,
            "alarm": False,
            "data_valid": 0,
            "digital_in": 0,
            "digital_out": 0
        }
        self.gripper_31005_state = "未读取"
        self.gripper_31009_state = "未读取"

    def create_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.create_param_panel(main_layout)
        self.create_display_panel(main_layout)
        self.on_box_type_changed(self.selected_box_type)
        self.on_square_type_changed(self.selected_square_type)

    def create_param_panel(self, main_layout):
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)
        param_widget.setFixedWidth(350)

        box_type_group = QGroupBox("试管盒类型")
        box_type_layout = QHBoxLayout(box_type_group)
        self.box_type_combo = QComboBox()
        if self.tube_config:
            self.box_type_combo.addItems(list(self.tube_config.keys()))
        else:
            self.box_type_combo.addItems(["8*6试管盒", "12*8试管盒", "6*4试管盒", "4*4试管盒"])
        self.box_type_combo.setCurrentText(self.selected_box_type)
        self.box_type_combo.currentTextChanged.connect(self.on_box_type_changed)
        box_type_layout.addWidget(self.box_type_combo)
        param_layout.addWidget(box_type_group)

        square_type_group = QGroupBox("空盒类型")
        square_type_layout = QHBoxLayout(square_type_group)
        self.square_type_combo = QComboBox()
        if self.square_grid_config:
            self.square_type_combo.addItems(list(self.square_grid_config.keys()))
        else:
            self.square_type_combo.addItems(["8*6空盒", "12*8空盒", "6*4空盒", "4*4空盒"])
        self.square_type_combo.setCurrentText(self.selected_square_type)
        self.square_type_combo.currentTextChanged.connect(self.on_square_type_changed)
        square_type_layout.addWidget(self.square_type_combo)
        param_layout.addWidget(square_type_group)

        tab_widget = QTabWidget()
        # === 原有标签页 ===
        conn_tab = QWidget()
        conn_layout = QFormLayout(conn_tab)
        self.ip_edit = QLineEdit(self.ip)
        self.port_edit = QLineEdit(str(self.port))
        conn_layout.addRow("IP地址:", self.ip_edit)
        conn_layout.addRow("端口:", self.port_edit)
        tab_widget.addTab(conn_tab, "连接参数")

        offset_tab = QWidget()
        offset_layout = QFormLayout(offset_tab)
        self.offset_x_edit = QLineEdit(str(self.offset_x))
        self.offset_y_edit = QLineEdit(str(self.offset_y))
        offset_layout.addRow("X偏移量:", self.offset_x_edit)
        offset_layout.addRow("Y偏移量:", self.offset_y_edit)
        tab_widget.addTab(offset_tab, "偏移参数")

        roi_tab = QWidget()
        roi_layout = QVBoxLayout(roi_tab)
        self.roi_checkbox = QCheckBox("启用ROI")
        self.roi_checkbox.setChecked(self.use_roi)
        roi_form = QFormLayout()
        self.roi_x_edit = QLineEdit(str(self.roi_x))
        self.roi_y_edit = QLineEdit(str(self.roi_y))
        self.roi_width_edit = QLineEdit(str(self.roi_width))
        self.roi_height_edit = QLineEdit(str(self.roi_height))
        roi_form.addRow("ROI X:", self.roi_x_edit)
        roi_form.addRow("ROI Y:", self.roi_y_edit)
        roi_form.addRow("宽度:", self.roi_width_edit)
        roi_form.addRow("高度:", self.roi_height_edit)
        roi_layout.addWidget(self.roi_checkbox)
        roi_layout.addLayout(roi_form)
        roi_layout.addStretch()
        tab_widget.addTab(roi_tab, "ROI设置")

        circle_tab = QWidget()
        circle_layout = QFormLayout(circle_tab)
        self.dp_edit = QLineEdit(str(self.dp))
        self.min_dist_edit = QLineEdit(str(self.min_dist))
        self.param1_edit = QLineEdit(str(self.param1))
        self.param2_edit = QLineEdit(str(self.param2))
        self.min_radius_edit = QLineEdit(str(self.min_radius))
        self.max_radius_edit = QLineEdit(str(self.max_radius))
        self.template_path_edit = QLineEdit(self.template_path)
        self.match_threshold_edit = QLineEdit(str(self.match_threshold))
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
        self.rotation_edit = QLineEdit(str(self.rotation_angle))
        self.matrix_edit = []
        for i in range(3):
            row = QHBoxLayout()
            for j in range(3):
                edit = QLineEdit(str(self.H_vision[i][j]))
                edit.setFixedWidth(80)
                self.matrix_edit.append(edit)
                row.addWidget(edit)
            correction_layout.addRow(f"矩阵行 {i + 1}:", row)
        correction_layout.addRow("旋转角度:", self.rotation_edit)
        tab_widget.addTab(correction_tab, "矫正参数")

        square_tab = QWidget()
        square_layout = QFormLayout(square_tab)
        self.square_rows_edit = QLineEdit(str(self.square_rows))
        self.square_cols_edit = QLineEdit(str(self.square_cols))
        self.square_bottom_left_x_edit = QLineEdit(str(self.square_bottom_left_x))
        self.square_bottom_left_y_edit = QLineEdit(str(self.square_bottom_left_y))
        self.square_distance_edit = QLineEdit(str(self.square_distance))
        self.square_offset_x_edit = QLineEdit(str(self.square_offset_x))
        self.square_offset_y_edit = QLineEdit(str(self.square_offset_y))
        self.show_grid_btn = QPushButton("显示网格坐标")
        self.show_grid_btn.clicked.connect(self.show_grid_coordinates)
        square_layout.addRow("网格行数:", self.square_rows_edit)
        square_layout.addRow("网格列数:", self.square_cols_edit)
        square_layout.addRow("左下角X坐标:", self.square_bottom_left_x_edit)
        square_layout.addRow("左下角Y坐标:", self.square_bottom_left_y_edit)
        square_layout.addRow("中心距离:", self.square_distance_edit)
        square_layout.addRow("X方向偏移:", self.square_offset_x_edit)
        square_layout.addRow("Y方向偏移:", self.square_offset_y_edit)
        square_layout.addRow(self.show_grid_btn)
        tab_widget.addTab(square_tab, "正方形网格")

        fixed_tab = QWidget()
        fixed_layout = QFormLayout(fixed_tab)
        self.z_edit = QLineEdit(str(self.z))
        self.rx_edit = QLineEdit(str(self.rx))
        self.ry_edit = QLineEdit(str(self.ry))
        self.rz_edit = QLineEdit(str(self.rz))
        fixed_layout.addRow("Z:", self.z_edit)
        fixed_layout.addRow("RX:", self.rx_edit)
        fixed_layout.addRow("RY:", self.ry_edit)
        fixed_layout.addRow("RZ:", self.rz_edit)
        tab_widget.addTab(fixed_tab, "固定参数")

        task_tab = QWidget()
        task_layout = QFormLayout(task_tab)
        self.task_edit = QLineEdit(str(self.task_number))
        self.height_edit = QLineEdit(str(self.height_value))
        task_layout.addRow("任务号:", self.task_edit)
        task_layout.addRow("高度参数:", self.height_edit)
        tab_widget.addTab(task_tab, "任务参数")

        # ========== 新增：机器人状态标签页 ==========
        robot_state_tab = QWidget()
        robot_state_layout = QFormLayout(robot_state_tab)
        self.auto_exit_label = QLabel(str(self.robot_state["auto_exit"]))
        self.ready_label = QLabel(str(self.robot_state["ready"]))
        self.paused_label = QLabel(str(self.robot_state["paused"]))
        self.running_label = QLabel(str(self.robot_state["running"]))
        self.alarm_label = QLabel(str(self.robot_state["alarm"]))
        self.data_valid_label = QLabel(str(self.robot_state["data_valid"]))
        self.digital_in_label = QLabel(str(self.robot_state["digital_in"]))
        self.digital_out_label = QLabel(str(self.robot_state["digital_out"]))
        state_labels = ["自动退出:", "已准备好:", "暂停中:", "运行状态:", "报警状态:", "数据有效性:", "数字输入:",
                        "数字输出:"]
        state_values = [self.auto_exit_label, self.ready_label, self.paused_label, self.running_label,
                        self.alarm_label, self.data_valid_label, self.digital_in_label, self.digital_out_label]
        for lbl_text, widget in zip(state_labels, state_values):
            robot_state_layout.addRow(QLabel(lbl_text), widget)
        self.refresh_robot_state_btn = QPushButton("刷新机器人状态")
        self.refresh_robot_state_btn.clicked.connect(self.read_robot_state)
        robot_state_layout.addRow(self.refresh_robot_state_btn)
        tab_widget.addTab(robot_state_tab, "机器人状态")

        # ========== 新增：夹爪状态标签页 ==========
        gripper_tab = QWidget()
        gripper_layout = QFormLayout(gripper_tab)
        self.gripper_31005_label = QLabel("未读取")
        self.gripper_31009_label = QLabel("未读取")
        gripper_layout.addRow(QLabel("31005（夹爪到位状态）:"), self.gripper_31005_label)
        gripper_layout.addRow(QLabel("31009（夹爪动作指令）:"), self.gripper_31009_label)
        self.refresh_gripper_btn = QPushButton("刷新夹爪状态")
        self.refresh_gripper_btn.clicked.connect(self.read_robot_state)
        gripper_layout.addRow(self.refresh_gripper_btn)
        tab_widget.addTab(gripper_tab, "夹爪状态")

        param_layout.addWidget(tab_widget)

        btn_layout = QVBoxLayout()
        self.start_camera_btn = QPushButton("启动相机")
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.select_tube_btn = QPushButton("选择抓取试管")
        self.select_tube_btn.clicked.connect(self.select_tube)
        self.select_place_btn = QPushButton("选择放置位置")
        self.select_place_btn.clicked.connect(self.select_place_position)
        self.send_btn = QPushButton("发送数据")
        self.send_btn.clicked.connect(self.send_data)
        self.save_coords_btn = QPushButton("保存检测坐标")
        self.save_coords_btn.clicked.connect(self.save_detected_coords)
        self.save_coords_btn.setEnabled(False)
        self.select_tube_btn.setEnabled(False)
        self.select_place_btn.setEnabled(False)
        btn_layout.addWidget(self.start_camera_btn)
        btn_layout.addWidget(self.select_tube_btn)
        btn_layout.addWidget(self.select_place_btn)
        btn_layout.addWidget(self.send_btn)
        btn_layout.addWidget(self.save_coords_btn)
        btn_layout.addStretch()
        param_layout.addLayout(btn_layout)

        main_layout.addWidget(param_widget)

    # ========== 新增：保存检测坐标 ==========
    def save_detected_coords(self):
        if not self.latest_detection:
            QMessageBox.warning(self, "警告", "暂无检测结果可保存")
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "保存检测坐标", "detected_tubes.json", "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(self.latest_detection, f, indent=2, ensure_ascii=False)
                self.update_log(f"✅ 检测坐标已保存至: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存失败: {str(e)}")

    # ========== 新增：机器人状态读取方法 ==========
    def read_robot_state(self):
        try:
            ip = self.ip_edit.text()
            port = int(self.port_edit.text())
            self.update_log(f"正在读取机器人状态: {ip}:{port}")
            client = ModbusTcpClient(ip, port=port)
            if not client.connect():
                self.update_log("❌ 连接机器人失败")
                QMessageBox.critical(self, "错误", "连接机器人失败")
                return

            discrete_result = client.read_discrete_inputs(address=0, count=5)
            input_result = client.read_input_registers(address=999, count=13)

            if not discrete_result.isError():
                bits = discrete_result.bits[:5]
                self.robot_state.update({
                    "auto_exit": bits[0],
                    "ready": bits[1],
                    "paused": bits[2],
                    "running": bits[3],
                    "alarm": bits[4]
                })
                self.auto_exit_label.setText(str(bits[0]))
                self.ready_label.setText(str(bits[1]))
                self.paused_label.setText(str(bits[2]))
                self.running_label.setText(str(bits[3]))
                self.alarm_label.setText(str(bits[4]))
            else:
                self.update_log(f"❌ 离散输入读取失败: {discrete_result}")

            if not input_result.isError():
                regs = input_result.registers
                self.robot_state.update({
                    "data_valid": regs[0],
                    "digital_in": (regs[5] << 48) | (regs[6] << 32) | (regs[7] << 16) | regs[8],
                    "digital_out": (regs[9] << 48) | (regs[10] << 32) | (regs[11] << 16) | regs[12]
                })
                reg_31005 = regs[5]
                reg_31009 = regs[9]

                state_31005_map = {
                    17: "盒夹爪夹紧到位",
                    18: "盒夹爪松开到位",
                    19: "管夹爪夹紧到位",
                    20: "管夹爪松开到位"
                }
                gripper_31005_state = state_31005_map.get(reg_31005, f"未知状态({reg_31005})")
                gripper_31009_state = parse_gripper_31009(reg_31009)

                self.data_valid_label.setText(str(regs[0]))
                self.digital_in_label.setText(str(self.robot_state["digital_in"]))
                self.digital_out_label.setText(str(self.robot_state["digital_out"]))
                self.gripper_31005_label.setText(gripper_31005_state)
                self.gripper_31009_label.setText(gripper_31009_state)
                self.update_log(f"✅ 机器人状态更新成功 | 夹爪动作: {gripper_31009_state}")

            client.close()
        except Exception as e:
            self.update_log(f"❌ 读取机器人状态错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"读取失败: {str(e)}")

    def start_robot_state_thread(self):
        ip = self.ip_edit.text()
        port = int(self.port_edit.text())
        self.robot_state_thread = RobotStateThread(ip, port, interval=1.0)
        self.robot_state_thread.update_robot_state_signal.connect(self.update_robot_state_ui)
        self.robot_state_thread.update_gripper_state_signal.connect(self.update_gripper_state_ui)
        self.robot_state_thread.log_signal.connect(self.update_log)
        self.robot_state_thread.start()

    def update_robot_state_ui(self, state_dict):
        self.robot_state.update(state_dict)
        self.auto_exit_label.setText(str(state_dict["auto_exit"]))
        self.ready_label.setText(str(state_dict["ready"]))
        self.paused_label.setText(str(state_dict["paused"]))
        self.running_label.setText(str(state_dict["running"]))
        self.alarm_label.setText(str(state_dict["alarm"]))
        self.data_valid_label.setText(str(state_dict["data_valid"]))
        self.digital_in_label.setText(str(state_dict["digital_in"]))
        self.digital_out_label.setText(str(state_dict["digital_out"]))

    def update_gripper_state_ui(self, state_31005, state_31009):
        self.gripper_31005_label.setText(state_31005)
        self.gripper_31009_label.setText(state_31009)

    # ========== 原有功能（优化相机启停逻辑）==========
    def show_grid_coordinates(self):
        try:
            rows = int(self.square_rows_edit.text())
            cols = int(self.square_cols_edit.text())
            x = float(self.square_bottom_left_x_edit.text())
            y = float(self.square_bottom_left_y_edit.text())
            distance = float(self.square_distance_edit.text())
            offset_x = float(self.square_offset_x_edit.text())
            offset_y = float(self.square_offset_y_edit.text())
            if cols > 26:
                QMessageBox.warning(self, "输入错误", "列数不能超过26")
                return

            coords = SquareGridCalculator.calculate_labeled_square_centers(
                x, y, distance, rows, cols, offset_x, offset_y)
            grid_info = SquareGridCalculator.print_labeled_coordinates(coords, rows, cols)
            self.update_log(f"\n{rows}×{cols}网格坐标（行: 1-{rows}, 列: A-{chr(ord('A') + cols - 1)}）")
            self.update_log(f"左下角坐标: ({x}, {y}), 相邻中心距离: {distance}")
            self.update_log(f"整体偏移量: X={offset_x}, Y={offset_y}")
            self.update_log("-" * 60)
            self.update_log(grid_info)
            self.update_log("-" * 60)
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"请检查网格参数: {str(e)}")

    def on_box_type_changed(self, text):
        self.selected_box_type = text
        if text in self.tube_config:
            params = self.tube_config[text]
            self.dp_edit.setText(str(params["dp"]))
            self.min_radius_edit.setText(str(params["min_radius"]))
            self.max_radius_edit.setText(str(params["max_radius"]))
            self.height_edit.setText(str(params["height_value"]))
            self.offset_x_edit.setText(str(params["offset_x"]))
            self.offset_y_edit.setText(str(params["offset_y"]))
            self.min_dist_edit.setText(str(params.get("minDist", 50)))
            self.param1_edit.setText(str(params.get("param1", 50)))
            self.param2_edit.setText(str(params.get("param2", 23)))
            self.template_path_edit.setText(params.get("template_path", "template.jpg"))
            self.match_threshold_edit.setText(str(params.get("match_threshold", 0.38)))
            self.update_log(f"已加载{text}的参数配置")

    def on_square_type_changed(self, text):
        self.selected_square_type = text
        if text in self.square_grid_config:
            params = self.square_grid_config[text]
            self.square_rows_edit.setText(str(params["rows"]))
            self.square_cols_edit.setText(str(params["cols"]))
            self.square_bottom_left_x_edit.setText(str(params["bottom_left_x"]))
            self.square_bottom_left_y_edit.setText(str(params["bottom_left_y"]))
            self.square_distance_edit.setText(str(params["distance"]))
            self.square_offset_x_edit.setText(str(params.get("offset_x", 0.0)))
            self.square_offset_y_edit.setText(str(params.get("offset_y", 0.0)))
            self.update_log(f"已加载空盒{text}的参数配置（包含偏移量）")

    def create_display_panel(self, main_layout):
        display_widget = QWidget()
        display_layout = QVBoxLayout(display_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        display_layout.addWidget(self.image_label)

        coords_group = QGroupBox("坐标信息")
        coords_layout = QFormLayout(coords_group)
        self.grab_coords_label = QLabel("未选择")
        self.place_coords_label = QLabel("未选择")
        coords_layout.addRow("抓取点坐标:", self.grab_coords_label)
        coords_layout.addRow("放置点坐标:", self.place_coords_label)
        display_layout.addWidget(coords_group)

        log_group = QGroupBox("日志信息")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        display_layout.addWidget(log_group)

        main_layout.addWidget(display_widget, 1)

    # ========== 优化：相机启停逻辑（核心修改部分） ==========
    def start_camera(self):
        try:
            # 1. 若相机已运行：停止线程并重置状态
            if self.camera_thread and self.camera_thread.isRunning():
                self.camera_thread.stop()
                self.camera_thread = None
                self.start_camera_btn.setText("启动相机")
                self.select_tube_btn.setEnabled(False)
                self.select_place_btn.setEnabled(False)
                self.save_coords_btn.setEnabled(False)
                self.update_log("相机已成功关闭")
                return
            # 2. 启动相机
            rotation_angle = float(self.rotation_edit.text())
            offset_x = float(self.offset_x_edit.text())
            offset_y = float(self.offset_y_edit.text())
            use_roi = self.roi_checkbox.isChecked()
            roi_x = int(self.roi_x_edit.text())
            roi_y = int(self.roi_y_edit.text())
            roi_width = int(self.roi_width_edit.text())
            roi_height = int(self.roi_height_edit.text())
            dp = float(self.dp_edit.text())
            min_dist = int(self.min_dist_edit.text())
            param1 = int(self.param1_edit.text())
            param2 = int(self.param2_edit.text())
            min_radius = int(self.min_radius_edit.text())
            max_radius = int(self.max_radius_edit.text())
            template_path = self.template_path_edit.text().strip()
            match_threshold = float(self.match_threshold_edit.text())

            H_vision = []
            for i in range(3):
                row = []
                for j in range(3):
                    row.append(float(self.matrix_edit[i * 3 + j].text()))
                H_vision.append(row)
            H_vision = np.array(H_vision)

            self.camera_thread = CameraThread(
                rotation_angle, H_vision, offset_x, offset_y,
                use_roi, (roi_x, roi_y, roi_width, roi_height),
                (dp, min_dist, param1, param2, min_radius, max_radius),
                template_path, match_threshold
            )
            self.camera_thread.update_signal.connect(self.update_image)
            self.camera_thread.log_signal.connect(self.update_log)
            self.camera_thread.coords_signal.connect(self.set_tube_coords)
            self.camera_thread.detection_complete_signal.connect(self.update_latest_detection)
            self.camera_thread.start()

            self.start_camera_btn.setText("停止相机")
            self.select_tube_btn.setEnabled(True)
            self.select_place_btn.setEnabled(True)
            self.save_coords_btn.setEnabled(True)
            self.update_log("相机启动成功（模板匹配增强版）")

        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"请检查输入参数: {str(e)}")

    def update_latest_detection(self, detection_dict):
        self.latest_detection = detection_dict

    def update_image(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_log(self, message):
        self.log_text.append(f"{time.strftime('%H:%M:%S')} - {message}")
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def select_tube(self):
        if not self.camera_thread or not self.camera_thread.isRunning():
            QMessageBox.warning(self, "警告", "请先启动相机")
            return
        label, ok = QInputDialog.getText(self, "选择抓取试管", "请输入要抓取的试管序号（如1A）:")
        if ok and label:
            self.camera_thread.select_tube(label.strip())

    def select_place_position(self):
        try:
            rows = int(self.square_rows_edit.text())
            cols = int(self.square_cols_edit.text())
            bottom_x = float(self.square_bottom_left_x_edit.text())
            bottom_y = float(self.square_bottom_left_y_edit.text())
            distance = float(self.square_distance_edit.text())
            offset_x = float(self.square_offset_x_edit.text())
            offset_y = float(self.square_offset_y_edit.text())
            label, ok = QInputDialog.getText(self, "选择放置位置", "请输入放置位置序号（如1A）:")
            if ok and label:
                grid_coords = SquareGridCalculator.calculate_labeled_square_centers(
                    bottom_x, bottom_y, distance, rows, cols, offset_x, offset_y)
                if label in grid_coords:
                    self.place_coords = grid_coords[label]
                    self.place_coords_label.setText(f"({self.place_coords[0]:.4f}, {self.place_coords[1]:.4f})")
                    self.update_log(f"放置位置 {label} 的坐标为: {self.place_coords}（已应用偏移）")
                else:
                    QMessageBox.warning(self, "错误", f"位置 {label} 不在网格范围内")
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"请检查网格参数: {str(e)}")

    def set_tube_coords(self, coords):
        self.tube_coords = coords
        self.grab_coords_label.setText(f"({coords[0]:.4f}, {coords[1]:.4f})")

    def send_data(self):
        try:
            ip = self.ip_edit.text()
            port = int(self.port_edit.text())
            task_number = int(self.task_edit.text())
            height_value = int(self.height_edit.text())
            client = ModbusTcpClient(ip, port=port)
            if not client.connect():
                self.update_log("❌ 连接失败")
                return

            self.update_log("✅ 连接成功")

            if task_number == 6:
                if not self.tube_coords:
                    QMessageBox.warning(self, "警告", "请先选择抓取试管获取坐标")
                    client.close()
                    return
                if not self.place_coords:
                    QMessageBox.warning(self, "警告", "请先选择放置位置")
                    client.close()
                    return

                x_take, y_take = self.tube_coords
                z = float(self.z_edit.text())
                rx = float(self.rx_edit.text())
                ry = float(self.ry_edit.text())
                rz = float(self.rz_edit.text())
                x_place, y_place = self.place_coords

                take_coords = [x_take, y_take, z, rx, ry, rz]
                take_regs = []
                for val in take_coords:
                    reg1, reg2 = self.float32_to_registers(val)
                    take_regs.extend([reg1, reg2])

                place_coords = [x_place, y_place]
                place_regs = []
                for val in place_coords:
                    reg1, reg2 = self.float32_to_registers(val)
                    place_regs.extend([reg1, reg2])
            else:
                take_regs = [0] * 12
                place_regs = [0] * 4

            r1 = client.write_registers(4000, [task_number])
            r2 = client.write_registers(4001, [height_value])
            r3 = client.write_registers(4002, take_regs)
            r4 = client.write_registers(4014, place_regs)

            if r1.isError() or r2.isError() or r3.isError() or r4.isError():
                self.update_log("❌ 写入失败")
                if r1.isError(): self.update_log(f"任务号错误: {r1}")
                if r2.isError(): self.update_log(f"高度错误: {r2}")
                if r3.isError(): self.update_log(f"抓取点坐标错误: {r3}")
                if r4.isError(): self.update_log(f"放置点坐标错误: {r4}")
            else:
                self.update_log(f"✅ 写入成功！任务: {task_number}, 高度: {height_value}")
                if task_number == 6:
                    self.update_log(f"抓取点: X={x_take:.4f}, Y={y_take:.4f}")
                    self.update_log(f"放置点: X={x_place:.4f}, Y={y_place:.4f}")
                    self.update_log(f"Z/RX/RY/RZ: Z={z}, RX={rx}, RY={ry}, RZ={rz}")

        except Exception as e:
            self.update_log(f"❌ 错误: {str(e)}")

        finally:
            if 'client' in locals() and client:
                client.close()
                self.update_log("🔌 连接关闭")

    def float32_to_registers(self, f):
        packed = struct.pack('>f', f)
        high_word, low_word = struct.unpack('>HH', packed)
        return [low_word, high_word]

    def closeEvent(self, event):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        if self.robot_state_thread and self.robot_state_thread.isRunning():
            self.robot_state_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()