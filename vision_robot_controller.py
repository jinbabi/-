"""
视觉与机器人控制核心功能管理器
- 支持固定编号试管检测（基于 JSON 预定义位置）
- 支持模板匹配 + 圆检测融合
- 支持空盒占用状态管理
- 完全解耦 UI，通过 Signal 通信
- ✅ 新增：接收 HTTP 8080 数据后，向 UI 发送 new_task_received 信号以更新任务表格
"""
import struct
import cv2
import numpy as np
import time
import json
import os
import socket
from pymodbus.client import ModbusTcpClient
from MvCameraControl_class import *
from CameraParams_header import *
from ctypes import *
from PySide6.QtCore import QObject, Signal, QThread
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

# ========== 工具函数 ==========
def load_tube_positions(json_path):
    """从 JSON 文件加载固定试管位置（原始像素坐标）。"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        positions = {}
        for label, info in data.items():
            orig_px = info.get("original_pixel")
            if orig_px and len(orig_px) == 2:
                positions[label] = (int(orig_px[0]), int(orig_px[1]))
        return positions
    except Exception as e:
        return {}

def parse_gripper_31005(reg_value):
    """根据新规则解析寄存器 31005 的夹爪状态。"""
    mapping = {
        4: "管夹爪关盒夹爪关",
        6: "管夹爪开盒夹爪关",
        8: "管夹爪关盒夹爪开",
        10: "管夹爪开盒夹爪开"
    }
    return mapping.get(reg_value, f"未知状态({reg_value})")

def template_match_on_image(img, fixed_template_gray, threshold):
    """在图像中执行模板匹配，并返回去重后的检测结果。"""
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
    """判断圆心是否落在模板匹配框内。"""
    return tx <= circle_x <= tx + tw and ty <= circle_y <= ty + th

# ========== HTTP 请求处理器 ==========
class TargetTubeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode('utf-8'))
            box_code = data.get('box_code')
            tube_codes = data.get('tube_codes', [])
            if not isinstance(box_code, str) or not isinstance(tube_codes, list):
                raise ValueError("Invalid format")
            self.server.controller.set_target_tubes(box_code, tube_codes)
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
        except Exception as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode('utf-8'))
    def log_message(self, format, *args):
        return  # 禁用日志

# ========== 相机线程 ==========（保持完整）
class CameraThread(QThread):
    update_signal = Signal(np.ndarray)
    coords_signal = Signal(tuple)
    log_signal = Signal(str)
    def __init__(self, rotation_angle, H_vision, offset_x, offset_y, use_roi, roi_params,
                 circle_params, template_path, match_threshold, fixed_tube_positions, parent_controller):
        super().__init__()
        self.parent_controller = parent_controller
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
        self.fixed_tube_positions = fixed_tube_positions
        self.detected_circles = {}
        self.selected_label = None
        self.max_circle_count = 0
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
        target_ip = "192.168.5.100"
        selected_device_index = -1
        camera = MvCamera()
        for i in range(deviceList.nDeviceNum):
            stDeviceList = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if stDeviceList.nTLayerType == MV_GIGE_DEVICE:
                try:
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
                        ret = camera.MV_CC_CreateHandle(stDeviceList)
                        if ret != 0:
                            self.log_signal.emit(f"为设备 {i} 创建句柄失败，错误码: {ret}")
                            selected_device_index = -1
                            continue
                        break
                except Exception as e:
                    self.log_signal.emit(f"获取设备 {i} 信息失败: {e}")
        if selected_device_index == -1:
            self.log_signal.emit(f"❌ 未找到IP为 {target_ip} 的相机")
            return
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
        orig_to_corr_matrix = None
        try:
            self.log_signal.emit("正在检测试管（固定编号 + 模板匹配），请稍候...")
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
                        orig_to_corr_matrix = self.get_original_to_corrected_matrix(inv_transform_matrix)
                        angle_calibrated = True
                    corrected_frame, _ = self.rotate_image(frame, self.rotation_angle)
                    frame_result, _, _ = self.opencv_action(
                        corrected_frame, transform_matrix, inv_transform_matrix)
                    self.update_signal.emit(frame_result)
                    if self.selected_label is not None:
                        if self.selected_label in self.detected_circles and self.detected_circles[
                            self.selected_label] is not None:
                            orig_x, orig_y = self.detected_circles[self.selected_label]
                            world_x, world_y = self.original_to_world(
                                orig_x, orig_y, orig_to_corr_matrix, self.H_vision)
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
        orig_homo = np.array([u_orig, v_orig, 1], dtype=np.float32).T
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
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=self.dp,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
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
        self.detected_circles.clear()
        valid_circle_list = [(x, y, r) for x, y, r in valid_circles]
        for label, (orig_x, orig_y) in self.fixed_tube_positions.items():
            corr_x, corr_y = self.original_to_corrected(orig_x, orig_y, M)
            corr_x, corr_y = int(round(corr_x)), int(round(corr_y))
            if not (0 <= corr_x < full_width and 0 <= corr_y < full_height):
                self.detected_circles[label] = None
                self.parent_controller.tube_detection_updated.emit(label, False)
                continue
            matched_circle = None
            for (cx, cy, r) in valid_circle_list:
                if abs(corr_x - cx) <= 20 and abs(corr_y - cy) <= 20:
                    matched_circle = (cx, cy)
                    break
            color = (0, 255, 0) if matched_circle is not None else (0, 0, 255)
            cv2.putText(output, label, (corr_x - 15, corr_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.circle(output, (corr_x, corr_y), 2, color, 3)
            if matched_circle is not None:
                cv2.circle(output, (corr_x, corr_y), 38, color, 2)
            if matched_circle is not None:
                cx_corr, cy_corr = matched_circle
                orig_cx, orig_cy = self.get_original_coords(cx_corr, cy_corr, M_inv)
                self.detected_circles[label] = (orig_cx, orig_cy)
                self.parent_controller.tube_detection_updated.emit(label, True)
            else:
                self.detected_circles[label] = None
                self.parent_controller.tube_detection_updated.emit(label, False)
        for idx, (x, y, score) in enumerate(template_detections):
            if idx in matched_template_indices:
                cv2.rectangle(output, (x, y), (x + self.TEMPLATE_W, y + self.TEMPLATE_H), (0, 255, 0), 2)
                cv2.putText(output, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return output, gray, blurred

# ========== 机器人状态线程 ==========
class RobotStateThread(QThread):
    update_robot_state_signal = Signal(dict)
    update_gripper_state_signal = Signal(str)
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
                    gripper_31005_state = parse_gripper_31005(reg_31005)
                self.update_robot_state_signal.emit(robot_state)
                self.update_gripper_state_signal.emit(gripper_31005_state)
                client.close()
            except Exception as e:
                self.log_signal.emit(f"❌ 机器人状态线程异常: {str(e)}")
            time.sleep(self.interval)
    def stop(self):
        self.running = False
        self.wait()

# ========== 扫码器线程 ==========
class BarcodeScannerThread(QThread):
    barcode_data_received = Signal(dict, str)
    log_signal = Signal(str)
    def __init__(self, parent_controller):
        super().__init__()
        self.parent_controller = parent_controller
        self.running = False
        self.ip_address = "192.168.5.3"
        self.port = 2001
    def run(self):
        self.running = True
        self.log_signal.emit("📡 开始连接扫码器...")
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5.0)
            client_socket.connect((self.ip_address, self.port))
            self.log_signal.emit("✅ 连接扫码器成功！")
            client_socket.send(b"start")
            self.log_signal.emit("📤 触发指令已发送！")
            time.sleep(2)
            client_socket.settimeout(3.0)
            buffer = b""
            while True:
                try:
                    chunk = client_socket.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                except socket.timeout:
                    break
            client_socket.close()
            try:
                text = buffer.decode('utf-8')
            except UnicodeDecodeError:
                text = buffer.decode('latin-1', errors='replace')
            self.log_signal.emit("\n=====================================")
            self.log_signal.emit("📡 设备返回的原始数据:")
            self.log_signal.emit("-------------------------------------")
            self.log_signal.emit(text)
            self.log_signal.emit("\n=====================================")
            self.log_signal.emit(f"📦 总共接收 {len(buffer)} 字节数据")
            mapping, box_code = self.parse_tube_data(text)
            if mapping is None:
                self.log_signal.emit("❌ 数据解析失败。")
                return
            self.barcode_data_received.emit(mapping, box_code)
            self.log_signal.emit("✅ 扫码数据解析完成，结果已发送至界面。")
        except Exception as ex:
            error_msg = f"❌ 扫码器通信发生错误: {str(ex)}"
            self.log_signal.emit(error_msg)
            self.barcode_data_received.emit({}, error_msg)
    def parse_tube_data(self, raw_text):
        lines = raw_text.strip().splitlines()
        if len(lines) < 50:
            self.log_signal.emit("⚠️  警告：接收到的数据少于50行，可能不完整。")
            return None, None
        tube_codes = []
        for i in range(48):
            parts = lines[i].split(',', 1)
            if len(parts) < 2:
                tube_codes.append("NoRead")
            else:
                tube_codes.append(parts[1].rstrip(';'))
        box_code = lines[48].split(',', 1)[1].rstrip(';') if len(lines[48].split(',', 1)) > 1 else "NoRead"
        mapping = {}
        rows = list(range(8, 0, -1))
        cols = ['A', 'B', 'C', 'D', 'E', 'F']
        for i in range(48):
            row_index = i // 6
            col_index = i % 6
            row = rows[row_index]
            col = cols[col_index]
            pos = f"{row}{col}"
            mapping[pos] = tube_codes[i]
        return mapping, box_code

# ========== 网格计算 ==========
class SquareGridCalculator:
    @staticmethod
    def calculate_labeled_square_centers(x_1A, y_1A, d, rows, cols, offset_x=0, offset_y=0):
        coordinates = {}
        col_labels = [chr(ord('A') + i) for i in range(cols)]
        for row_idx in range(rows):
            row_label = row_idx + 1
            for col_idx, col_label in enumerate(col_labels):
                center_x = x_1A + row_idx * d + offset_x
                center_y = y_1A - col_idx * d + offset_y
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
        return "".join(result)

# ========== 核心功能管理器 ==========
class VisionRobotController(QObject):
    image_updated = Signal(np.ndarray)
    log_message = Signal(str)
    tube_coords_updated = Signal(tuple)
    robot_state_updated = Signal(dict)
    gripper_state_updated = Signal(str)
    tube_detection_updated = Signal(str, bool)
    barcode_data_received = Signal(dict, str)
    comparison_result_ready = Signal(dict, str, dict, list)  # mapping, box_code, found_dict, missing_list

    # ========== ✅ 新增信号：用于更新主界面任务表格 ==========
    new_task_received = Signal(str, str, str, str, str)  # 任务编号, 任务类型, 任务编码, 任务状态, 接收时间

    def __init__(self):
        super().__init__()
        self.load_configs()
        self.init_params()
        self.camera_thread = None
        self.robot_state_thread = None
        self.barcode_scanner_thread = None
        self.tube_coords = None
        self.place_coords = None
        self.selected_place_label = None
        self.fixed_tube_positions = {}
        self.occupied_place_positions = {}
        self.tube_position_files = {
            "8*6试管盒": "8X6.json",
            "12*8试管盒(2ml)": "12X8(2ml).json",
            "12*8试管盒(1ml)": "12X8(1ml).json",
            "4*4试管盒": "4X4.json"
        }
        self.target_box_code = None
        self.target_tube_codes = []
        self.last_scan_result = None
        self.start_http_server()

    def load_configs(self):
        try:
            with open("tube_config.json", "r", encoding="utf-8") as f:
                self.tube_config = json.load(f)
        except:
            self.tube_config = {}
        try:
            with open("square_grid_config.json", "r", encoding="utf-8") as f:
                self.square_grid_config = json.load(f)
        except:
            self.square_grid_config = {}

    def init_params(self):
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
        self.min_dist = 52
        self.param1 = 50
        self.param2 = 23
        self.min_radius = 37
        self.max_radius = 40
        self.rotation_angle = 187.0
        self.H_vision = np.array([
            [-1.71154336e-02, -1.13269066e-01, -5.19657998e+02],
            [-1.08948255e-01, -2.99949295e-03, 3.64389627e+02],
            [2.88956236e-05, -2.38136201e-06, 1.00000000e+00]
        ])
        self.square_rows = 5
        self.square_cols = 5
        self.square_bottom_left_x = 100.0
        self.square_bottom_left_y = 100.0
        self.square_distance = 30.0
        self.square_offset_x = 0.0
        self.square_offset_y = 1.0
        self.z = 721.3144
        self.rx = -0.5395
        self.ry = -3.5085
        self.rz = 98.1735
        self.task_number = 0
        self.height_value = 0
        self.selected_box_type = "8*6试管盒"
        self.selected_square_type = "8*6空盒"
        self.template_path = "template_8x6.jpg"
        self.match_threshold = 0.35
        self.robot_state = {
            "auto_exit": False, "ready": False, "paused": False,
            "running": False, "alarm": False, "data_valid": 0,
            "digital_in": 0, "digital_out": 0
        }
        self.gripper_31005_state = "未读取"
        self.matrix_edit_values = [
            self.H_vision[0][0], self.H_vision[0][1], self.H_vision[0][2],
            self.H_vision[1][0], self.H_vision[1][1], self.H_vision[1][2],
            self.H_vision[2][0], self.H_vision[2][1], self.H_vision[2][2]
        ]

    def on_box_type_changed(self, box_type):
        self.selected_box_type = box_type
        if box_type in self.tube_config:
            p = self.tube_config[box_type]
            self.dp = p["dp"]
            self.min_radius = p["min_radius"]
            self.max_radius = p["max_radius"]
            self.height_value = p["height_value"]
            self.offset_x = p["offset_x"]
            self.offset_y = p["offset_y"]
            self.min_dist = p.get("minDist", 50)
            self.param1 = p.get("param1", 50)
            self.param2 = p.get("param2", 23)
            self.template_path = p.get("template_path", "template.jpg")
            self.match_threshold = p.get("match_threshold", 0.4)
            self.log_message.emit(f"已加载{box_type}的参数配置")
        json_file = self.tube_position_files.get(box_type)
        if json_file and os.path.exists(json_file):
            self.fixed_tube_positions = load_tube_positions(json_file)
            self.log_message.emit(f"✅ 已加载试管位置文件: {json_file}，共 {len(self.fixed_tube_positions)} 个位置")
        else:
            self.fixed_tube_positions = {}
            self.log_message.emit(f"⚠️ 未找到试管位置文件: {json_file}")

    def on_square_type_changed(self, square_type):
        self.selected_square_type = square_type
        if square_type in self.square_grid_config:
            p = self.square_grid_config[square_type]
            self.square_rows = p["rows"]
            self.square_cols = p["cols"]
            self.square_bottom_left_x = p["bottom_left_x"]
            self.square_bottom_left_y = p["bottom_left_y"]
            self.square_distance = p["distance"]
            self.square_offset_x = p.get("offset_x", 0.0)
            self.square_offset_y = p.get("offset_y", 0.0)
            self.log_message.emit(f"已加载空盒{square_type}的参数配置（包含偏移量）")
            col_labels = [chr(ord('A') + i) for i in range(self.square_cols)]
            self.occupied_place_positions = {
                f"{row}{col}": False
                for row in range(1, self.square_rows + 1)
                for col in col_labels
            }
            self.log_message.emit(f"✅ 空盒 {square_type} 的 {self.square_rows}×{self.square_cols} 位置已初始化为空")

    # ========== ✅ 修改：触发 new_task_received 信号 ==========
    def set_target_tubes(self, box_code: str, tube_codes: list):
        self.target_box_code = box_code
        self.target_tube_codes = tube_codes
        self.log_message.emit(f"设置目标：盒码 {box_code}, 管码 {tube_codes}")
        # 生成任务信息
        task_id = str(int(time.time()))
        task_type = "扫码任务"
        task_code = f"{box_code} - {', '.join(tube_codes)}"
        task_status = "待执行"
        receive_time = time.strftime('%Y-%m-%d %H:%M:%S')
        # ✅ 发出信号，供 UI 添加到表格
        self.new_task_received.emit(task_id, task_type, task_code, task_status, receive_time)

    def start_http_server(self):
        server_address = ('', 8083)
        httpd = HTTPServer(server_address, TargetTubeHandler)
        httpd.controller = self
        self.http_server = httpd
        self.http_thread = threading.Thread(target=httpd.serve_forever)
        self.http_thread.daemon = True
        self.http_thread.start()
        self.log_message.emit("✅ HTTP 服务器启动，监听 8080 端口")

    def stop_http_server(self):
        if hasattr(self, 'http_server') and self.http_server:
            self.http_server.shutdown()
            self.http_server.server_close()
            self.http_server = None

    def _on_barcode_data_received(self, mapping, box_code):
        self.last_scan_result = {'box_code': box_code, 'mapping': mapping}
        self.barcode_data_received.emit(mapping, box_code)
        found_tubes = {}
        missing_tubes = []
        if self.target_box_code != box_code:
            self.comparison_result_ready.emit(
                mapping, box_code, found_tubes, [f"盒码不匹配! 目标:{self.target_box_code}"]
            )
            return
        for target in self.target_tube_codes:
            found = False
            for pos, code in mapping.items():
                if code == target:
                    found_tubes[pos] = code
                    found = True
                    break
            if not found:
                missing_tubes.append(target)
        self.comparison_result_ready.emit(mapping, box_code, found_tubes, missing_tubes)

    def start_camera(self):
        try:
            H_vision = np.array([
                [float(self.matrix_edit_values[0]), float(self.matrix_edit_values[1]), float(self.matrix_edit_values[2])],
                [float(self.matrix_edit_values[3]), float(self.matrix_edit_values[4]), float(self.matrix_edit_values[5])],
                [float(self.matrix_edit_values[6]), float(self.matrix_edit_values[7]), float(self.matrix_edit_values[8])]
            ])
        except:
            H_vision = self.H_vision
        self.camera_thread = CameraThread(
            self.rotation_angle, H_vision, self.offset_x, self.offset_y,
            self.use_roi, (self.roi_x, self.roi_y, self.roi_width, self.roi_height),
            (self.dp, self.min_dist, self.param1, self.param2, self.min_radius, self.max_radius),
            self.template_path, self.match_threshold,
            self.fixed_tube_positions, self
        )
        self.camera_thread.update_signal.connect(self.image_updated)
        self.camera_thread.log_signal.connect(self.log_message)
        self.camera_thread.coords_signal.connect(self._on_tube_coords)
        self.camera_thread.start()
        self.log_message.emit("相机启动成功（固定编号版）")

    def stop_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None
            self.log_message.emit("相机已关闭")

    def select_tube(self, label: str):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.select_tube(label)

    def _on_tube_coords(self, coords):
        self.tube_coords = coords
        self.tube_coords_updated.emit(coords)

    def select_place_position(self, label: str):
        try:
            label = label.strip().upper()
            coords = SquareGridCalculator.calculate_labeled_square_centers(
                self.square_bottom_left_x, self.square_bottom_left_y,
                self.square_distance, self.square_rows, self.square_cols,
                self.square_offset_x, self.square_offset_y
            )
            if label not in coords:
                self.log_message.emit(f"错误：位置 {label} 不在网格范围内")
                return
            if self.occupied_place_positions.get(label, False):
                self.log_message.emit(f"⚠ 位置 {label} 已被占用，不能重复放置！")
                return
            self.selected_place_label = label
            self.place_coords = coords[label]
            self.log_message.emit(f"放置位置 {label} 坐标: {self.place_coords}")
        except Exception as e:
            self.log_message.emit(f" 选择放置位置失败: {str(e)}")

    def clear_place_positions(self):
        for key in self.occupied_place_positions:
            self.occupied_place_positions[key] = False
        self.log_message.emit(" 空盒所有位置已重置为空")

    def send_data(self):
        try:
            client = ModbusTcpClient(self.ip, port=self.port)
            if not client.connect():
                self.log_message.emit(" Modbus 连接失败")
                return
            if self.task_number == 6:
                if not self.tube_coords:
                    self.log_message.emit(" 未选择抓取试管")
                    return
                if not self.place_coords:
                    self.log_message.emit("❌ 未选择放置位置")
                    return
                x_take, y_take = self.tube_coords
                x_place, y_place = self.place_coords
                z, rx, ry, rz = self.z, self.rx, self.ry, self.rz
                take_regs = []
                for val in [x_take, y_take, z, rx, ry, rz]:
                    take_regs.extend(self.float32_to_registers(val))
                place_regs = []
                for val in [x_place, y_place]:
                    place_regs.extend(self.float32_to_registers(val))
            else:
                take_regs = [0] * 12
                place_regs = [0] * 4
            r1 = client.write_registers(4000, [self.task_number])
            r2 = client.write_registers(4001, [self.height_value])
            r3 = client.write_registers(4002, take_regs)
            r4 = client.write_registers(4014, place_regs)
            if not (r1.isError() or r2.isError() or r3.isError() or r4.isError()):
                task_names = {1: "大夹爪归位", 11: "小夹爪归位", 12: "到拍摄点"}
                task_name = task_names.get(self.task_number, f"任务{self.task_number}")
                self.log_message.emit(f"✅ 数据发送成功: {task_name} (任务={self.task_number})")
                if self.task_number == 6 and self.selected_place_label:
                    self.occupied_place_positions[self.selected_place_label] = True
                    self.log_message.emit(f"📌 位置 {self.selected_place_label} 已标记为占用")
            else:
                self.log_message.emit("❌ 数据发送失败")
            client.close()
        except Exception as e:
            self.log_message.emit(f"❌ 发送异常: {str(e)}")

    def float32_to_registers(self, f):
        packed = struct.pack('>f', f)
        high_word, low_word = struct.unpack('>HH', packed)
        return [low_word, high_word]

    def start_robot_state_thread(self):
        self.robot_state_thread = RobotStateThread(self.ip, self.port, interval=1.0)
        self.robot_state_thread.update_robot_state_signal.connect(self._on_robot_state)
        self.robot_state_thread.update_gripper_state_signal.connect(self._on_gripper_state)
        self.robot_state_thread.log_signal.connect(self.log_message)
        self.robot_state_thread.start()

    def _on_robot_state(self, state_dict):
        self.robot_state.update(state_dict)
        self.robot_state_updated.emit(state_dict)

    def _on_gripper_state(self, state_31005):
        self.gripper_31005_state = state_31005
        self.gripper_state_updated.emit(state_31005)

    def get_grid_coordinates_text(self):
        try:
            coords = SquareGridCalculator.calculate_labeled_square_centers(
                self.square_bottom_left_x, self.square_bottom_left_y,
                self.square_distance, self.square_rows, self.square_cols,
                self.square_offset_x, self.square_offset_y
            )
            return SquareGridCalculator.print_labeled_coordinates(coords, self.square_rows, self.square_cols)
        except Exception as e:
            return f"网格计算错误: {e}"

    def shutdown(self):
        self.stop_camera()
        if self.robot_state_thread and self.robot_state_thread.isRunning():
            self.robot_state_thread.stop()
        if self.barcode_scanner_thread and self.barcode_scanner_thread.isRunning():
            self.barcode_scanner_thread.stop()
        self.stop_http_server()

    # ========== 参数设置接口 ==========
    def set_ip(self, ip: str): self.ip = ip
    def set_port(self, port: int): self.port = port
    def set_offset_x(self, val: float): self.offset_x = val
    def set_offset_y(self, val: float): self.offset_y = val
    def set_use_roi(self, val: bool): self.use_roi = val
    def set_roi_params(self, x: int, y: int, w: int, h: int): self.roi_x, self.roi_y, self.roi_width, self.roi_height = x, y, w, h
    def set_circle_params(self, dp, min_dist, param1, param2, min_radius, max_radius):
        self.dp = dp; self.min_dist = min_dist; self.param1 = param1; self.param2 = param2
        self.min_radius = min_radius; self.max_radius = max_radius
    def set_template_params(self, path: str, threshold: float):
        self.template_path = path; self.match_threshold = threshold
    def set_rotation_angle(self, angle: float): self.rotation_angle = angle
    def set_matrix_values(self, values: list): self.matrix_edit_values = values
    def set_square_params(self, rows, cols, x, y, d, ox, oy):
        self.square_rows = rows; self.square_cols = cols
        self.square_bottom_left_x = x; self.square_bottom_left_y = y
        self.square_distance = d; self.square_offset_x = ox; self.square_offset_y = oy
    def set_fixed_pose(self, z, rx, ry, rz):
        self.z = z; self.rx = rx; self.ry = ry; self.rz = rz
    def set_task_params(self, task: int, height: int):
        self.task_number = task; self.height_value = height

    def read_robot_state_once(self):
        try:
            client = ModbusTcpClient(self.ip, port=self.port)
            if not client.connect():
                self.log_message.emit(" 主动读取：Modbus 连接失败")
                return
            discrete_result = client.read_discrete_inputs(address=0, count=5)
            input_result = client.read_input_registers(address=999, count=13)
            if not discrete_result.isError() and not input_result.isError():
                bits = discrete_result.bits[:5]
                regs = input_result.registers
                state_dict = {
                    "auto_exit": bits[0], "ready": bits[1], "paused": bits[2],
                    "running": bits[3], "alarm": bits[4], "data_valid": regs[0],
                    "digital_in": (regs[5] << 48) | (regs[6] << 32) | (regs[7] << 16) | regs[8],
                    "digital_out": (regs[9] << 48) | (regs[10] << 32) | (regs[11] << 16) | regs[12]
                }
                self._on_robot_state(state_dict)
                self._on_gripper_state(parse_gripper_31005(regs[5]))
            client.close()
        except Exception as e:
            self.log_message.emit(f" 主动读取异常: {str(e)}")

    def start_barcode_scanning(self):
        if self.barcode_scanner_thread and self.barcode_scanner_thread.isRunning():
            self.log_message.emit(" 扫码器线程已在运行中，请勿重复启动。")
            return
        self.barcode_scanner_thread = BarcodeScannerThread(self)
        self.barcode_scanner_thread.barcode_data_received.connect(self._on_barcode_data_received)
        self.barcode_scanner_thread.log_signal.connect(self.log_message)
        self.barcode_scanner_thread.start()
        self.log_message.emit(" 扫码器线程已启动，正在等待数据...")
