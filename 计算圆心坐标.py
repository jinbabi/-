import cv2
import numpy as np
import os


def detect_circles(image_path, use_roi=True):
    """
    检测图片中的圆形并返回处理结果，包括每个圆心的像素值

    参数:
        image_path: 图片路径
        use_roi: 是否使用ROI区域进行检测

    返回:
        result_image: 标记了圆形的图像
        circle_info: 包含每个圆形信息的列表，每个元素为字典
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    original_img = img.copy()
    img_height, img_width = img.shape[:2]

    # 定义ROI区域（可根据实际情况调整）
    roi_x, roi_y = 747, 486
    roi_width, roi_height = 756, 1011

    # 确保ROI在图像范围内
    roi_x = max(0, min(roi_x, img_width - 1))
    roi_y = max(0, min(roi_y, img_height - 1))
    roi_width = max(1, min(roi_width, img_width - roi_x))
    roi_height = max(1, min(roi_height, img_height - roi_y))

    # 根据开关决定是否使用ROI
    if use_roi:
        roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    else:
        roi = img.copy()
        roi_x, roi_y = 0, 0
        roi_width, roi_height = img_width, img_height

    # 转为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 对比度增强
    gray = cv2.equalizeHist(gray)

    # 降噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 圆形检测
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=52,
        param1=50,
        param2=23,
        minRadius=37,
        maxRadius=40
    )

    # 准备输出图像
    output = original_img.copy()

    # 存储圆形信息的列表
    circle_info = []

    # 绘制ROI区域
    if use_roi:
        cv2.rectangle(output, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 0, 255), 2)
        cv2.putText(output, "ROI: ON", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(output, "ROI: OFF", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 显示圆形数量
    circle_count = len(circles[0]) if (circles is not None) else 0
    cv2.putText(output, f"Circle Count: {circle_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 绘制检测到的圆形并获取像素值
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # 提取中心点
        centers = [(x, y, r) for x, y, r in circles]

        # 按y坐标排序（从上到下，用于分组）
        centers.sort(key=lambda p: p[1])
        rows = []
        current_row = [centers[0]] if centers else []
        row_y = centers[0][1] if centers else 0

        # 分组行（同一行的y坐标相近，阈值50）
        for i in range(1, len(centers)):
            if abs(centers[i][1] - row_y) < 50:
                current_row.append(centers[i])
            else:
                rows.append(current_row)
                current_row = [centers[i]]
                row_y = centers[i][1]
        if current_row:
            rows.append(current_row)

        # 关键修改1：将行顺序反转，使最下方的行成为第1行
        rows.reverse()

        # 关键修改2：每行内按x坐标从小到大排序（从左到右：A, B, C...）
        for row in rows:
            row.sort(key=lambda p: p[0])

        # 标记圆形、标签和像素值
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for row_idx, row in enumerate(rows):
            row_number = row_idx + 1  # 行号从1开始（1 = 最下方一行）
            for col_idx, (x, y, r) in enumerate(row):
                # 计算在原始图像中的坐标
                orig_x = x + roi_x
                orig_y = y + roi_y

                # 确保坐标在图像范围内
                orig_x_clamped = max(0, min(orig_x, img_width - 1))
                orig_y_clamped = max(0, min(orig_y, img_height - 1))

                # 获取圆心像素值（BGR格式）
                b, g, r_val = original_img[orig_y_clamped, orig_x_clamped]
                pixel_value = (int(b), int(g), int(r_val))

                # 生成标签：行号 + 列字母（如 1A = 最下行最左圆）
                label = f"{row_number}{letters[col_idx]}"
                circle_info.append({
                    "label": label,
                    "center_coords": (orig_x, orig_y),
                    "radius": r,
                    "pixel_value_bgr": pixel_value,
                    "pixel_value_rgb": (pixel_value[2], pixel_value[1], pixel_value[0])
                })

                # 绘制标签和圆形
                cv2.putText(output, label, (orig_x - 15, orig_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.circle(output, (orig_x, orig_y), 2, (0, 0, 255), 3)
                cv2.circle(output, (orig_x, orig_y), r, (0, 0, 255), 2)

                # 显示像素值（BGR格式）
                pixel_text = f"B:{b},G:{g},R:{r_val}"
                cv2.putText(output, pixel_text, (orig_x - 40, orig_y + r + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return output, circle_info


# 使用示例
if __name__ == "__main__":
    # 输入图片路径
    image_path = "2.jpg"  # 替换为你的图片路径

    if not os.path.exists(image_path):
        print(f"图片不存在: {image_path}")
    else:
        # 检测圆形
        result_img, circle_info = detect_circles(image_path, use_roi=True)

        # 显示结果
        cv2.namedWindow("Circle Detection Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Circle Detection Result", result_img)

        # 输出每个圆形的详细信息
        print("\n圆形详细信息:")
        for info in circle_info:
            print(f"标签: {info['label']}")
            print(f"圆心坐标: {info['center_coords']}")
            print(f"半径: {info['radius']}")
            print(f"像素值(BGR): {info['pixel_value_bgr']}")
            print(f"像素值(RGB): {info['pixel_value_rgb']}")
            print("---")

        # 等待按键关闭窗口
        cv2.waitKey(0)
        cv2.destroyAllWindows()