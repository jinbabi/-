'''import cv2
import numpy as np


def compute_vision_homography(corrected_pixel_coords, gripper_world_coords,
                              offset_x=0, offset_y=0):
    """
    计算校正图像素坐标到圆心世界坐标的单应性矩阵

    参数:
        corrected_pixel_coords: list of (x, y) tuples
            校正图像中的圆心像素坐标
        gripper_world_coords: list of [X, Y] lists
            对应的夹爪中心世界坐标
        offset_x, offset_y: float
            夹爪相对于圆心的偏移量（gripper = vision + offset）

    返回:
        H_vision: 3x3 单应性矩阵
        vision_world_coords: list of [X, Y] 圆心世界坐标（用于验证）
    """
    # Step 1: 从夹爪坐标推算圆心世界坐标
    vision_world_coords = []
    for gx, gy in gripper_world_coords:
        vx = gx - offset_x
        vy = gy - offset_y
        vision_world_coords.append([vx, vy])

    print("✅ 推算出的圆心世界坐标:")
    for i, (vx, vy) in enumerate(vision_world_coords):
        print(f"  试管 {i + 1}: ({vx:.4f}, {vy:.4f}) mm")

    # Step 2: 转换为 numpy 数组
    src_points = np.array(corrected_pixel_coords, dtype=np.float32)  # (N, 2)
    dst_points = np.array(vision_world_coords, dtype=np.float32)  # (N, 2)

    # Step 3: 计算单应性矩阵（使用 RANSAC 提高鲁棒性）
    H_vision, mask = cv2.findHomography(
        srcPoints=src_points,
        dstPoints=dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0  # 允许 2mm 重投影误差
    )

    if H_vision is None:
        raise ValueError("❌ Homography 计算失败！检查输入点是否共线或数量不足")

    # Step 4: 验证重投影误差
    projected_points = cv2.perspectiveTransform(src_points[None, :, :], H_vision)[0]
    errors = np.linalg.norm(dst_points - projected_points, axis=1)
    mean_error = np.mean(errors)

    print(f"\n✅ Homography 标定成功！")
    print(f"📊 平均重投影误差: {mean_error:.4f} mm")
    print(f"🔍 各点误差: {[f'{e:.3f}' for e in errors]} mm")

    if mean_error < 0.5:
        print("🎯 精度评价: 优秀！")
    elif mean_error < 1.0:
        print("✅ 精度评价: 良好")
    else:
        print("⚠️ 精度评价: 需要检查点对应关系或检测精度")

    return H_vision, vision_world_coords


def corrected_pixel_to_vision_world(u_corr, v_corr, H_vision):
    """
    将校正图像中的像素坐标转换为圆心世界坐标

    参数:
        u_corr, v_corr: float
            校正图像中的像素坐标
        H_vision: 3x3 单应性矩阵

    返回:
        vision_x, vision_y: float
            圆心世界坐标 (mm)
    """
    # 齐次坐标
    pixel_homo = np.array([[u_corr, v_corr, 1]], dtype=np.float32).T

    # 应用单应性变换
    world_homo = H_vision @ pixel_homo

    # 透视除法
    vision_x = world_homo[0, 0] / world_homo[2, 0]
    vision_y = world_homo[1, 0] / world_homo[2, 0]

    return vision_x, vision_y


# ========================
# 使用示例：48个点完整标定（按 P1~P49 顺序，跳过 P27）
# ========================
if __name__ == "__main__":
    # 按 P1, P2, P3, ..., P26, P28, ..., P49 顺序排列的校正图圆心像素坐标
    corrected_pixel_coords = [
        (810, 1414),   # 1A
        (924, 1417),   # 1B
        (1045, 1417),  # 1C
        (1168, 1427),  # 1D
        (1290, 1424),  # 1E
        (1413, 1421),  # 1F
        (811, 1291),   # 2A
        (932, 1299),   # 2B
        (1048, 1300),  # 2C
        (1169, 1300),  # 2D
        (1290, 1300),  # 2E
        (1412, 1300),  # 2F
        (813, 1176),   # 3A
        (928, 1178),   # 3B
        (1048, 1177),  # 3C
        (1168, 1183),  # 3D
        (1294, 1181),  # 3E
        (1415, 1183),  # 3F
        (810, 1055),   # 4A
        (928, 1059),   # 4B
        (1045, 1055),  # 4C
        (1166, 1055),  # 4D
        (1285, 1059),  # 4E
        (1415, 1056),  # 4F
        (810, 935),    # 5A
        (930, 930),    # 5B
        (1051, 934),   # 5C
        (1169, 934),   # 5D
        (1290, 934),   # 5E
        (1416, 933),   # 5F
        (809, 809),    # 6A
        (934, 814),    # 6B
        (1049, 814),   # 6C
        (1171, 817),   # 6D
        (1291, 809),   # 6E
        (1415, 810),   # 6F
        (811, 699),    # 7A
        (928, 700),    # 7B
        (1052, 696),   # 7C
        (1170, 691),   # 7D
        (1294, 697),   # 7E
        (1412, 692),   # 7F
        (814, 579),    # 8A
        (932, 572),    # 8B
        (1053, 576),   # 8C
        (1169, 570),   # 8D
        (1291, 574),   # 8E
        (1411, 565),   # 8F
    ]

    # 对应的夹爪中心世界坐标（按相同顺序）
    gripper_world_coords = [
        [231.6762, -405.6024],  # 1A
        [298.7882, -409.6864],  # 1F
        [236.0762, -311.6644],  # 8A
        [304.6263, -315.1324],  # 8F
        [261.4922, -366.3364],  # 4C
        [274.1402, -367.2633],  # 4D
        [261.5302, -353.7793],  # 5C
        [274.8802, -353.6793],  # 5D
        [232.9976, -393.071],   # 2A
        [259.0076, -407.147],   # 1C
        [272.3003, -408.247],   # 1D
        [264.4463, -312.265],   # 8C
        [277.4203, -313.081],   # 8D
        [234.1923, -364.673],   # 4A
        [235.1923, -351.739],   # 5A
        [302.1862, -367.651],   # 4F
        [302.1862, -354.779],   # 5F
        [245.9643, -392.233],   # 2B
        [286.9683, -394.233],   # 2E
        [249.7383, -324.363],   # 7B
        [291.3542, -327.015],   # 7E
        [260.8563, -380.279],   # 3C
        [274.2183, -380.279],   # 3D
        #[236.3563, -338.969],   # 6A
        [276.1563, -339.969],   # 6D
        [272.8922, -393.805],   # 2D
        [246.3362, -409.2158],  # 1B
        [286.6263, -410.2158],  # 1E
        [260.8203, -394.7118],  # 2C
        [301.1103, -397.3638],  # 2F
        [234.0762, -380.6154],  # 3A
        [247.4382, -380.6154],  # 3B
        [287.8302, -383.1654],  # 3E
        [302.2943, -383.1654],  # 3F
        [248.6422, -367.8654],  # 4B
        [275.5902, -369.0004],  # 4E
        [249.2943, -353.3944],  # 5B
        [276.4952, -355.2864],  # 5E
        [249.2943, -339.7464],  # 6B
        [263.6232, -340.9464],  # 6C
        [291.0392, -342.5984],  # 6E
        [304.3812, -343.5984],  # 6F
        [238.8152, -326.4624],  # 7A
        [265.8452, -326.4624],  # 7C
        [278.7992, -327.6624],  # 7D
        [304.4032, -330.7224],  # 7F
        [252.3892, -314.4044],  # 8B
        [278.8072, -314.4044],  # 8E
    ]

    # 已知偏移量（假设夹爪中心与圆心重合）
    OFFSET_X = 0
    OFFSET_Y = 0

    # 计算 Homography
    H_vision, vision_world_coords = compute_vision_homography(
        corrected_pixel_coords,
        gripper_world_coords,
        offset_x=OFFSET_X,
        offset_y=OFFSET_Y
    )

    # 直接输出单应性矩阵
    print(f"\n💾 单应性矩阵:")
    print(H_vision)

    # 测试转换函数：输入任意像素坐标（例如 5C 的像素）
    test_u, test_v = 1505, 1261  # 5C 的像素坐标
    vx, vy = corrected_pixel_to_vision_world(test_u, test_v, H_vision)
    print(f"\n🧪 测试转换:")
    print(f"  输入像素: ({test_u}, {test_v})")
    print(f"  输出圆心世界坐标: ({vx:.4f}, {vy:.4f}) mm")


'''
import cv2
import numpy as np


def compute_vision_homography(corrected_pixel_coords, gripper_world_coords,
                              offset_x=0, offset_y=0):
    """
    计算校正图像素坐标到圆心世界坐标的单应性矩阵

    参数:
        corrected_pixel_coords: list of (x, y) tuples
            校正图像中的圆心像素坐标
        gripper_world_coords: list of [X, Y] lists
            对应的夹爪中心世界坐标
        offset_x, offset_y: float
            夹爪相对于圆心的偏移量（gripper = vision + offset）

    返回:
        H_vision: 3x3 单应性矩阵
        vision_world_coords: list of [X, Y] 圆心世界坐标（用于验证）
    """
    # Step 1: 从夹爪坐标推算圆心世界坐标
    vision_world_coords = []
    for gx, gy in gripper_world_coords:
        vx = gx - offset_x
        vy = gy - offset_y
        vision_world_coords.append([vx, vy])

    print("✅ 推算出的圆心世界坐标:")
    for i, (vx, vy) in enumerate(vision_world_coords):
        print(f"  试管 {i + 1}: ({vx:.4f}, {vy:.4f}) mm")

    # Step 2: 转换为 numpy 数组
    src_points = np.array(corrected_pixel_coords, dtype=np.float32)  # (N, 2)
    dst_points = np.array(vision_world_coords, dtype=np.float32)  # (N, 2)

    # Step 3: 计算单应性矩阵（使用 RANSAC 提高鲁棒性）
    H_vision, mask = cv2.findHomography(
        srcPoints=src_points,
        dstPoints=dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=2.0  # 允许 2mm 重投影误差
    )

    if H_vision is None:
        raise ValueError(" Homography 计算失败！检查输入点是否共线或数量不足")

    # Step 4: 验证重投影误差
    projected_points = cv2.perspectiveTransform(src_points[None, :, :], H_vision)[0]
    errors = np.linalg.norm(dst_points - projected_points, axis=1)
    mean_error = np.mean(errors)

    print(f"\n✅ Homography 标定成功！")
    print(f"📊 平均重投影误差: {mean_error:.4f} mm")
    print(f"🔍 各点误差: {[f'{e:.3f}' for e in errors]} mm")

    if mean_error < 0.5:
        print("🎯 精度评价: 优秀！")
    elif mean_error < 1.0:
        print("✅ 精度评价: 良好")
    else:
        print("⚠️ 精度评价: 需要检查点对应关系或检测精度")

    return H_vision, vision_world_coords


def corrected_pixel_to_vision_world(u_corr, v_corr, H_vision):
    """
    将校正图像中的像素坐标转换为圆心世界坐标

    参数:
        u_corr, v_corr: float
            校正图像中的像素坐标
        H_vision: 3x3 单应性矩阵

    返回:
        vision_x, vision_y: float
            圆心世界坐标 (mm)
    """
    # 齐次坐标
    pixel_homo = np.array([[u_corr, v_corr, 1]], dtype=np.float32).T

    # 应用单应性变换
    world_homo = H_vision @ pixel_homo

    # 透视除法
    vision_x = world_homo[0, 0] / world_homo[2, 0]
    vision_y = world_homo[1, 0] / world_homo[2, 0]

    return vision_x, vision_y


# ========================
# 使用示例
# ========================
if __name__ == "__main__":
    # 按指定标签顺序排列的校正图圆心坐标
    corrected_pixel_coords = [
        (810, 1414),   # 1A
        (1045, 1417),  # 1C
        (1168, 1427),  # 1D
        (1413, 1421),  # 1F
        (811, 1291),   # 2A
        (932, 1299),   # 2B
        (1290, 1300),  # 2E
        (1048, 1177),  # 3C
        (1168, 1183),  # 3D
        (810, 1055),   # 4A
        (1415, 1056),  # 4F
        (810, 935),    # 5A
        (1416, 933),   # 5F
        (809, 809),    # 6A
        (1171, 817),   # 6D
        (928, 700),    # 7B
        (1294, 697),   # 7E
        (814, 579),    # 8A
        (1053, 576),   # 8C
        (1169, 570),   # 8D
        (1411, 565),   # 8F

    ]

    # 2. 对应的夹爪中心世界坐标（手动记录的）
    gripper_world_coords = [
        [-680.474,   266.8965],   # P2  -> 1A
        [-679.474,   240.0477],   # P3  -> 1C
        [-680.474,   225.9997],   # P4  -> 1D
        [-679.474,   198.1997],   # P5  -> 1F
        [-658.0092,  264.2287],   # P6  -> 2A
        [-658.6592,  251.9287],   # P7  -> 2B
        [-657.9064,  211.5131],   # P8  -> 2E
        [-645.0056,  237.9735],   # P9  -> 3C
        [-644.9056,  224.9511],   # P10 -> 3D
        [-632.2617,  265.0511],   # P11 -> 4A
        [-638.9133,  199.8485],   # P12 -> 4F
        [-626.2982,  266.25],     # P13 -> 5A
        [-625.5361,  200.0831],   # P14 -> 5F
        [-612.2166,  266.0004],   # P15 -> 6A
        [-612.2166,  227.8544],   # P16 -> 6D
        [-599.6686,  255.1024],   # P17 -> 7B
        [-599.4686,  212.9822],   # P18 -> 7E
        [-586.2813,  268.6287],   # P19 -> 8A
        [-585.7133,  240.3561],   # P20 -> 8C
        [-585.1785,  228.7005],   # P21 -> 8D
        [-585.0785,  200.4043],   # P22 -> 8F
    ]

    # 3. 已知偏移量
    OFFSET_X = 0
    OFFSET_Y = 0

    # 4. 计算 Homography
    H_vision, vision_world_coords = compute_vision_homography(
        corrected_pixel_coords,
        gripper_world_coords,
        offset_x=OFFSET_X,
        offset_y=OFFSET_Y
    )

    # 5. 直接输出单应性矩阵
    print(f"\n💾 单应性矩阵:")
    print(H_vision)

    # 6. 测试转换函数
    test_u, test_v = 1169, 934 # 测试一个已知点
    vx, vy = corrected_pixel_to_vision_world(test_u, test_v, H_vision)
    print(f"\n🧪 测试转换:")
    print(f"  输入像素: ({test_u}, {test_v})")
    print(f"  输出圆心世界坐标: ({vx:.4f}, {vy:.4f}) mm")
