                  Pick-Tube Vision System
硬件环境：配套机械臂:越疆nova5机械手臂、相机海康MV-CU060-10GC 、海康扫码器、pc、Modbus TCP 、IO  IP：锁控正常。
软件环境：Python
          Dobotstudio 
基于工业相机的视觉定位与机械臂抓取系统，用于试管阵列的自动识别与定位。
 项目简介

本项目实现了一个完整的工业视觉流程：

工业相机采集图像
视觉算法进行模板匹配与坐标计算
输出精确的抓取坐标
对接机械臂执行自动抓取

适用于：

* 实验室自动化
* 医疗试剂分拣
* 工业视觉定位
  
 功能特点

* 支持多种试管阵列（8x6 / 12x8）
* 基于模板图像进行定位
* 自动计算世界坐标
* 支持相机标定配置（JSON）
* 与机械臂控制系统对接
* 可扩展不同规格容器

---

##项目结构

```
pick-tube/
│── vision_robot_controller.py   # 主控制程序
│── UI.py                        # 可视化界面
│── CamOperation_class.py        # 相机操作封装
│── MvCameraControl_class.py     # 海康相机SDK封装
│── config/
│   ├── tube_config.json
│   ├── square_grid_config.json
│── templates/
│   ├── template_12X8.jpg
│   ├── template_8x6.jpg
│── README.md
```

---

## 环境依赖

建议使用 Python 3.8+

安装依赖：

```
pip install numpy opencv-python
```

注意：

* 需要安装工业相机 SDK（如海康相机）
* `.dll` 文件需放置在正确路径或使用官方安装包

---

## 使用方法

### 1️配置相机

修改参数文件：

```
CameraParams_const.py
CameraParams_header.py
```

---

###  配置模板与阵列

编辑：

```
tube_config.json
square_grid_config.json
```

---

###  运行程序

```
python vision_robot_controller.py
```

---

## 核心流程

1. 相机采集图像
2. 模板匹配定位阵列
3. 计算每个试管中心点
4. 转换为机械臂坐标
5. 输出抓取位置

---

## 技术要点

* OpenCV 图像处理
* 模板匹配
* 相机标定（像素 → 世界坐标）
* 工业相机 SDK 调用
* 坐标系转换

---

## 注意事项

* 相机需完成标定，否则坐标不准确
* 光照对识别效果影响较大
* 模板图像需与实际场景一致
* 建议使用英文文件名避免编码问题

---

## 后续优化方向

* [ ] 深度学习目标检测（替代模板匹配）
* [ ] 多相机支持
* [ ] 自动标定系统
* [ ] UI界面优化
* [ ] 抓取路径规划



