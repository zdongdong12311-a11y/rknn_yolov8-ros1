# coding=utf-8
# ------------------------------------------------------------
# rknn_v8_simplified.py — YOLOv8 RKNN 实时检测简化版
# ------------------------------------------------------------
import os, sys, cv2, time, queue, threading
import numpy as np
from rknnlite.api import RKNNLite
from collections import deque
import rospy
import logging
from std_msgs.msg import String
import os
import math

# 确保使用正确的配置文件
os.environ['ROSCONSOLE_CONFIG_FILE'] = '/opt/ros/noetic/share/ros/config/rosconsole.config'

# 手动定义缺失的日志级别
if 'DEBUG' not in logging._nameToLevel:
    logging.addLevelName(logging.DEBUG, 'DEBUG')
if 'INFO' not in logging._nameToLevel:
    logging.addLevelName(logging.INFO, 'INFO')

# ================ 全局参数 ================
MODEL_PATH = '/home/orangepi/Desktop/yolo_move/rknn3588/best.rknn'
IMG_SIZE = (640, 640)  # 推理输入分辨率
OBJ_THRESH = 0.60  # 置信度阈值
NMS_THRESH = 0.5  # IoU-NMS 阈值
Flags = True
CLASSES = ("circle", "rectangle")  # 类别名

# ================ 相机参数（需要标定） ================
CAMERA_MATRIX = np.array([
    [1285, 0, 960],   # fx, 0, cx (假设1920x1080分辨率，cx=960, cy=540)
    [0, 1285, 540],   # 0, fy, cy
    [0, 0, 1]         # 0, 0, 1
])


# ================ 后处理工具函数 ================
def dfl(pos):
    """Distribution Focal Loss (DFL) 解码 —— 纯 NumPy 实现"""
    n, c, h, w = pos.shape
    pos = pos.reshape(n, 4, c // 4, h, w)
    softmax = np.exp(pos) / np.sum(np.exp(pos), axis=2, keepdims=True)
    acc = np.arange(c // 4, dtype=np.float32).reshape(1, 1, -1, 1, 1)
    return np.sum(softmax * acc, axis=2)


def box_process(pos):
    """DFL 偏移量 → xyxy 絶对坐标 (输入分辨率)"""
    gh, gw = pos.shape[2:4]
    col, row = np.meshgrid(np.arange(gw), np.arange(gh))
    grid = np.stack((col, row), 0).reshape(1, 2, gh, gw)
    stride = np.array([IMG_SIZE[1] // gw, IMG_SIZE[0] // gh]).reshape(1, 2, 1, 1)
    pos = dfl(pos)
    xy1 = grid + 0.5 - pos[:, :2]
    xy2 = grid + 0.5 + pos[:, 2:4]
    return np.concatenate((xy1 * stride, xy2 * stride), 1)


def nms_xyxy(boxes, scores):
    """纯 NumPy IoU-NMS，返回保留索引"""
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[order[1:]] - inter)
        order = order[1 + np.where(iou <= NMS_THRESH)[0]]
    return keep


def filter_and_nms(boxes, cconf, oconf):
    """置信度过滤 + 按类 NMS"""
    cls_max = cconf.max(-1)
    cls_ids = cconf.argmax(-1)
    mask = cls_max * oconf.reshape(-1) >= OBJ_THRESH
    if not mask.any():
        return None, None, None
    boxes, scores, cls_ids = boxes[mask], (cls_max * oconf.reshape(-1))[mask], cls_ids[mask]
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        idx = np.where(cls_ids == cid)[0]
        keep = nms_xyxy(boxes[idx], scores[idx])
        fb.append(boxes[idx][keep])
        fs.append(scores[idx][keep])
        fc.append(np.full(len(keep), cid))
    return np.concatenate(fb), np.concatenate(fc), np.concatenate(fs)


def letter_box(img, new_shape=IMG_SIZE):
    """保持比例缩放 + 灰边填充"""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape[1] - nw) // 2, (new_shape[0] - nh) // 2
    img = cv2.resize(img, (nw, nh), cv2.INTER_LINEAR)
    canvas = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = img
    return canvas, r, (dw, dh)


def scale_boxes(boxes, src_shape, dw, dh, r):
    """把推理坐标映射回原分辨率"""
    b = boxes.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - dw) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dh) / r
    h, w = src_shape
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
    return b


# ================ 距离和角度计算函数 (相对于飞机的)================
def distance_ws(length):
    """保持原有的距离计算函数不变"""
    k = 4000 / 2.86
    distance = k / length
    return distance


def calculate_angles(center_x, center_y, distance):
    """
    计算目标相对于摄像头光轴的角度
    返回：水平角(偏航角), 垂直角(俯仰角) 单位：度
    """
    fx = CAMERA_MATRIX[0, 0]
    fy = CAMERA_MATRIX[1, 1]
    cx = CAMERA_MATRIX[0, 2]
    cy = CAMERA_MATRIX[1, 2]
    
    # 计算归一化坐标（相对于光轴）
    x_normalized = (center_x - cx) / fx
    y_normalized = (center_y - cy) / fy
    
    # 计算角度（弧度）
    yaw_angle = math.atan(x_normalized)    # 水平偏航角
    pitch_angle = -math.atan(y_normalized)   # 垂直俯仰角
    
    # 转换为度
    yaw_deg = math.degrees(yaw_angle)
    pitch_deg = math.degrees(pitch_angle)
    
    return yaw_deg, pitch_deg


def calculate_box_center_position(distance, yaw_deg, pitch_deg):
    """
    计算框中心点在无人机坐标系中的位置
    坐标系：前X轴，左Y轴，上Z轴
    相机安装在无人机中心上方0.04米处
    """
    # 将角度转换为弧度
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)
    
    # 计算框中心点在无人机坐标系中的位置
    # 注意：这里计算的是从无人机到框中心的向量
    box_x = distance * math.cos(pitch_rad) * math.cos(yaw_rad)  # 前方距离
    box_y = -distance * math.cos(pitch_rad) * math.sin(yaw_rad)  # 左方距离
    box_z = distance * math.sin(pitch_rad) + 0.04               # 上方距离 + 相机安装高度
    
    return box_x, box_y, box_z


# ================ 推理线程 ================
class InferenceWorker(threading.Thread):
    CORE_MAP = {0: RKNNLite.NPU_CORE_0,
                1: RKNNLite.NPU_CORE_1,
                2: RKNNLite.NPU_CORE_2}

    def __init__(self, idx, model_path, in_q, out_q):
        super().__init__(daemon=True)
        self.in_q, self.out_q = in_q, out_q
        self.rknn = RKNNLite(verbose=False)
        assert self.rknn.load_rknn(model_path) == 0
        assert self.rknn.init_runtime(core_mask=self.CORE_MAP[idx]) == 0
        print(f'[Worker-{idx}] init OK')

    def run(self):
        while True:
            fid, frame = self.in_q.get()
            if fid is None:
                break
            h0, w0 = frame.shape[:2]
            img, r, (dw, dh) = letter_box(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out = self.rknn.inference([np.expand_dims(img, 0)])

            branch, pair = 3, len(out) // 3
            boxes, cconfs, oconfs = [], [], []
            for i in range(branch):
                boxes.append(box_process(out[pair * i]))
                cconfs.append(out[pair * i + 1])
                oconfs.append(np.ones_like(out[pair * i + 1][:, :1, :, :], np.float32))
            merge = lambda xs: np.concatenate([x.transpose(0, 2, 3, 1).reshape(-1, x.shape[1]) for x in xs])
            b, cls, s = filter_and_nms(merge(boxes), merge(cconfs), merge(oconfs))
            if b is not None:
                b = scale_boxes(b, (h0, w0), dw, dh, r)
            self.out_q.put((fid, b, cls, s))
        self.rknn.release()


def main():
    frame_num = 0
    rospy.init_node("yolov8_detection_node")
    
    # 创建发布器，发布到 /yolov8_set_param 话题
    param_pub = rospy.Publisher("/yolov8_set_param", String, queue_size=10)

    in_qs = [queue.Queue(6) for _ in range(3)]
    out_q = queue.Queue(12)
    workers = [InferenceWorker(i, MODEL_PATH, in_qs[i], out_q) for i in range(3)]
    for w in workers:
        w.start()

    # 初始化摄像头变量
    cap = None
    camera_reconnect_count = 0
    MAX_RECONNECT = 5

    cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Detection', 1920, 1080)

    fid, t0 = 0, time.time()
    PERSIST_N = 7
    history = deque(maxlen=PERSIST_N)
    
    # 新增：类别跟踪列表
    class_track_list = [None]  # 初始化为[None]

    try:
        while True:
            # 摄像头重连逻辑
            if cap is None or not cap.isOpened():
                if camera_reconnect_count >= MAX_RECONNECT:
                    print('摄像头重连次数过多，退出程序')
                    break

                print('尝试连接摄像头...')
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    print('摄像头连接成功')
                    camera_reconnect_count = 0
                else:
                    print(f'摄像头连接失败，{camera_reconnect_count + 1}/{MAX_RECONNECT}')
                    camera_reconnect_count += 1
                    time.sleep(2)
                    continue

            # 读取帧
            try:
                ok, frame = cap.read()
                if not ok:
                    print('读取帧失败，尝试重新连接摄像头')
                    cap.release()
                    cap = None
                    camera_reconnect_count += 1
                    continue

                # 重置重连计数器
                camera_reconnect_count = 0

            except Exception as e:
                print(f'读取帧时发生错误: {e}')
                if cap is not None:
                    cap.release()
                    cap = None
                continue

            # -------- 推理输入分发 --------
            target = fid % 3
            in_qs[target].put((fid, frame.copy()))
            fid += 1

            # -------- 取回推理结果 --------
            has_det = False
            while not out_q.empty():
                has_det = True
                _, boxes, cls_ids, scores = out_q.get()
                history.append((boxes, cls_ids, scores))

            # 若本帧没有新结果→沿用最近结果
            if not has_det and history:
                boxes, cls_ids, scores = history[-1]
            elif not history:  # 还没有任何推理输出
                boxes, cls_ids, scores = None, None, None

            # 绘制检测结果 - 在整个图像范围内检测最近的目标
            if boxes is not None:
                frame_num = 0
                min_distance = float('inf')
                closest_target = None

                # 在整个图像范围内寻找距离最近的目标
                for box, cls, sc in zip(boxes, cls_ids, scores):
                    x1, y1, x2, y2 = box.astype(int)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # 使用原有的距离计算函数
                    distance = distance_ws(x2 - x1)

                    # 在整个图像范围内检测
                    in_roi = True

                    if in_roi and distance < min_distance:
                        min_distance = distance
                        closest_target = (box, cls, sc, center_x, center_y, distance)
                        frame_num = 1  # 只统计最近的一个

                # 绘制最近的目标
                if closest_target is not None:
                    box, cls, sc, center_x, center_y, distance = closest_target
                    x1, y1, x2, y2 = box.astype(int)

                    # 计算角度和框中心位置
                    yaw_deg, pitch_deg = calculate_angles(center_x, center_y, distance)
                    box_x, box_y, box_z = calculate_box_center_position(distance, yaw_deg, pitch_deg)

                    # 绘制检测框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 绘制标签 - 显示更多信息
                    label1 = f'{CLASSES[cls]} {sc:.2f} dist:{distance:.1f}'
                    label2 = f'Yaw:{yaw_deg:.1f}° Pitch:{pitch_deg:.1f}°'
                    label3 = f'BoxPos-X:{box_x:.1f} BoxPos-Y:{box_y:.1f} BoxPos-Z:{box_z:.1f}'
                    
                    cv2.putText(frame, label1, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, label2, (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame, label3, (x1, y1 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 绘制中心点
                    cv2.circle(frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                    
                    # 绘制角度指示线
                    center_screen_x = CAMERA_MATRIX[0, 2]  # 图像中心X
                    center_screen_y = CAMERA_MATRIX[1, 2]  # 图像中心Y
                    cv2.line(frame, (int(center_screen_x), int(center_screen_y)), 
                            (int(center_x), int(center_y)), (255, 255, 0), 2)

                    # 新增：类别变化检测逻辑
                    current_class = CLASSES[cls]
                    if current_class != class_track_list[0]:
                        # 打印信息并发布到话题
                        print(f"目标：{current_class}，距离：{distance:.2f}，偏航角：{yaw_deg:.1f}°，俯仰角：{pitch_deg:.1f}°")
                        print(f"框中心在无人机坐标系中的位置：X={box_x:.2f}, Y={box_y:.2f}, Z={box_z:.2f}")
                        
                        # 将所有参数打包成字符串发布到 /yolov8_set_param 话题
                        param_msg = String()
                        param_msg.data = f"center_x:{center_x:.2f},center_y:{center_y:.2f},frame_name:{current_class},distance:{distance:.2f},yaw_angle:{yaw_deg:.2f},pitch_angle:{pitch_deg:.2f},box_x:{box_x:.2f},box_y:{box_y:.2f},box_z:{box_z:.2f}"
                        param_pub.publish(param_msg)
                        
                        # 更新类别列表
                        class_track_list[0] = current_class
                    # 如果类别相同，不执行任何操作（不打印也不发布）
                else:
                    frame_num = 0
                    print("未检测到目标")
            else:
                frame_num = 0

            # 显示FPS
            fps = 1 / (time.time() - t0)
            t0 = time.time()
            cv2.putText(frame, f'FPS:{fps:.2f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 显示画面
            if Flags:
                cv2.imshow('Detection', frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        for q in in_qs:
            q.put((None, None))
        for w in workers:
            w.join()


if __name__ == '__main__':
    main()
