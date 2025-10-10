#!/usr/bin/env python3
import rospy
import math
import re
import numpy as np
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import String
import tf
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from nav_msgs.msg import Odometry

class YOLOToMoveBase:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('yolo_to_move_base', anonymous=True)
        
        # 创建发布者
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        
        # 创建订阅者
        rospy.Subscriber("/yolov8_set_param", String, self.yolo_callback)
        rospy.Subscriber("/mavros/local_position/odom", Odometry, self.odom_callback)
        
        # TF广播器
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        # YOLO数据变量
        self.center_x = 0.0
        self.center_y = 0.0
        self.frame_name = ""
        self.distance = 0.0
        self.yaw_angle = 0.0
        self.pitch_angle = 0.0
        self.box_x = 0.0
        self.box_y = 0.0
        self.box_z = 0.0
        
        # 目标在世界坐标系中的位置（相对于起飞原点）
        self.target_world_x = 0.0
        self.target_world_y = 0.0  
        self.target_world_z = 0.0
        
        # 序列号
        self.seq = 1
        
        # 飞机当前位姿（相对于起飞原点）
        self.drone_world_x = 0.0
        self.drone_world_y = 0.0
        self.drone_world_z = 0.0
        self.drone_yaw = 0.0
        
        # 存储不同目标的位置（相对于起飞原点）
        self.targets = {}  # key: 目标名称, value: (x, y, z)
        
        # 固定的偏移量
        self.x_offset = 0.1  # X轴固定偏移
        self.z_offset = 1.3  # Z轴固定偏移
        
        # 发布控制变量
        self.last_publish_time = rospy.Time.now()
        self.publish_delay = rospy.Duration(8)  # 8秒延迟
        self.pending_goal = None  # 等待发布的目标
        
        # 目标点计数器
        self.target_counter = 0
        
        # 标志位，表示是否已收到无人机位姿
        self.has_odom = False
        
        rospy.loginfo("YOLO to MoveBase 节点已启动，等待YOLO检测数据...")
        rospy.loginfo("所有目标坐标均相对于起飞原点（世界坐标系）")
        rospy.loginfo("目标点发布间隔: 8秒")

    def odom_callback(self, msg):
        """处理接收到的无人机位姿数据"""
        try:
            # 更新飞机位置（相对于起飞原点）
            self.drone_world_x = msg.pose.pose.position.x
            self.drone_world_y = msg.pose.pose.position.y
            self.drone_world_z = msg.pose.pose.position.z
            
            # 将四元数转换为欧拉角
            orientation = msg.pose.pose.orientation
            euler = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.drone_yaw = euler[2]
            
            # 设置标志位
            self.has_odom = True
            
        except Exception as e:
            rospy.logwarn(f"处理无人机位姿数据时出错: {e}")

    def parse_yolo_data(self, data_str):
        """解析YOLO数据字符串"""
        try:
            # 使用正则表达式提取键值对
            pattern = r'(\w+):([^,]+)'
            matches = re.findall(pattern, data_str)
            
            # 转换为字典
            data_dict = {}
            for key, value in matches:
                data_dict[key.strip()] = value.strip()
            
            return data_dict
        except Exception as e:
            rospy.logerr(f"解析YOLO数据时出错: {e}")
            return None

    def calculate_target_world_position(self, box_x, box_y, box_z):
        """
        计算目标在世界坐标系（起飞原点）中的绝对位置
        基于无人机当前位置和目标相对位置计算
        """
        try:
            # 检查是否已收到无人机位姿
            if not self.has_odom:
                rospy.logwarn("尚未收到无人机位姿数据，使用默认位姿进行坐标计算")
                return box_x, box_y, box_z
            
            # 核心计算：将相对坐标转换为世界坐标
            # box_x, box_y, box_z 是目标相对于无人机的坐标
            # 无人机在世界坐标系中的位置是 (drone_world_x, drone_world_y, drone_world_z)
            # 无人机偏航角是 drone_yaw
            
            yaw = self.drone_yaw
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            
            # 将目标相对坐标旋转到世界坐标系方向
            rotated_x = box_x * cos_yaw - box_y * sin_yaw
            rotated_y = box_x * sin_yaw + box_y * cos_yaw
            
            # 加上无人机在世界坐标系中的位置，得到目标的绝对世界坐标
            world_x = self.drone_world_x + rotated_x
            world_y = self.drone_world_y + rotated_y
            world_z = self.drone_world_z - box_z
            
            rospy.loginfo(f"=== 坐标计算 ===")
            rospy.loginfo(f"无人机世界坐标: ({self.drone_world_x:.2f}, {self.drone_world_y:.2f}, {self.drone_world_z:.2f})")
            rospy.loginfo(f"无人机偏航角: {math.degrees(yaw):.2f}°")
            rospy.loginfo(f"目标相对坐标: ({box_x:.2f}, {box_y:.2f}, {box_z:.2f})")
            rospy.loginfo(f"旋转后坐标: ({rotated_x:.2f}, {rotated_y:.2f}, {-box_z:.2f})")
            rospy.loginfo(f"目标世界坐标: ({world_x:.2f}, {world_y:.2f}, {world_z:.2f})")
            rospy.loginfo(f"=================")
            
            return world_x, world_y, world_z
            
        except Exception as e:
            rospy.logerr(f"坐标计算错误: {e}")
            return box_x, box_y, box_z

    def broadcast_target_tf(self):
        """广播目标点的TF坐标系"""
        try:
            # 获取当前时间
            current_time = rospy.Time.now()
            
            # 广播目标点坐标系（原始位置）
            self.tf_broadcaster.sendTransform(
                (self.target_world_x, self.target_world_y, self.target_world_z),
                (0, 0, 0, 1),  # 单位四元数，无旋转
                current_time,
                "target_original",
                "world"
            )
            
            # 广播最终目标点坐标系（带偏移的位置）
            final_x = self.target_world_x + self.x_offset
            final_y = self.target_world_y
            final_z = self.target_world_z + self.z_offset
            
            self.tf_broadcaster.sendTransform(
                (final_x, final_y, final_z),
                (0, 0, 0, 1),  # 单位四元数，无旋转
                current_time,
                "target_final",
                "world"
            )
            
            # 广播无人机当前姿态
            if self.has_odom:
                quaternion = quaternion_from_euler(0, 0, self.drone_yaw)
                self.tf_broadcaster.sendTransform(
                    (self.drone_world_x, self.drone_world_y, self.drone_world_z),
                    quaternion,
                    current_time,
                    "drone_current",
                    "world"
                )
            
            # 广播所有历史目标点
            for i, (target_name, (x, y, z)) in enumerate(self.targets.items()):
                self.tf_broadcaster.sendTransform(
                    (x, y, z),
                    (0, 0, 0, 1),
                    current_time,
                    f"target_{i}_{target_name}",
                    "world"
                )
            
        except Exception as e:
            rospy.logwarn(f"广播TF时出错: {e}")

    def yolo_callback(self, msg):
        """处理接收到的YOLO数据并发布目标点"""
        rospy.loginfo(f"收到YOLO数据: {msg.data}")
        
        # 解析数据
        data_dict = self.parse_yolo_data(msg.data)
        
        if data_dict:
            try:
                # 更新变量值
                self.center_x = float(data_dict.get('center_x', 0))
                self.center_y = float(data_dict.get('center_y', 0))
                self.frame_name = data_dict.get('frame_name', '')
                self.distance = float(data_dict.get('distance', 0))
                self.yaw_angle = float(data_dict.get('yaw_angle', 0))
                self.pitch_angle = float(data_dict.get('pitch_angle', 0))
                self.box_x = float(data_dict.get('box_x', 0))
                self.box_y = float(data_dict.get('box_y', 0))
                self.box_z = float(data_dict.get('box_z', 0))
                
                # 计算目标在世界坐标系中的绝对位置
                world_x, world_y, world_z = self.calculate_target_world_position(
                    self.box_x, self.box_y, self.box_z
                )
                
                # 更新目标在世界坐标系中的位置
                self.target_world_x = world_x
                self.target_world_y = world_y
                self.target_world_z = world_z
                
                # 存储不同目标的位置
                self.targets[self.frame_name] = (world_x, world_y, world_z)
                
                # 打印存储的数据
                self.print_stored_data()
                
                # 广播TF坐标系
                self.broadcast_target_tf()
                
                # 检查是否可以立即发布目标点
                current_time = rospy.Time.now()
                time_since_last_publish = current_time - self.last_publish_time
                
                if time_since_last_publish >= self.publish_delay:
                    # 可以直接发布
                    self.publish_goal()
                    self.last_publish_time = current_time
                else:
                    # 需要等待，设置定时器
                    remaining_delay = self.publish_delay - time_since_last_publish
                    rospy.loginfo(f"距离上次发布仅 {time_since_last_publish.to_sec():.1f} 秒，等待 {remaining_delay.to_sec():.1f} 秒后发布目标点")
                    
                    # 取消之前的定时器（如果有）
                    if hasattr(self, 'publish_timer') and self.publish_timer:
                        self.publish_timer.shutdown()
                    
                    # 设置新的定时器
                    self.publish_timer = rospy.Timer(remaining_delay, self.delayed_publish, oneshot=True)
                    self.pending_goal = True
                
            except ValueError as e:
                rospy.logerr(f"数据类型转换错误: {e}")
            except Exception as e:
                rospy.logerr(f"处理YOLO数据时出错: {e}")

    def delayed_publish(self, event):
        """定时器回调函数，延迟发布目标点"""
        rospy.loginfo("延迟结束，发布目标点")
        self.publish_goal()
        self.last_publish_time = rospy.Time.now()
        self.pending_goal = False

    def print_stored_data(self):
        """打印当前存储的数据"""
        rospy.loginfo("=== 当前存储的YOLO数据 ===")
        rospy.loginfo(f"目标类别: {self.frame_name}")
        rospy.loginfo(f"距离: {self.distance:.2f} 米")
        rospy.loginfo(f"偏航角: {self.yaw_angle:.2f} 度")
        rospy.loginfo(f"俯仰角: {self.pitch_angle:.2f} 度")
        rospy.loginfo(f"相对位置 X: {self.box_x:.2f} 米")
        rospy.loginfo(f"相对位置 Y: {self.box_y:.2f} 米")
        rospy.loginfo(f"相对位置 Z: {self.box_z:.2f} 米")
        rospy.loginfo(f"世界坐标系位置 X: {self.target_world_x:.2f} 米")
        rospy.loginfo(f"世界坐标系位置 Y: {self.target_world_y:.2f} 米")
        rospy.loginfo(f"世界坐标系位置 Z: {self.target_world_z:.2f} 米")
        rospy.loginfo(f"无人机位置: ({self.drone_world_x:.2f}, {self.drone_world_y:.2f}, {self.drone_world_z:.2f})")
        rospy.loginfo(f"无人机偏航角: {math.degrees(self.drone_yaw):.2f}°")
        
        # 打印所有已检测目标的位置
        rospy.loginfo("已检测目标位置（世界坐标系）:")
        for target_name, (x, y, z) in self.targets.items():
            rospy.loginfo(f"  {target_name}: ({x:.2f}, {y:.2f}, {z:.2f})")
        
        rospy.loginfo("==========================")

    def publish_goal(self):
        """发布目标点到move_base（在世界坐标系下）"""
        # 创建目标消息
        goal_msg = PoseStamped()
        
        # 设置消息头
        goal_msg.header.seq = self.seq
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "world"  # 使用世界坐标系
        
        # 设置目标位置 - 使用计算得到的世界坐标并添加固定偏移
        goal_msg.pose.position.x = self.target_world_x + self.x_offset
        goal_msg.pose.position.y = self.target_world_y  
        goal_msg.pose.position.z = self.target_world_z + self.z_offset
        
        # 设置目标朝向 - 保持当前无人机朝向
        quaternion = quaternion_from_euler(0, 0, self.drone_yaw)
        goal_msg.pose.orientation.x = quaternion[0]
        goal_msg.pose.orientation.y = quaternion[1]
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]
        
        # 发布目标点
        self.goal_pub.publish(goal_msg)
        
        # 日志输出
        rospy.loginfo(f"已发布目标 '{self.frame_name}' 到 /move_base_simple/goal")
        rospy.loginfo(f"  目标世界坐标: (x: {self.target_world_x:.2f}, y: {self.target_world_y:.2f}, z: {self.target_world_z:.2f})")
        rospy.loginfo(f"  固定偏移: (x: +{self.x_offset:.2f}, z: +{self.z_offset:.2f})")
        rospy.loginfo(f"  最终目标坐标: (x: {goal_msg.pose.position.x:.2f}, y: {goal_msg.pose.position.y:.2f}, z: {goal_msg.pose.position.z:.2f})")
        rospy.loginfo(f"  目标偏航角: {math.degrees(self.drone_yaw):.2f}°")
        
        # 序列号自增
        self.seq += 1
        self.target_counter += 1

    def run(self):
        """运行节点"""
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            try:
                # 持续广播TF坐标系
                self.broadcast_target_tf()
                
                rate.sleep()
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                rospy.logerr(f"主循环错误: {e}")
                rate.sleep()

if __name__ == '__main__':
    try:
        node = YOLOToMoveBase()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("YOLO to MoveBase 节点已关闭")
