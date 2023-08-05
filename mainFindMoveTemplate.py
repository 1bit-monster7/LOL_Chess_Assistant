import configparser
import ctypes
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
from ctypes import windll
from datetime import datetime

import cv2
import numpy as np
import pyautogui
import win32api
import win32con
import win32gui
import win32ui
import winsound
from PyQt5.QtCore import Qt, QSize, QObject, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QButtonGroup, QRadioButton, QLayout, QLineEdit
from PyQt5.QtWidgets import QWidget, QGridLayout, QScrollArea, QFrame
from pynput import keyboard, mouse

is_debug = False
is_grayscale = False  # 默认不开启
is_show_not_find_window = True
window_size = None

logging.disable(logging.DEBUG)  # 关闭DEBUG日志的打印
logging.disable(logging.WARNING)  # 关闭WARNING日志的打印

money_range = (813, 877, 131, 37)  # 金币所在位置

rect = (480, 1030, 1000, 45)  # 英雄所在区域

region = (482, 930, 1000, 110)

cap_win_name = "League of Legends (TM) Client"
lol_hw = 'RiotWindowClass'
cv2_win_name = "win"
top_window_width = 300
# 调整图片尺寸
folder_path = "images/624"  # 替换为包含图片的文件夹路径
width = 190
height = 108
# 请求下来的图存储路径
save_p = "images/320"
# 读取的文件夹路径
rw_path = "images/120"
config = configparser.ConfigParser()
ini_file = '1bit.ini'
ocr = None
lol_hwnd = win32gui.FindWindow('RiotWindowClass', None)

# 注册窗口类
wc = win32gui.WNDCLASS()
wc.hCursor = win32gui.LoadCursor(0, win32con.IDC_ARROW)
wc.hbrBackground = win32con.COLOR_WINDOW
wc.lpszClassName = 'PythonWindowClass'
wc.lpfnWndProc = lambda hwnd, msg, wParam, lParam: 0
win32gui.RegisterClass(wc)

# 加载user32.dll
user32 = ctypes.WinDLL("user32.dll")


def get_files():
    images_files = os.path.join(os.getcwd(), rw_path)
    files = os.listdir(images_files)
    images = []
    for f in files:
        if f.endswith('.png') or f.endswith('.jpg'):
            path = os.path.join(rw_path, f)  # 在文件名前面拼接路径
            decoded_file_name = decode_unicode_escape(path.encode('raw_unicode_escape').decode("utf-8"))
            images.append(decoded_file_name)

    # 按价格从小到大排序
    images = sorted(images, key=lambda x: extract_filename(x)[1])

    # 创建费用分类字典
    card_dict = {1: [], 2: [], 3: [], 4: [], 5: []}

    # 将卡片按费用分类
    for img in images:
        name, price = extract_filename(img)
        card_dict[price].append(img)
    return card_dict


def grab_gpt_win(grab_rect=None, toColor=True):
    hwnd = win32gui.GetDesktopWindow()
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()
    screen_width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
    x, y, w, h = grab_rect or (0, 0, screen_width, screen_height)
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
    saveDC.SelectObject(saveBitMap)

    saveDC.BitBlt((0, 0), (w, h), mfcDC, (x, y), win32con.SRCCOPY)

    signed_ints_array = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(signed_ints_array, dtype="uint8")
    img.shape = (h, w, 4)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)
    if toColor:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def extract_filename(path):
    # 去除文件夹路径
    filename = path.split("/")[-1]
    # 去除尾部的文件夹路径
    filename = filename.rsplit("\\", 1)[-1]
    # 去除后缀名
    filename = filename.split(".")[0]
    # 使用正则表达式匹配数字并提取
    match = re.search(r'\d+', filename)
    price = match.group() if match else ''
    if price == '':
        price = '1'  # 如果价格为空字符串，则默认设置为1
    # 切割字符串
    name = filename.replace(price, '')  # 去除数字部分的前缀
    # 返回英雄名和价格
    return name, int(price)


def filter_list(lst, num):
    lst.sort()  # 排序列表
    result = []  # 存储结果的列表
    i = 0
    while i < len(lst):
        current = lst[i]
        result.append(current)  # 将当前元素添加到结果列表中
        j = i + 1
        while j < len(lst) and lst[j][0] - current[0] <= num:
            j += 1
        i = j

    return result


def decode_unicode_escape(string):
    pattern = re.compile(r'\\u([\da-fA-F]{4})')
    result = re.sub(pattern, lambda x: chr(int(x.group(1), 16)), string)
    return result


class ImageListWindow(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window
        self.labels = {}  # 存储image_label的字典

        # 设置窗口为透明,无边框
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 使用网格布局
        self.main_layout = QGridLayout()
        self.main_layout.setSpacing(0)  # 设置间距
        self.main_layout.setAlignment(Qt.AlignTop)  # 顶部对齐

        count = 0
        for row in range(len(self.main_window.active_list) // 5 + 1):
            for col in range(5):
                index = row * 5 + col
                if index >= len(self.main_window.active_list):
                    break

                name = self.main_window.active_list[index]
                label = QLabel()
                label.setFixedSize(QSize(30, 30))
                pixmap = QPixmap(f"images/label/{name}.png")  # 调整图片大小
                label.setPixmap(pixmap)
                # 绑定双击事件
                label.mouseDoubleClickEvent = self.create_double_click_event(name)

                self.main_layout.addWidget(label, row, col)
                self.labels[name] = label  # 提前存入

                count += 1

        # 更新布局
        self.setLayout(self.main_layout)
        # 移动到左上角
        self.move(-10, -10)
        # 显示窗口
        self.show()

    def create_double_click_event(self, name):
        def mouseDoubleClickEvent(event):
            if name in self.main_window.active_list:
                self.main_window.active_list.remove(name)
                self.main_window.remove_hero(name)
                # 重新载入图片列表
                self.reload_image_list()

        return mouseDoubleClickEvent

    def reload_image_list(self):
        self.labels = {}  # 存储image_label的字典
        # 清空原有布局
        while self.main_layout.count():
            item = self.main_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        count = 0
        for row in range(len(self.main_window.active_list) // 5 + 1):
            for col in range(5):
                index = row * 5 + col
                if index >= len(self.main_window.active_list):
                    break

                name = self.main_window.active_list[index]
                label = QLabel()
                label.setFixedSize(QSize(30, 30))
                pixmap = QPixmap(f"images/label/{name}.png")  # 调整图片大小
                label.setPixmap(pixmap)
                # 绑定双击事件
                label.mouseDoubleClickEvent = self.create_double_click_event(name)

                self.main_layout.addWidget(label, row, col)
                self.labels[name] = label  # 提前存入

                count += 1
        # 更新布局
        self.main_layout.setSizeConstraint(QLayout.SetFixedSize)  # 设置布局大小固定
        self.main_layout.update()


class UpdateUISignal(QObject):
    signal = pyqtSignal(str)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # ui图片列表
        self.update_ui_signal = UpdateUISignal()
        self.update_ui_signal.signal.connect(self.update_ui_slot)
        self.rect_items = None
        self.flag = None
        self.dict_list = ALL_IMAGE_LIST
        # 是否开启
        self.is_open = True
        # 添加这个属性
        self.is_running = True
        # 配置文件名
        self.ini_file = '1bit.ini'
        # 存储图片选中状态的字典
        self.selected_images = {}
        # 选中列表
        self.active_list = []
        # 读取配置文件
        self.load_ini_file()
        # 加载样式
        self.load_styles()
        # 设置ico
        self.setWindowIcon(QIcon('1.ico'))
        self.image_labels = {}  # 存储image_label的字典
        self.frames = {}  # 存储frame的字典
        # 创建并启动键盘监听线程
        self.keyboard_thread = threading.Thread(target=self.keyboard_listener)
        self.keyboard_thread.start()
        self.hwnd = win32gui.FindWindow(lol_hw, None)
        print(self.hwnd, 'LOL')
        # 如果debug模式开启 则另启动线程
        # 创建图片列表窗口
        self.imClass = ImageListWindow(self)
        # 创建单行文本框
        self.text_box = QLineEdit()
        self.text_box.setPlaceholderText("")  # 设置占位文本
        self.text_box.setStyleSheet("font-weight:bold; color: green; font-size: 26px;  text-align: center;")
        self.text_box.setAlignment(Qt.AlignCenter)
        # 将单行文本框添加到布局中
        if is_debug:
            threading.Thread(target=self.debug_fun).start()
        # 创建网格布局
        layout = QGridLayout()
        layout.setSpacing(10)  # 设置间距
        layout.addWidget(self.text_box, 0, 0, 1, -1)  # 将文本框添加到布局中并跨越整行
        content_widget = QWidget()  # 创建内容部件
        content_widget.setLayout(layout)  # 将布局设置给内容部件
        # 添加到布局
        # 添加图片到布局中
        row = 1
        col = 0

        for fee, zone in self.dict_list.items():
            fee_label = QLabel(f"{fee}费区:")
            fee_label.setStyleSheet("""
                QLabel {
                    color: green;
                    font-size:36px;
                }
            """)
            fee_label.setAlignment(Qt.AlignLeft)
            layout.addWidget(fee_label, row, col, 1, 1)
            for i, file in enumerate(zone):
                name, price = extract_filename(file)
                image_frame = QFrame()
                image_label = QLabel(image_frame)
                self.selected_images[name] = False  # 初始化选中状态
                image_frame.setStyleSheet(
                    """ 
                      QFrame:hover{
                            border: 1px solid red;
                      }
                       QFrame > QLabel {
                            border: 0;  /* 移除内部QLabel的边框样式 */
                      }
                       QFrame > QLabel:hover{
                            border: 0;  /* 移除内部QLabel的边框样式 */
                       }      
                    """)
                if name in self.active_list:
                    self.active_styles(image_frame, image_label)  # 初始化选中样式
                    self.selected_images[name] = True  # 初始化选中状态

                image_label.setPixmap(QPixmap(file))

                name_label = QLabel(f"英雄名称：{name} 费用：{price}")

                name_label.setStyleSheet(
                    """
                       QLabel {
                           color: white;
                           font-size:18px;
                       }
                   """)

                image_frame.mouseDoubleClickEvent = self.create_double_click_event(image_frame, fee, i, image_label)

                # 将图像和名称标签的对齐方式设置为居中对齐
                image_label.setAlignment(Qt.AlignHCenter)
                name_label.setAlignment(Qt.AlignHCenter)

                frame_layout = QVBoxLayout()
                frame_layout.addWidget(image_label)
                frame_layout.addWidget(name_label)

                # 将帧布局的对齐方式设置为居中对齐
                frame_layout.setAlignment(Qt.AlignHCenter)

                image_frame.setLayout(frame_layout)

                item_row = row + (i // 4) + 1  # 计算当前图片应该在布局中的行索引
                item_col = col + (i % 4)  # 计算当前图片应该在布局中的列索引

                # 添加到字典中
                self.image_labels[name] = image_label
                self.frames[name] = image_frame

                layout.addWidget(image_frame, item_row, item_col)

            last_row_items = len(zone) % 4  # 最后一行的图片数量
            if last_row_items != 4:  # 如果最后一行不足4个图片
                empty_cols = 4 - last_row_items
                empty_frame = QFrame()
                empty_label = QLabel(empty_frame)
                empty_label.setStyleSheet("background-color: rgba(0, 0, 0, 0)")  # 设置背景透明
                layout.addWidget(empty_frame, row + (len(zone) // 4) + 1, col + last_row_items, 1, empty_cols)

            row += (len(zone) // 4) + 2  # 调整行号，为下一个费区预留空白行

        # 创建滚动区域，并将内容部件放入其中
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(content_widget)

        # 设置窗口最小宽度
        self.setMinimumWidth(1200)
        self.setMinimumHeight(600)

        # 将滚动区域设置为主窗口的布局
        self.setLayout(QGridLayout())
        self.layout().addWidget(scroll_area)

        threading.Thread(target=self.find_image_hero).start()

    def py_nms(self, dets, thresh):
        """Pure Python NMS baseline."""
        # x1、y1、x2、y2、以及score赋值
        # （x1、y1）（x2、y2）为box的左上和右下角标
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        # 每一个候选框的面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        # order是按照score降序排序的
        order = scores.argsort()[::-1]
        # print("order:",order)
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # 找到重叠度不高于阈值的矩形框索引
            inds = np.where(ovr <= thresh)[0]
            # print("inds:",inds)
            # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
            order = order[inds + 1]
        return keep

    def template(self, img_gray, template_img, template_threshold):
        '''
        img_gray:待检测的灰度图片格式
        template_img:模板小图，也是灰度化了
        template_threshold:模板匹配的置信度
        '''
        h, w = template_img.shape[:2]
        res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
        start_time = time.time()
        loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标
        score = res[res >= template_threshold]  # 大于模板阈值的目标置信度
        # 将模板数据坐标进行处理成左上角、右下角的格式
        xmin = np.array(loc[1])
        ymin = np.array(loc[0])
        xmax = xmin + w
        ymax = ymin + h
        xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
        xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
        ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
        ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
        score = score.reshape(-1, 1)  # 变成n行1列维度
        data_hlist = []
        data_hlist.append(xmin)
        data_hlist.append(ymin)
        data_hlist.append(xmax)
        data_hlist.append(ymax)
        data_hlist.append(score)
        data_hstack = np.hstack(data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接
        thresh = 0.3  # NMS里面的IOU交互比阈值
        keep_dets = self.py_nms(data_hstack, thresh)
        # print("nms time:", time.time() - start_time)  # 打印数据处理到nms运行时间
        dets = data_hstack[keep_dets]  # 最终的nms获得的矩形框
        return dets

    def find_node_by_name(self, name):
        image_label = self.image_labels.get(name)
        frame = self.frames.get(name)
        return image_label, frame

    def find_image_hero(self):
        while self.is_running:
            # 在适当的时机执行绘制操作
            if self.is_open:
                t1 = time.time()
                self.rect_items = []  # 存储所有矩形坐标的列表
                for name in self.active_list:
                    path = os.path.join("images/hero/", f"{name}.png")
                    target_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
                    big_image = grab_gpt_win(region, False)
                    img_gray = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)  # 转化成灰色
                    tg_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)  # 转化成灰色
                    template_threshold = 0.7  # 模板置信度
                    dets = self.template(img_gray, tg_gray, template_threshold)
                    for det in dets:
                        x1, y1, x2, y2, confidence = det
                        region_x = region[0]  # 截图区域的起始横坐标
                        region_y = region[1]  # 截图区域的起始纵坐标
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        absolute_x = region_x + center_x
                        absolute_y = region_y + center_y
                        self.mouse_click(absolute_x, absolute_y, 2)
                    # 目标图片
                current_time = datetime.now()
                # 将日期时间格式化为字符串（例如：2023-08-01 13:05:41）
                formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
                log_message = f"寻找卡片数量:{len(self.active_list)} 性能:{round(time.time() - t1, 2)}秒 {formatted_time}"
                self.update_ui_signal.signal.emit(log_message)
            time.sleep(0.001)

    def update_ui_slot(self, message):
        # 在槽函数中更新UI界面
        self.text_box.setText(message)

    def on_click(self, x, y, button, pressed):
        if not pressed:
            if button == mouse.Button.x1:
                self.is_open = True
                winsound.PlaySound('music/8855.wav', flags=1)
                print("功能：开")
            if button == mouse.Button.x2:
                winsound.PlaySound('music/close.wav', flags=1)
                self.is_open = False
                print("功能：关")

    def mouse_listener(self):
        listener = mouse.Listener(on_click=self.on_click)
        listener.start()

    @staticmethod
    def mouse_click(x, y, count=1):
        win32api.SetCursorPos((x, y))
        for i in range(count):
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)  # 鼠标左键按下
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)  # 鼠标左键释放

    @staticmethod
    def active_styles(dom, d2):
        dom.setStyleSheet(
            """
            QFrame {
                background: rgb(61, 61, 61);
                border: 2px solid red;
            }
            QFrame > QLabel {
                border: 0;  /* 移除内部QLabel的边框样式 */
            }
            QFrame > QLabel:hover{
                border: 0;  /* 移除内部QLabel的边框样式 */
            }
            """
        )

    @staticmethod
    def remove_styles(dom, d2):
        dom.setStyleSheet(
            """ 
                QFrame{
                    background: transparent;  /* 设置背景色为亮绿色 */
                    border:0;
                }
                QFrame:hover{
                    border: 1px solid red;
                }
                QFrame > QLabel{
                    border: 0;  /* 移除内部QLabel的边框样式 */
                }
                QFrame > QLabel:hover{
                    border: 0;  /* 移除内部QLabel的边框样式 */
                }
            """)

    @staticmethod
    def debug_fun():
        print('debug模式')
        while 1:
            image = grab_gpt_win(region, False)
            cv2.namedWindow(cv2_win_name, cv2.WINDOW_NORMAL)  # 创建窗口
            cv2.resizeWindow(cv2_win_name, region[0] + region[1], int(1920 * (region[3] / region[2])))  # 重置窗口大小
            cv2.moveWindow(cv2_win_name, 0, 0)
            cv2.setWindowProperty(cv2_win_name, cv2.WND_PROP_TOPMOST, 1)  # 设置窗口置顶
            cv2.imshow(cv2_win_name, image)
            cv2.waitKey(1)

    def remove_hero(self, name):
        image_label, frame = self.find_node_by_name(name)
        self.selected_images[name] = not self.selected_images[name]
        self.remove_styles(frame, image_label)
        print(self.active_list, '选中列表变更')
        self.update_ini_file()  # 同步ini

    def create_double_click_event(self, frame, free, index, image_label):
        def mouseDoubleClickEvent(event):
            # 判断是否已经存在相同的名称
            name, price = extract_filename(self.dict_list[free][index])
            self.selected_images[name] = not self.selected_images[name]
            # 更改样式
            if name not in self.active_list:
                print(f"选中图片 所属费别：{free}费 下标:{index} 图片名称：{name}")
                # 添加名称到数组
                self.active_list.append(name)
                self.active_styles(frame, image_label)
            else:
                print(f"选中图片 所属费别：{free}费 下标:{index} 图片名称：{name}")
                # 从数组中移除名称
                self.active_list.remove(name)
                self.remove_styles(frame, image_label)

            print(self.active_list, '选中列表变更')
            self.imClass.reload_image_list()  # 更新左侧选中列表
            self.update_ini_file()  # 同步ini

        return mouseDoubleClickEvent

    def on_press_c(self, key):
        if key == keyboard.KeyCode(char='`'):
            self.is_open = not self.is_open
            print(f"开启状态：{self.is_open}")
            if self.is_open:
                winsound.PlaySound('music/8855.wav', flags=1)
            else:
                winsound.PlaySound('music/close.wav', flags=1)

    def keyboard_listener(self):
        # 创建键盘监听器
        listener = keyboard.Listener(on_press=self.on_press_c)
        # 启动监听器
        listener.start()

    def update_ini_file(self):
        global config
        if os.access(self.ini_file, os.W_OK):
            config.set('section_name', 'group', ','.join(self.active_list))
            with open(self.ini_file, 'w') as file:
                config.write(file)

    def load_ini_file(self):
        if os.access(self.ini_file, os.R_OK):
            if os.path.isfile(self.ini_file):
                config.read(self.ini_file)
                group = config.get('section_name', 'group')
                self.active_list = group.split(',') if group else []
                print(self.active_list, '初始化时的列表')
            else:
                self.active_list = []
                # 创建新的INI文件并初始化配置项
                config['section_name'] = {'group': ''}
                with open(self.ini_file, 'w') as file:
                    config.write(file)

    def load_styles(self):
        # 设置窗口透明度
        self.setStyleSheet("""
                  /* 设置窗口背景颜色 */
                  background-color: rgb(15,15,15);  /* 使用深色背景 */

                  /* 设置窗口圆角 */
                  border-radius: 10px;

                  /* 设置按钮样式 */
                  QPushButton {
                      background-color: #444;  /* 设置按钮背景色为深灰色 */
                      color: #fff;  /* 设置按钮文字颜色为白色 */
                      border: none;
                      padding: 10px 20px;
                      border-radius: 5px;
                  }
                  QPushButton:hover {
                      background-color: #666;  /* 设置鼠标悬停时的背景色为较浅的灰色 */
                  }
                  QPushButton:pressed {
                      background-color: #333;  /* 设置按下时的背景色为稍深的灰色 */
                  }

                  /* 设置标签样式 */
                  QLabel {
                      color: #fff;  /* 设置文字颜色为白色 */
                      font-size: 16px;
                  }

                  /* 设置文本框样式 */
                  QLineEdit {
                      background-color: #333;  /* 设置文本框背景色为深灰色 */
                      color: #fff;  /* 设置文本框文字颜色为白色 */
                      border: 1px solid #666;  /* 设置文本框边框样式为浅灰色 */
                      border-radius: 5px;
                      padding: 5px;
                  }
              """)
        # 配置窗口标题
        self.setWindowTitle("云顶秒卡小工具 快捷键~开关 管理员模式运行 鼠标双击选中 or 取消")

    def closeEvent(self, event):
        self.is_running = False  # 设置标志位
        self.keyboard_thread.join()
        app.quit()


class StartWindow(QWidget):
    def __init__(self):
        global is_grayscale, ocr
        super().__init__()
        self.setWindowTitle("作者1bit 软件免费 软件免费 软件免费 如对您有帮助 请支持一下 谢谢 ")
        self.setFixedSize(600, 400)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)  # 设置窗口前置
        self.load_styles()
        self.setWindowIcon(QIcon('1.ico'))
        # 创建布局和控件
        layout = QVBoxLayout()
        self.setLayout(layout)
        # 创建 CSS 类
        css_class = "white-label { color: white; font-size:18px; }"
        # 将 CSS 类应用于所有标签
        self.setStyleSheet(css_class)
        # 创建水平布局
        hbox = QHBoxLayout()

        # 添加第一个图片和标签
        image1_label = QLabel(self)
        pixmap1 = QPixmap('images/1.jpg')
        image1_label.setPixmap(pixmap1)

        label1 = QLabel("微信扫上面↑↑↑~", self)
        label1.setObjectName("white-label")
        label1.setAlignment(Qt.AlignCenter)
        # 创建垂直布局并添加图片和标签
        vbox1 = QVBoxLayout()
        vbox1.addWidget(image1_label, alignment=Qt.AlignCenter)
        vbox1.addWidget(label1, alignment=Qt.AlignCenter)

        # 将垂直布局添加到水平布局中
        hbox.addLayout(vbox1)

        # 添加间隔
        hbox.addSpacing(30)

        # 添加第二个图片和标签
        image2_label = QLabel(self)
        pixmap2 = QPixmap('images/2.png')
        image2_label.setPixmap(pixmap2)

        label2 = QLabel("支付宝扫上面↑↑↑~", self)
        label2.setObjectName("white-label")
        label2.setAlignment(Qt.AlignCenter)

        # 创建垂直布局并添加图片和标签
        vbox2 = QVBoxLayout()
        vbox2.addWidget(image2_label, alignment=Qt.AlignCenter)
        vbox2.addWidget(label2, alignment=Qt.AlignCenter)

        # 将垂直布局添加到水平布局中
        hbox.addLayout(vbox2)

        # 添加水平布局到垂直布局中
        layout.addLayout(hbox)
        # 添加布局
        h_layout_debug = QHBoxLayout()
        # 添加Label
        label_debug = QLabel("debug模式", self)

        # 设置左边距为50像素，下边距为30像素
        h_layout_debug.setContentsMargins(35, 0, 0, 30)

        # 设置标签和按钮之间的间距为50像素
        h_layout_debug.setSpacing(50)

        # 添加按钮组
        self.gpu_button_group_debug = QButtonGroup(self)
        self.gpu_button_group_debug.setExclusive(True)  # 确保只能选择一个按钮

        # 添加按钮
        self.gpu_button3 = QRadioButton("是", self)
        self.gpu_button4 = QRadioButton("否", self)

        self.gpu_button_group_debug.addButton(self.gpu_button3, 0)
        self.gpu_button_group_debug.addButton(self.gpu_button4, 1)

        # 默认选择"否"
        self.gpu_button4.setChecked(True)

        self.gpu_button_group_debug.buttonClicked[int].connect(self.update_is_debug)

        # 设置stretch因子，使Label和按钮均分一行
        h_layout_debug.addWidget(label_debug, 1)
        h_layout_debug.addWidget(self.gpu_button3, 1)
        h_layout_debug.addWidget(self.gpu_button4, 1)
        # ___
        # 添加主布局
        layout.addLayout(h_layout_debug)

        button = QPushButton("进入主程序", self)
        button.clicked.connect(self.open_main_window)

        button.setStyleSheet('''
            QPushButton {
                background-color: rgb(61, 61, 61);
                color: white;
                font-size: 20px;
                border-radius: 10px;
                padding: 5px;
            }
        ''')

        layout.addWidget(button)

        # 设置窗口样式
        self.setStyleSheet("""
            StartWindow {
                background-color: rgb(30, 30, 30);
            }
        """)

    @staticmethod
    def update_is_grayscale(gpu_button):
        global is_grayscale
        if gpu_button == 0:
            is_grayscale = True
        else:
            is_grayscale = False
        print(f"是否灰度匹配:{is_grayscale}")

    @staticmethod
    def update_is_debug(gpu_button):
        global is_debug
        if gpu_button == 0:
            is_debug = True
        else:
            is_debug = False
        print(f"是否开启debug:{is_debug}")

    def load_styles(self):
        # 设置窗口透明度
        self.setStyleSheet("""
                      /* 设置窗口圆角 */
                      border-radius: 10px;

                      /* 设置按钮样式 */
                      QPushButton {
                          background-color: #444;  /* 设置按钮背景色为深灰色 */
                          color: #fff;  /* 设置按钮文字颜色为白色 */
                          border: none;
                          padding: 10px 20px;
                          border-radius: 5px;
                      }
                      QPushButton:hover {
                          background-color: #666;  /* 设置鼠标悬停时的背景色为较浅的灰色 */
                      }
                      QPushButton:pressed {
                          background-color: #333;  /* 设置按下时的背景色为稍深的灰色 */
                      }

                      /* 设置标签样式 */
                      QLabel {
                          color: #fff;  /* 设置文字颜色为白色 */
                          font-size: 16px;
                      }

                      /* 设置文本框样式 */
                      QLineEdit {
                          background-color: #333;  /* 设置文本框背景色为深灰色 */
                          color: #fff;  /* 设置文本框文字颜色为白色 */
                          border: 1px solid #666;  /* 设置文本框边框样式为浅灰色 */
                          border-radius: 5px;
                          padding: 5px;
                      }
                  """)

    def open_main_window(self):
        self.close()
        # 创建并显示主窗口
        main_window = MainWindow()
        main_window.show()
        # 将窗口前置
        main_window.setWindowState(main_window.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        main_window.activateWindow()
        main_window.raise_()


if __name__ == '__main__':
    # 解决多线程打包问题
    multiprocessing.freeze_support()
    if not windll.shell32.IsUserAnAdmin():
        ctypes.windll.user32.MessageBoxW(0, "需要管理员权限来运行此程序", "权限错误", 0x10)
        sys.exit()
    ALL_IMAGE_LIST = get_files()
    # print(ALL_IMAGE_LIST, 'ALL_IMAGE_LIST')
    # 创建应用程序实例
    app = QApplication([])
    # 设置图标
    app.setWindowIcon(QIcon('1.icon'))
    # 创建并显示 StartWindow
    start_window = StartWindow()
    start_window.show()

    # 运行应用程序主循环
    sys.exit(app.exec())
