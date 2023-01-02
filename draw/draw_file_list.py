from PyQt5.QtWidgets import QListWidget
import os

# NAS_PATH = "Z:\정제데이터"
# NAS_PATH = r"\\192.168.50.251\드론데이터\정제데이터"

# def draw_file_list(file_list_path): # NAS에서 영상 파일 리스트를 불러오고 Listwidget을 반환하는 함수
#     list_widget = QListWidget()
#     if file_list_path != None:
#         print(file_list_path)
#         for root, dirs, files in os.walk(file_list_path):
#             for file in files:
#                 print(file)
#                 _, extension = os.path.splitext(file)
#                 if extension == ".mp4":
#                     list_widget.addItem(root + "/" + file)
#     return list_widget