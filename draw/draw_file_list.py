from PyQt5.QtWidgets import QListWidget
import os

NAS_PATH = "Z:\정제데이터"

def draw_file_list(): # NAS에서 영상 파일 리스트를 불러오고 Listwidget을 반환하는 함수
    list_widget = QListWidget()
    for root, dirs, files in os.walk(NAS_PATH):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension == ".mp4":
                list_widget.addItem(root + "/" + file)

    return list_widget