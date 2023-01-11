import pyqtgraph as pg
import numpy as np

def draw_month_barchart():  # barchar 그리기 함수
    pg.setConfigOptions(background=(240, 240, 240), foreground=(0, 0, 0))  # pyqtgraph 옵션 설정(전경,배경 색 지정)

    month_axis = np.arange(1, 13)  # x축 데이터 생성 (1~12)
    # data = np.random.randint(100, size=12)  # y축 랜덤 데이터 생성 <- 월별 탐지 객체수 데이터 삽입 장소
    data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    pw = pg.PlotWidget(title="Detect Count per Month")
    pw.showGrid(x=False, y=False)  # x축, y축 격자 무늬 제거
    pw.setXRange(0, 13, padding=0)  # x축 범위 설정(0~13)
    pw.setYRange(0, max(data) + (max(data) * 0.01), padding=0)  # y축 범위 설정(0~y최대값)

    barchar = pg.BarGraphItem(x=month_axis, height=data, width=0.8, brush=(31, 56, 120), pen=(100, 100, 100))  # barchart 생성 / x축 = 1~12 / y축 = 랜덤 데이터 12개
    pw.addItem(barchar)

    return pw