from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QGroupBox

def draw_camera_object_groupbox(): # 객체 통계 그룹박스 그리기 함수
    vbox = QVBoxLayout()
    total_object_hbox = QHBoxLayout()
    person_hbox = QHBoxLayout()
    car_hbox = QHBoxLayout()
    boat_hbox = QHBoxLayout()

    # detection total count
    total_object_count = QLabel("Total Object Count : ")
    total_object_count_num = QLabel("1", objectName='total_object_count') # 추후 init_ui에선 objectName을 통해 해당 라벨 접근 가능
    '''
        init_ui에서 해당 라벨 접근 코드
        detect_count = self.camera_groupbox_widget.findChild(QLabel, "detect_count")
        print(detect_count.text())
    '''
    total_object_hbox.addWidget(total_object_count)
    total_object_hbox.addWidget(total_object_count_num)

    # detection person count
    person_count = QLabel("Person Count : ", objectName='person_count')
    person_count_num = QLabel("1")
    person_hbox.addWidget(person_count)
    person_hbox.addWidget(person_count_num)

    # detection car count
    car_count = QLabel("Car Count : ", objectName='car_count')
    car_count_num = QLabel("1")
    car_hbox.addWidget(car_count)
    car_hbox.addWidget(car_count_num)

    # detection boat count
    boat_count = QLabel("Boat Count : ", objectName='boat_count')
    boat_count_num = QLabel("1")
    boat_hbox.addWidget(boat_count)
    boat_hbox.addWidget(boat_count_num)

    vbox.addLayout(total_object_hbox)
    vbox.addLayout(person_hbox)
    vbox.addLayout(car_hbox)
    vbox.addLayout(boat_hbox)

    groupbox = QGroupBox('Object')
    groupbox.setLayout(vbox)

    return groupbox

def draw_camera_action_groupbox(): # 액션 통계 그룹박스 그리기 함수
    vbox = QVBoxLayout()
    total_action_hbox = QHBoxLayout()
    sos_hbox = QHBoxLayout()
    fall_down_hbox = QHBoxLayout()

    # detection total count
    total_action_count = QLabel("Total Action Count : ")
    total_action_count_num = QLabel("1", objectName='total_action_count')
    total_action_hbox.addWidget(total_action_count)
    total_action_hbox.addWidget(total_action_count_num)

    # detection person count
    sos_count = QLabel("SOS Count : ", objectName='sos_count')
    sos_count_num = QLabel("1")
    sos_hbox.addWidget(sos_count)
    sos_hbox.addWidget(sos_count_num)

    # detection car count
    fall_down_count = QLabel("Fall Down Count : ", objectName='fall_down_count')
    fall_down_count_num = QLabel("1")
    fall_down_hbox.addWidget(fall_down_count)
    fall_down_hbox.addWidget(fall_down_count_num)

    vbox.addLayout(total_action_hbox)
    vbox.addLayout(sos_hbox)
    vbox.addLayout(fall_down_hbox)

    groupbox = QGroupBox('Action')
    groupbox.setLayout(vbox)

    return groupbox

