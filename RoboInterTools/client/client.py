import sys
import os
import argparse
from PyQt5.QtCore import QPoint, QTimer, Qt
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton, QMessageBox, QLineEdit, QDialogButtonBox, QTextEdit, QGridLayout,
                             QLabel, QSlider, QDialog, QHBoxLayout, QFrame, QProgressDialog, QRadioButton, QPlainTextEdit, QComboBox, QCheckBox)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal, QThread

import yaml, time
from utils import request_video_and_anno, save_anno, drawback_video, get_avaiable_username
import numpy as np

BASE_CLIP_DES = ['拿着[某物体]从[某位置1]移动到[某位置2]', '在[某位置]抓起[某物体]', '把[某物体]放置到[某位置]', '在[某位置]按压[某物体]', '把[某物体]推到[某位置]', '把[某物体]拉到[某位置]', '[顺时针/逆时针]转动[某物体]',  '把[某物体]倒到[某位置]', '在[某位置]折叠[某物体]', '在[某位置]滑动[某物体]', '把[某物体]插入到[某位置]', '在[某位置]摇动[某物体]', '在[某位置]敲击[某物体]', '把[某物体]扔到[某位置]', '在[某位置]操作[某物体]']
BASE_PRIM = ['拿着物体移动','抓起','放下','按压','推动','拉动','转动','倾倒','折叠','滑动','插入','摇动','敲击','扔掉','其余操作']

class TextInputDialog(QDialog):
    
    def __init__(self, initial_text='', parent=None, is_video=True, video_anno_json=None, origin_text=None):
        super().__init__(parent)
        self.setWindowTitle('Please input the language annotation.')
        self.setFocusPolicy(Qt.StrongFocus)
        self.is_video = is_video
        self.main_layout = QGridLayout(self)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.language_edit = QTextEdit()
        self.language_title = QLabel('Please confirm the atom action:', self)
        self.language_title.hide()
        self.mode_title = QLabel('Please choose the atom action:', self)
        self.mode_select = QComboBox()
        self.mode_title.hide()
        self.mode_select.hide()
        
        if not is_video:
            global_instruction_C = video_anno_json['instructionC']
            clip_lang_C_options = video_anno_json['task_stepsC']
            task_stepsC_list = video_anno_json['task_stepsC_list']
            
            self.prim_title = QLabel('Please choose the language Pre-annotation:', self)
            self.prim_select = QComboBox()
            self.prim_select.setFixedSize(400, 30)
            self.mode_title.show()
            self.mode_select.show()
            clip_des = [i for i in task_stepsC_list if clip_lang_C_options[i] is None or i == origin_text] + ['空']
            primitive_action_options_C = [video_anno_json['action_stepsC'][idx] for idx, i in enumerate(task_stepsC_list) if clip_lang_C_options[i] is None or i == origin_text] + ['空']
            
            self.prim_select.addItems(clip_des)
            self.prim_select.insertSeparator(len(clip_des))
            self.prim_select.addItems(BASE_CLIP_DES)
            self.prim_select.setMaxVisibleItems(30)            
            
            self.mode_select.setFixedSize(400, 30)
            self.mode_select.addItems(primitive_action_options_C)
            # 添加分割线
            self.mode_select.insertSeparator(len(primitive_action_options_C))
            self.mode_select.addItems(BASE_PRIM)
            self.mode_select.setCurrentIndex(-1)
            self.mode_select.setMaxVisibleItems(30)
            
            
            if origin_text and len(origin_text) > 0:
                # self.prim_select.addItems([initial_action])
                self.prim_select.setCurrentText(origin_text)
                self.language_select(clip_des)
            else:
                self.prim_select.setCurrentIndex(-1)
            
            if initial_text is not None and len(initial_text) > 0:
                self.language_edit.setText(initial_text)
            else:
                self.language_edit.hide()
                self.language_edit.setText('')
            
            self.language_edit.setFixedSize(400, 150)
            self.prim_select.currentIndexChanged.connect(lambda: self.language_select(clip_des))
            self.main_layout.addWidget(self.language_edit, 2, 1)
            self.main_layout.addWidget(self.language_title, 2, 0)
            self.main_layout.addWidget(self.mode_title, 1, 0)
            self.main_layout.addWidget(self.mode_select, 1, 1)
            self.main_layout.addWidget(self.prim_title, 0, 0)
            self.main_layout.addWidget(self.prim_select, 0, 1)
            self.main_layout.addWidget(self.button_box, 3, 0, 1, 2)
        else:
            global_instruction_C = video_anno_json['instructionC']
            self.text_title = QLabel('Input the language annotation:', self)
            self.text_input = QPlainTextEdit(self)
            self.text_input.setPlainText(global_instruction_C if initial_text is None or len(initial_text) == 0 else initial_text)
            self.text_input.setFixedSize(500,50)
            # self.text_input.setWordWrap(True)
            # self.text_input.setPlainText(initial_text)
            self.main_layout.addWidget(self.text_title, 0, 0)
            self.main_layout.addWidget(self.text_input, 0, 1)
            self.main_layout.addWidget(self.button_box, 1, 0, 1, 2)
                
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
    def get_text(self):
        return self.language_edit.toPlainText() if not self.is_video else self.text_input.toPlainText()
    
    def get_prim(self):
        if not self.is_video:
            return self.mode_select.currentText()
        else:
            return ''
    
    def get_select_lang(self):
        if not self.is_video:
            return self.prim_select.currentText()
        else:
            return ''
    
    def language_select(self, clip_des):
        if not self.is_video:
            self.language_title.show()
            self.language_edit.show()
            self.language_edit.setText(self.prim_select.currentText())
            # if self.prim_select.currentText() == '空':
            #     self.mode_select.setCurrentIndex(-1)
            # else:
            clip_lang_C_options_keys = clip_des + ['Sep'] + BASE_CLIP_DES
            prim_idx_in_action = clip_lang_C_options_keys.index(self.prim_select.currentText())
            self.mode_select.setCurrentIndex(prim_idx_in_action)


class ObjectAnnotationDialog(QDialog):
    
    def __init__(self, object_size, frame_number, keyframes, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Please input the object annotation')
        self.setFocusPolicy(Qt.StrongFocus)
        self.main_layout = QGridLayout(self)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        # add object size input box
        self.object_size = object_size
        self.start_frame_id = [0] + keyframes
        for idx, start_idx in enumerate(self.start_frame_id):
            
            if idx == len(self.start_frame_id)-1:
                end_idx = frame_number - 1
            else:
                end_idx = self.start_frame_id[idx+1] - 1
            
            self.main_layout.addWidget(QLabel(f'Video Range: [{start_idx+1}:{end_idx+1}]', self), idx, 0)
            select_button = QComboBox()
            select_button.addItems([f"Object{i+1}" for i in range(self.object_size)])
            select_button.setFixedSize(100, 30)
            select_button.setCurrentIndex(idx if idx < self.object_size else -1)
            self.main_layout.addWidget(select_button, idx, 1)
        
        self.main_layout.addWidget(self.button_box, len(self.start_frame_id), 0, 1, 2)
        # self.main_layout.itemAt(self.object_size*2-1).widget().setText(str(frame_number))
        
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
    
    def get_obj_id(self):
        return [int(self.main_layout.itemAt(i+1).widget().currentText().replace('Object','')) for i in range(0, len(self.start_frame_id)*2, 2)]
    
    def get_frame_id(self):
        return [int(self.main_layout.itemAt(i).widget().text().split('[')[1].split(':')[0]) for i in range(0, len(self.start_frame_id)*2, 2)]
    
    def get_result(self):
        return dict(zip(self.get_frame_id(), self.get_obj_id()))
    

class VideoPlayer(QWidget):
    def __init__(self, args):
        super().__init__()
        self.setWindowTitle("RoboInter-Tool")
        ###########################################################
        #################### Main Area Layout ####################
        ###########################################################
        main_layout = QHBoxLayout()     
        self.mode, self.username, self.ip_address, self.port, self.time = self.mode_choose()
        if self.mode == '语言标注':
            #resize the window
            self.setFixedSize(1300, 700)
        else:
            self.setFixedSize(1900, 730)
        ###########################################################
        #################### Video Area Layout ####################
        ###########################################################
        video_layout = QVBoxLayout()
        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setMouseTracking(True)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        # Progress slider
        self.progress_slider = QSlider(self)
        self.progress_slider.setOrientation(Qt.Horizontal)  
        self.progress_slider.valueChanged.connect(self.seek_video)
        self.progress_slider.hide()  # Hide initially
        video_layout.addWidget(self.progress_slider)
        # Keyframe indicator bar
        self.keyframe_bar = QLabel(self)
        self.keyframe_bar.setFixedHeight(10) 
        self.keyframe_bar.setMouseTracking(True)
        self.keyframe_bar.setAlignment(Qt.AlignCenter)  
        self.keyframe_bar.installEventFilter(self)  
        video_layout.addWidget(self.keyframe_bar)
        # Dynamic frame position label that floats above the slider
        self.frame_position_label = QLabel(self)
        self.frame_position_label.setStyleSheet("background-color: #E3E3E3;")
        self.frame_position_label.setAlignment(Qt.AlignCenter)
        self.frame_position_label.setFixedSize(150, 30)
        self.frame_position_label.hide()
        video_control_button_layout = QHBoxLayout()
        # Video position label
        self.video_position_label = QLabel(self)
        self.video_position_label.setStyleSheet("background-color: #E3E3E3;")
        self.video_position_label.setAlignment(Qt.AlignCenter)
        if self.mode != '语言标注':
            self.video_position_label.setFixedSize(130, 30)
        else:
            self.video_position_label.setFixedSize(310, 30)
        video_control_button_layout.addWidget(self.video_position_label)
        self.hist_num_label = QLabel(self)
        self.hist_num_label.setStyleSheet("background-color: #E3E3E3;")
        self.hist_num_label.setAlignment(Qt.AlignCenter)
        if self.mode != '语言标注':
            self.hist_num_label.setFixedSize(160, 30)
        else:
            self.hist_num_label.setFixedSize(310, 30)
        video_control_button_layout.addWidget(self.hist_num_label)
        
        if self.mode != '语言标注':
            self.one_num_label = QLabel(self)
            self.one_num_label.setStyleSheet("background-color: #E3E3E3;")
            self.one_num_label.setAlignment(Qt.AlignCenter)
            self.one_num_label.setFixedSize(270, 30)
            video_control_button_layout.addWidget(self.one_num_label)
        
        if self.mode != '语言标注':
            self.two_num_label = QLabel(self)
            self.two_num_label.setStyleSheet("background-color: #E3E3E3;")
            self.two_num_label.setAlignment(Qt.AlignCenter)
            self.two_num_label.setFixedSize(270, 30)
            video_control_button_layout.addWidget(self.two_num_label)
        
        if self.mode != '语言标注':
            self.three_num_label = QLabel(self)
            self.three_num_label.setStyleSheet("background-color: #E3E3E3;")
            self.three_num_label.setAlignment(Qt.AlignCenter)
            self.three_num_label.setFixedSize(270, 30)
            video_control_button_layout.addWidget(self.three_num_label)
        
        # Next video button
        video_layout.addLayout(video_control_button_layout)
        video_load_button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play", self)
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_playback)
        video_load_button_layout.addWidget(self.play_button)
        
        # 标注次数选择
        if self.mode != '语言标注':
            self.re_annotation_text = QLabel("Quant Time: ", self)
            self.re_annotation_text.setFixedSize(80, 30)
            video_load_button_layout.addWidget(self.re_annotation_text)
            self.re_annotation_button = QComboBox()
            self.re_annotation_button.setFixedSize(80, 30)
            self.re_annotation_button.addItem('0')
            self.re_annotation_button.addItem('1')
            self.re_annotation_button.addItem('2')
            self.re_annotation_button.addItem('3')
            self.re_annotation_button.setCurrentIndex(self.time)
            text_sep = QLabel(' ')
            text_sep.setFixedSize(40, 30)
            video_load_button_layout.addWidget(self.re_annotation_button)
            video_load_button_layout.addWidget(text_sep)
            # video_load_button_layout.addStretch(1)
            # 检测状态变化
            self.re_annotation_button.currentIndexChanged.connect(self.check_re_anno)
        
        # 选择是否完成
        if self.mode != '语言标注':
            self.is_finished_button = QCheckBox("Finish", self)
            self.is_finished_button.setChecked(False)
            self.is_finished_button.setFixedSize(80, 30)
            self.is_finished_button.toggled.connect(self.change_hard_button)
            video_load_button_layout.addWidget(self.is_finished_button)
            self.is_finished_button.hide()
            
            self.is_hard_sample_button = QCheckBox("Hard Sample", self)
            self.is_hard_sample_button.setChecked(False)
            self.is_hard_sample_button.setFixedSize(110, 30)
            self.is_hard_sample_button.toggled.connect(self.change_finish_button)
            video_load_button_layout.addWidget(self.is_hard_sample_button)
            self.is_hard_sample_button.hide()

            text_sep = QLabel(' ')
            text_sep.setFixedSize(40, 30)
            video_load_button_layout.addWidget(text_sep)
        
        
        # 复选框
        self.is_pre_button = QCheckBox("Back", self)
        self.is_pre_button.setChecked(False)
        self.is_pre_button.setFixedSize(100, 30)
        self.is_pre_button.setDisabled(True)
        # 检测状态变化
        self.is_pre_button.toggled.connect(self.set_button_text)
        video_load_button_layout.addWidget(self.is_pre_button)
        
        
        self.next_button = QPushButton("Save and Load", self)
        self.next_button.clicked.connect(self.next_video_and_load)
        video_load_button_layout.addWidget(self.next_button)
        video_layout.addLayout(video_load_button_layout)
        main_layout.addLayout(video_layout)
        ###########################################################
        ######################## Separator ########################
        ###########################################################
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(separator)
        ###########################################################
        #################### Toolbar Area Layout ##################
        ###########################################################
        self.toolbar_layout = QVBoxLayout()
        # Auto-annotation function title layout
        function_title_layout = QHBoxLayout()
        function_title = QLabel("Segmentation Annotation Tool Area", self)
        function_title.setAlignment(Qt.AlignLeft)  # Left align the title
        function_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        function_title_layout.addWidget(function_title)
        fline = QFrame(self)
        fline.setFrameShape(QFrame.HLine)
        fline.setFrameShadow(QFrame.Sunken)
        fline.setStyleSheet("color: grey;")  # Set the same color as the title
        function_title_layout.addWidget(fline)
        self.sam_object_layout = QHBoxLayout()
        if self.mode != '语言标注':
            self.toolbar_layout.addLayout(function_title_layout)
            # run button
            self.button_param_select = QComboBox()
            self.button_param_select.addItem('Bidirectional Mode')
            self.button_param_select.addItem('Forward Mode')
            self.button_param_select.addItem('Backward Mode')
            self.button_param_select.setFixedSize(150, 30)
            self.sam_object_layout.addWidget(self.button_param_select)
        
        # sam pre object button
        self.sam_pre_button = QPushButton("Pre Object", self)
        self.sam_pre_button.clicked.connect(self.pre_sam_object)
        self.sam_pre_button.setDisabled(True)
        # sam object position label
        self.sam_obj_pos_label = QLabel(self)
        self.sam_obj_pos_label.setStyleSheet("background-color: #E3E3E3;")
        self.sam_obj_pos_label.setAlignment(Qt.AlignCenter)
        self.sam_obj_pos_label.setFixedSize(150, 30)
        # sam next object button
        self.sam_next_button = QPushButton("Next Object", self)
        self.sam_next_button.clicked.connect(self.next_sam_object)
        self.sam_next_button.setDisabled(True)
        self.sam_object_layout.addWidget(self.sam_pre_button)
        self.sam_object_layout.addWidget(self.sam_obj_pos_label)
        self.sam_object_layout.addWidget(self.sam_next_button)
        if self.mode != '语言标注':
            self.toolbar_layout.addLayout(self.sam_object_layout)

        # edit mode are layout
        annotation_title_layout = QHBoxLayout()
        annotation_title = QLabel("Annotation editing area", self)
        annotation_title.setAlignment(Qt.AlignLeft)  # Left align the title
        annotation_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        annotation_title_layout.addWidget(annotation_title)
        annoline = QFrame(self)
        annoline.setFrameShape(QFrame.HLine)
        annoline.setFrameShadow(QFrame.Sunken)
        annoline.setStyleSheet("color: grey;")  # Set the same color as the title
        annotation_title_layout.addWidget(annoline)
        if self.mode != '语言标注':
            self.toolbar_layout.addLayout(annotation_title_layout)
        # clear all button
        self.control_button_layout = QHBoxLayout()
        self.clear_all_button = QPushButton("Delete all anno", self)
        self.clear_all_button.clicked.connect(self.clear_annotations)
        self.control_button_layout.addWidget(self.clear_all_button)
        # remove last button
        self.remove_last_button = QPushButton("Delete last anno", self)
        self.remove_last_button.clicked.connect(self.remove_last_annotation)
        self.control_button_layout.addWidget(self.remove_last_button)
        # remove frame button
        self.remove_frame_button = QPushButton("Delete object anno", self)
        self.remove_frame_button.clicked.connect(self.remove_obj_annotation)
        self.control_button_layout.addWidget(self.remove_frame_button)
        # save button
        if self.mode != '语言标注':
            self.toolbar_layout.addLayout(self.control_button_layout)

        # Video language annotation area
        lang_layout = QVBoxLayout()
        lang_title_layout = QHBoxLayout()
        lang_title = QLabel("Video Language Annotation Display Area", self)
        lang_title.setAlignment(Qt.AlignLeft)  # Left align the title
        lang_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        lang_title_layout.addWidget(lang_title)
        videoline = QFrame(self)
        videoline.setFrameShape(QFrame.HLine)
        videoline.setFrameShadow(QFrame.Sunken)
        videoline.setStyleSheet("color: grey;")  # Set the same color as the title
        lang_title_layout.addWidget(videoline)
        lang_layout.addLayout(lang_title_layout)
        # Video Language annotation show area
        self.video_lang_input = QTextEdit(self)
        self.video_lang_input.setReadOnly(True)
        self.video_lang_input.setFixedSize(610, 70)
        self.video_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
        lang_layout.addWidget(self.video_lang_input)
        
        # Video clip language annotation area 
        lang_title_layout = QHBoxLayout()
        clip_title = QLabel("Video segment language annotation display area", self)
        clip_title.setAlignment(Qt.AlignLeft)  # Left align the title
        clip_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        lang_title_layout.addWidget(clip_title)
        clipline = QFrame(self)
        clipline.setFrameShape(QFrame.HLine)
        clipline.setFrameShadow(QFrame.Sunken)
        clipline.setStyleSheet("color: grey;")  # Set the same color as the title
        lang_title_layout.addWidget(clipline)
        lang_layout.addLayout(lang_title_layout)
        # Video clip Language annotation show area
        self.clip_lang_input = QTextEdit(self)
        self.clip_lang_input.setReadOnly(True)
        self.clip_lang_input.setFixedSize(610, 100)
        self.clip_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
        lang_layout.addWidget(self.clip_lang_input)
        
        if self.mode == '分割标注':
            # 删除语言标注区域
            self.video_lang_input.hide()
            self.clip_lang_input.hide()
            clip_title.hide()
            lang_title.hide()
            clipline.hide()
            videoline.hide()
            tips_items = ['A: Pre Frame', 'D: Next Frame', 'W: Set Contact Frame', 'S: Set Sep Frame', 'Backspace: Delete Frame', 'L: Combine Object']
            self.sam_time = "first"

        elif self.mode == '语言标注':
            fline.hide()
            self.sam_pre_button.hide()
            self.sam_next_button.hide()
            self.sam_obj_pos_label.hide()
            self.keyframe_bar.hide()
            function_title.hide()
            annotation_title.hide()
            annoline.hide()
            self.clear_all_button.hide()
            self.remove_last_button.hide()
            self.remove_frame_button.hide()
            tips_items = ['W: 标志开始帧','S: 标记结束帧','F: 播放/暂停视频', '删除: 删除标记段','A: 上一帧','回车: 添加视频标注','D: 下一帧', 'L: 修改视频段语言']
            
            preview_clip_layout = QVBoxLayout()
            preview_lang_title_layout = QHBoxLayout()
            preview_clip_title = QLabel("视频段可用语言预览", self)
            preview_clip_title.setAlignment(Qt.AlignLeft)  # Left align the title
            preview_clip_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
            preview_lang_title_layout.addWidget(preview_clip_title)
            preview_clipline = QFrame(self)
            preview_clipline.setFrameShape(QFrame.HLine)
            preview_clipline.setFrameShadow(QFrame.Sunken)
            preview_clipline.setStyleSheet("color: grey;")  # Set the same color as the title
            preview_lang_title_layout.addWidget(preview_clipline)
            preview_clip_layout.addLayout(preview_lang_title_layout)
            # Video clip Language annotation show area
            self.preview_clip_lang_input = QTextEdit(self)
            self.preview_clip_lang_input.setReadOnly(True)
            self.preview_clip_lang_input.setFixedSize(610, 160)
            self.preview_clip_lang_input.setStyleSheet("background-color: #E3E3E3; font-weight: bold;")
            preview_clip_layout.addWidget(self.preview_clip_lang_input)
            lang_layout.addLayout(preview_clip_layout)
            self.toolbar_layout.addLayout(lang_layout)
              
        self.tips_layout = QVBoxLayout()
        self.tips_title_layout = QHBoxLayout()
        tips_title = QLabel("Shortcut keys", self)
        tips_title.setAlignment(Qt.AlignLeft)  # Left align the title
        tips_title.setStyleSheet("color: grey; font-weight: bold;")  # Set font color and weight
        self.tips_title_layout.addWidget(tips_title)
        line = QFrame(self)
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: grey;")  # Set the same color as the title
        self.tips_title_layout.addWidget(line)
        self.tips_layout.addLayout(self.tips_title_layout)
        self.toolbar_layout.addLayout(self.tips_layout)
        self.tips_text_layout = QGridLayout()
        
        for i, item in enumerate(tips_items):
            tips_input = QTextEdit(self)
            tips_input.setText(item)
            tips_input.setReadOnly(True)
            tips_input.setSizeAdjustPolicy(QTextEdit.AdjustToContents)
            if self.mode != '语言标注':
                tips_input.setFixedSize(240, 30)
            else:
                tips_input.setFixedSize(200, 30)
            
            if self.mode == '语言标注':
                num_per_line = 3
            else:
                num_per_line = 2

            line = int(i // num_per_line)
            self.tips_text_layout.addWidget(tips_input, line, i % num_per_line)
        
        self.toolbar_layout.addLayout(self.tips_text_layout)
        
        if self.mode != '语言标注':
            # 添加视频名称展示区域
            video_name_layout = QHBoxLayout()
            video_name_title = QLabel("Video Name: ", self)
            video_name_title.setAlignment(Qt.AlignLeft)
            video_name_title.setStyleSheet("color: grey; font-weight: bold;")
            video_name_layout.addWidget(video_name_title)
            self.video_name_input = QLineEdit(self)
            self.video_name_input.setReadOnly(True)
            self.video_name_input.setAlignment(Qt.AlignLeft)
            self.video_name_input.setStyleSheet("color: grey; font-weight: bold;")
            self.video_name_input.setFixedSize(610, 30)
            video_name_layout.addWidget(self.video_name_input)
            self.toolbar_layout.addLayout(video_name_layout)
        
        main_layout.addLayout(self.toolbar_layout)
        self.setLayout(main_layout)

        self.cur_video_idx = 1
        self.video_position_label.setText(f"Frame: -/-")
        if self.mode != '语言标注':
            self.sam_obj_pos_label.setText("Object: -/-")
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_video)

        ##############################################################
        ##################### initialize Configs #####################
        ##############################################################
        config_path = "./config/config.yaml"
        with open(self.get_exe_path(config_path), "r") as f:
            self.model_config = yaml.load(f, Loader=yaml.FullLoader)
        self.sam_config = self.model_config["sam"]
        self.co_tracker_config = self.model_config["cotracker"]

        ##############################################################
        #################### initialize Variables ####################
        ##############################################################
        self.frame_count = 0
        self.is_stop = False
        self.current_frame = 0
        self.is_first = True
        self.last_frame = None
        self.ori_video = {}
        self.vis_track_res = False
        self.sam_anno = False
        self.lang_anno = dict()
        self.max_point_num = dict()
        self.video_2_lang = dict()
        self.cur_frame_idx = self.progress_slider.value()
        self.keyframes = {} 
        self.selected_keyframe = None 
        self.key_frame_mode = 'start'
        self.sam_point_anno = dict()
        self.lang_only_anno = dict()
        self.video_path = None
        self.frame_contact = set()
        
        self.next_video_and_load(is_first=True)
        
    def change_finish_button(self):
        if self.is_hard_sample_button.isChecked():
            self.is_finished_button.setDisabled(True)
        else:
            self.is_finished_button.setDisabled(False)
    
    def change_hard_button(self):
        if self.is_finished_button.isChecked():
            self.is_hard_sample_button.setDisabled(True)
        else:
            self.is_hard_sample_button.setDisabled(False)
    
    def closeEvent(self, event):
        # 弹出一个消息框询问用户是否确认关闭窗口
        reply = QMessageBox.question(
            self, '确认退出', '你确定要退出吗？',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # 用户选择 Yes，允许关闭窗口
            event.accept()
            if self.mode == '语言标注':
                drawback_video(self.ip_address, self.port, self.video_path, 'lang')
            else:
                drawback_video(self.ip_address, self.port, self.video_path, 'sam', self.username)
        else:
            # 用户选择 No，阻止窗口关闭
            event.ignore()
    
    def check_re_anno(self):
        if self.re_annotation_button.currentText() == '0':
            self.is_pre_button.setDisabled(False)
            self.is_finished_button.show()
            self.is_hard_sample_button.hide()
        elif self.re_annotation_button.currentText() == '1':
            if not self.has_one_anno:
                self.smart_message("暂无需要一次复检的视频，请耐心等待")
                self.re_annotation_button.setCurrentIndex(0)
                return
            else:
                self.is_pre_button.setDisabled(False)
                self.is_finished_button.show()
                # self.is_hard_sample_button.hide()
                self.is_hard_sample_button.setText("问题样本")
                self.is_hard_sample_button.show()
        elif self.re_annotation_button.currentText() == '2':
            if not self.has_two_anno:
                self.smart_message("暂无需要二次复检的视频，请耐心等待")
                self.re_annotation_button.setCurrentIndex(0)
                return
            else:
                self.is_pre_button.setDisabled(False)
                self.is_finished_button.show()
                # self.is_hard_sample_button.hide()
                self.is_hard_sample_button.setText("问题样本")
                self.is_hard_sample_button.show()
        elif self.re_annotation_button.currentText() == '3':
            if not self.has_three_anno:
                self.smart_message("暂无需要三次复检的视频，请耐心等待")
                self.re_annotation_button.setCurrentIndex(0)
                return
            else:
                self.is_pre_button.setDisabled(False)
                self.is_finished_button.show()
                self.is_hard_sample_button.setText("困难样本")
                self.is_hard_sample_button.show()
              
    def get_exe_path(self, relative_path):
        try:
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = os.path.abspath(".")
    
        return os.path.normpath(os.path.join(base_path, relative_path))
      
    def set_button_text(self):
        if self.is_pre_button.isChecked():
            self.next_button.setText("加载上一个视频")
        else:
            self.next_button.setText("保存并进行下一次标注")
    
    def mode_choose(self):
        
        # 在主窗口上直接弹出对话框，选择模式
        dialog = QDialog(self)
        dialog.setWindowTitle("选择模式")
        dialog.setFixedSize(500, 300)
        # center the dialog
        # desktop = QApplication.desktop()
        # dialog.move(int(desktop.width()*0.4), int(desktop.height()*0.4))
        
        dialog_layout = QVBoxLayout()
        dialog.setLayout(dialog_layout)
        
        
        # 添加用户名字输入框
        username_layout = QHBoxLayout()
        username_label = QLabel("请输入用户名: ", self)
        username_label.setFixedSize(150, 30)
        username_layout.addWidget(username_label)
        
        user_name = QLineEdit(self)
        user_name.setPlaceholderText("请输入用户名")
        user_name.setFixedSize(275, 30)
        username_layout.addWidget(user_name)
        dialog_layout.addLayout(username_layout)
        
        ip_address_layout = QHBoxLayout()
        ip_address_label = QLabel("请输入服务器地址: ", self)
        ip_address_label.setFixedSize(150, 30)
        ip_address_layout.addWidget(ip_address_label)
        
        ip_address = QLineEdit(self)
        ip_address.setPlaceholderText("服务器地址")
        ip_address.setFixedSize(180, 30)
        ip_address_layout.addWidget(ip_address)
        
        port = QLineEdit(self)
        port.setPlaceholderText("端口地址")
        port.setFixedSize(80, 30)
        ip_address_layout.addWidget(port)
        dialog_layout.addLayout(ip_address_layout)
        
        mode_layout = QHBoxLayout()
        mode_label = QLabel("请选择标注模式: ", self)
        mode_label.setFixedSize(150, 30)
        mode_layout.addWidget(mode_label)
        
        self.mode_select = QComboBox()
        self.mode_select.addItem('语言标注')
        self.mode_select.addItem('分割标注')
        self.mode_select.setFixedSize(275, 30)
        self.mode_select.currentIndexChanged.connect(self.check_anno_mode)
        mode_layout.addWidget(self.mode_select)
        dialog_layout.addLayout(mode_layout)
        
        time_layout = QHBoxLayout()
        self.time_label = QLabel("请选择质检次数: ", self)
        self.time_label.setFixedSize(150, 30)
        time_layout.addWidget(self.time_label)
        self.time_select = QComboBox()
        self.time_select.addItem('0')
        self.time_select.addItem('1')
        self.time_select.addItem('2')
        self.time_select.addItem('3')
        self.time_select.setFixedSize(275, 30)
        time_layout.addWidget(self.time_select)
        dialog_layout.addLayout(time_layout)

        self.time_label.hide()
        self.time_select.hide()
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        dialog_layout.addWidget(button_box)
        if dialog.exec_() == QDialog.Accepted:
            if len(user_name.text()) == 0 or len(ip_address.text()) == 0:
                self.smart_message("用户名和服务器地址不能为空")
                sys.exit()
        else:
            sys.exit()
            
        while True:
            username = user_name.text().strip()
            ipaddress = ip_address.text().strip()
            ip_port = port.text().strip() 
            username = get_avaiable_username(ipaddress, ip_port, username)
            if username == '':
                self.smart_message("用户名不存在，请重新输入")
                dialog.exec_()
            else:
                break
            
        return self.mode_select.currentText(), username, ipaddress, ip_port, int(self.time_select.currentText())
    
    def check_anno_mode(self):
        if self.mode_select.currentText() == '分割标注':
            self.time_label.show()
            self.time_select.show()
        else:
            self.time_label.hide()
            self.time_select.hide()

    def pre_sam_object(self):
        if self.sam_object_id[self.progress_slider.value()] > 0:
            self.sam_object_id[self.progress_slider.value()] -= 1
        if self.sam_object_id[self.progress_slider.value()] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"标注物体: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def resizeEvent(self, event):
        self.seek_video()
        self.clear_keyframes()
        self.update_keyframe_bar()
        
        self.setAutoFillBackground(False)
        palette = self.palette()
        # palette.setBrush(self.backgroundRole(), QBrush(QPixmap('./demo/bg.png').scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))
        self.setPalette(palette)
        # if  len(self.ori_video)>0 and self.mode == '语言标注':
        self.keyframe_bar.show()
        # else:
        #     self.keyframe_bar.hide()
    
    def next_sam_object(self):
        
        self.sam_object_id[self.progress_slider.value()] += 1
        if self.sam_object_id[self.progress_slider.value()] == len(self.tracking_points_sam[self.progress_slider.value()]):
            self.tracking_points_sam[self.progress_slider.value()].append(
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            )
        self.sam_pre_button.setDisabled(False)
        
        if len(self.tracking_points_sam[self.progress_slider.value()][self.sam_object_id[self.progress_slider.value()]]['pos']) == 0:
            self.sam_next_button.setDisabled(True)
        else:
            self.sam_next_button.setDisabled(False)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"Object ID: {cur_id}/{all_object_size}")
        
        self.draw_image()
    
    def next_video_and_load(self, is_first=False):
        if is_first:
            self.load_video_async()
        else:            
            if self.has_anno() and not self.is_pre_button.isChecked():
                reply = QMessageBox.question(
                    self,
                    "提示",
                    "确认保存并加载下一个视频？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                if reply == QMessageBox.No:
                    return
                self.video_position_label.setText(f"帧: -/-")
                if self.mode == '语言标注':
                    res = self.save_lang_anno()
                else:
                    res = self.save_sam_anno()
                
                if res == -1:
                    return
                self.clear_video()
                self.load_video_async()
            
            elif self.is_pre_button.isChecked():
                self.clear_video()
                self.load_video_async()
            
            elif self.is_finished_button.isChecked() and self.mode != '语言标注': 
                res = self.save_sam_anno()
                if res == -1:
                    return
                self.load_video_async()
            
            elif self.is_hard_sample_button.isChecked() and self.mode != '语言标注':
                res = self.save_sam_anno()
                if res == -1:
                    return
                self.load_video_async()
            
            else:
                self.smart_message("请先完成当前视频的标注")

        return

    def has_anno(self):
        if self.mode == '语言标注':
            for i in self.lang_anno:
                if i != (0, 0):
                    return True
            return False
        else:
            for i in self.tracking_points_sam:
                if len(self.tracking_points_sam[i][0]['pos']) > 0:
                    return True
            return False
    
    def next_frame(self):
        if self.cur_frame_idx < self.frame_count - 1:
            self.cur_frame_idx += 1
        else:
            return
        self.video_position_label.setText(f"帧: {self.cur_frame_idx}/{self.frame_count}")
        self.sam_object_id[self.cur_frame_idx] = 0
        
        if self.sam_object_id[self.cur_frame_idx] == 0:
            self.sam_pre_button.setDisabled(True)     
        
        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"标注物体: {cur_id}/{all_object_size}")   
        
        if (0, 0) in self.lang_anno:
            desc = self.lang_anno[(0, 0)]
            self.video_lang_input.setText(f"视频描述: {desc}")
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim, origin_text) = self.get_clip_description()
        anno_loc = anno_loc[1]
        if anno_loc is not None:
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {clip_text}")
        else:
            self.clip_lang_input.setText('')
        
    def pre_frame(self):
        if self.cur_frame_idx >= 1:
            self.cur_frame_idx -= 1
        else:
            return
        self.video_position_label.setText(f"帧: {self.cur_frame_idx+1}/{self.frame_count}")
        self.sam_object_id[self.cur_frame_idx] = 0
        
        if self.sam_object_id[self.cur_frame_idx] == 0:
            self.sam_pre_button.setDisabled(True)
        
        self.sam_next_button.setDisabled(False)
        self.update_frame(self.cur_frame_idx)
        self.progress_slider.setValue(self.cur_frame_idx)
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"物体标注: {cur_id}/{all_object_size}")
        
        if (0, 0) in self.lang_anno:
            desc = self.lang_anno[(0, 0)]
            self.video_lang_input.setText(f"视频描述: {desc}")
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim, origin_text) = self.get_clip_description()
        anno_loc = anno_loc[1]
        if anno_loc is not None:
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {clip_text}")
        else:
            self.clip_lang_input.setText('') 
    
    def request_video(self):
        # try:
        if self.is_pre_button.isChecked():
            self.button_mode = 'pre'
            self.is_pre_button.setChecked(False)
            self.is_pre_button.setDisabled(True)
        else:
            if not self.is_first:
                self.is_pre_button.setDisabled(False)
            self.is_pre_button.setChecked(False)
            self.button_mode = 'next'
        if self.mode == '语言标注':
            return request_video_and_anno(self.ip_address, self.port, 'lang', self.username, self.button_mode, self.video_path)
        else:
            re_anno = int(self.re_annotation_button.currentText())
            return request_video_and_anno(self.ip_address, self.port, 'sam', self.username, self.button_mode, self.video_path, re_anno)
        
        # except Exception as e:
        #     if self.mode == '语言标注':
        #         return None
        #     else:
        #         return None
               
    def request_video_async(self):
        class VideoThread(QThread):
            finished = pyqtSignal(object)
            def __init__(self, parent):
                super().__init__()
                self.parent = parent
            def run(self):
                res = self.parent.request_video()
                try_count = 0
                while res is None:
                    if try_count > 3:
                        break
                    time.sleep(3)
                    res = self.parent.request_video()
                    try_count += 1
                if try_count > 3:
                    raise Exception("请求视频失败")
                self.finished.emit(res)
        video_thread = VideoThread(self)
        return video_thread
       
    def save_lang_anno(self):
        self.progress = QProgressDialog("请等待，正在储存标注结果...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        #################### parse video language annotation ####################
        lang_res = dict()
        if (0, 0) in self.lang_anno and self.lang_anno[(0, 0)] != '':
            lang_res['video'] = self.lang_anno[(0, 0)]
        else:
            self.progress.close()
            self.smart_message("请标注整体视频的语言描述")
            return 
        #################### parse clip language annotation ####################
        lang_res['clip'] = []
        for clip_range, lang in self.lang_anno.items():
            if clip_range == (0, 0):
                continue
            if len(lang) == 0:
                self.smart_message(f"请完成帧{clip_range[0]}到帧{clip_range[1]}之前的语言标注")
                return
            tmp_lang = dict()
            tmp_lang['start_frame'] = clip_range[0]
            tmp_lang['end_frame'] = clip_range[1]
            tmp_lang['description'] = lang
            lang_res['clip'].append(tmp_lang)
        lang_res['user'] = self.username
        lang_res['video_path'] = self.video_path
        lang_res['button_mode'] = self.button_mode
        lang_res['frames'] = self.progress_slider.maximum()

        res = save_anno(self.ip_address, self.port, self.save_path, lang_res)
        try_time = 0
        while not res:
            if try_time > 3:
                break
            self.smart_message("保存失败，进行第{}次重试".format(try_time+1))
            time.sleep(4)
            res = save_anno(self.ip_address, self.port, self.save_path, lang_res)
            try_time += 1
        
        if not res:
            self.progress.close()
            self.smart_message("保存失败，即将退出系统重启软件")
            drawback_video(self.ip_address, self.port, self.video_path, 'lang')
            sys.exit()
            
        self.progress.close()
        self.lang_anno = dict()
        return 0
    
    def clear_annotations(self):
        self.tracking_points_sam = dict()
        for k in range(self.frame_count):
            self.tracking_points_sam[k] = [
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            ]
        self.lang_anno = dict()
        self.sam_next_button.setDisabled(False)
        self.sam_pre_button.setDisabled(True)
        self.sam_object_id[self.progress_slider.value()] = 0
        self.sam_obj_pos_label.setText("标注物体: 1/1")
        self.sam_frame_id = set()
        
        if self.last_frame is not None:
            self.draw_image()

    def clear_keyframes(self):
        self.keyframes = {}
        self.lang_anno = dict()
        if self.last_frame is not None:
            self.draw_image()
        self.update_keyframe_bar()
           
    def clear_video(self):
        self.video_label.clear()
        self.progress_slider.setValue(0)
        self.frame_position_label.hide()
        self.keyframes = {}
        self.ori_video = {}
        self.selected_keyframe = None
        self.update_keyframe_bar()
        self.keyframe_bar.hide()
        self.last_frame = None
        self.cur_frame_idx = 0
        self.current_frame = 0
        self.sam_object_id = [0] * self.frame_count
        self.max_point_num = dict()
        # self.vis_ori.setChecked(True)
        self.lang_anno = dict()
        self.video_lang_input.clear()
        self.clip_lang_input.clear()
        self.video_position_label.setText(f"Frame: -/-")
    
    def remove_last_sam_annotation(self):
        if len(self.tracking_points_sam) == 0:
            self.smart_message("请先标注!")
            return
        
        sam_object_id = self.sam_object_id[self.progress_slider.value()]
        click_action = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels']
        pos_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos']
        neg_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg']
        
        if len(click_action) > 0 and click_action[-1] == 1 and len(pos_click_position) > 0:
            if len(pos_click_position) > 0:
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos'].pop()
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_pos'].pop()
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].pop()
        elif len(click_action) > 0 and click_action[-1] == -1 and len(neg_click_position) > 0:
            if len(neg_click_position) > 0:
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg'].pop()
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_neg'].pop()
                self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].pop()
        
        if len(self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos']) == 0 and len(self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg']) == 0:
            if self.progress_slider.value() in self.sam_frame_id:
                self.sam_frame_id.remove(self.progress_slider.value())
            cur_obj_id = self.sam_object_id[self.progress_slider.value()]
            if cur_obj_id > 0:
                self.tracking_points_sam[self.progress_slider.value()].pop(cur_obj_id)
                self.sam_object_id[self.progress_slider.value()] -= 1
        
            cur_id = self.sam_object_id[self.progress_slider.value()] + 1
            all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
            self.sam_obj_pos_label.setText(f"物体标注: {cur_id}/{all_object_size}")
            
        if self.last_frame is not None:
            self.draw_image()
     
    def remove_last_annotation(self):
        self.remove_last_sam_annotation()
    
    def remove_obj_annotation(self):
        if len(self.tracking_points_sam) == 0:
            self.smart_message("请先加载视频!")
            return
        cur_obj_id = self.sam_object_id[self.progress_slider.value()]
        self.tracking_points_sam[self.progress_slider.value()].pop(cur_obj_id)
        
        if cur_obj_id > 0:
            self.sam_object_id[self.progress_slider.value()] -= 1
        else:
            self.sam_object_id[self.progress_slider.value()] = 0
        
        if len(self.tracking_points_sam[self.progress_slider.value()]) == 0:
            self.tracking_points_sam[self.progress_slider.value()].append(
                dict(pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[])
            )
            self.sam_object_id[self.progress_slider.value()]=0
            if self.progress_slider.value() in self.sam_frame_id:
                self.sam_frame_id.remove(self.progress_slider.value())
        
        cur_id = self.sam_object_id[self.progress_slider.value()] + 1
        all_object_size = len(self.tracking_points_sam[self.progress_slider.value()])
        self.sam_obj_pos_label.setText(f"标注物体: {cur_id}/{all_object_size}")
        
        if self.last_frame is not None:
            self.draw_image()
    
    def load_video_callback(self, res):
        if res == 0:
            self.smart_message("暂无需要标注的视频，请耐心等待")
            sys.exit()
        
        if self.mode != '语言标注':
            video, save_path, video_path, hist_num, \
                one_anno_num, all_one_anno_num, two_anno_num, all_two_anno_num, three_anno_num, all_three_anno_num = res
        else:
            video, lang, save_path, video_path, hist_num = res
            if lang['has_ori_instruction']:
                self.video_2_lang = lang['annotation']
                task_stepsC = self.video_2_lang['task_stepsC'].copy()
                self.video_2_lang['task_stepsC'] = dict()
                for i in (task_stepsC + BASE_CLIP_DES):
                    self.video_2_lang['task_stepsC'][i] = None
                self.video_2_lang['task_stepsC_list'] = task_stepsC
            else:
                self.video_2_lang = dict(
                    task_stepsC=dict(),
                    instructionC='',
                    action_stepsC=[],
                    task_stepsC_list = []
                )
        self.save_path = save_path
        self.video_path = video_path
        self.hist_num = hist_num
        self.hist_num_label.setText(f"Annotated Num: {self.hist_num}")
        if self.mode != '语言标注':
            self.two_anno_num = two_anno_num
            self.one_anno_num = one_anno_num
            self.three_anno_num = three_anno_num
            self.all_one_anno_num = all_one_anno_num
            self.all_two_anno_num = all_two_anno_num
            self.all_three_anno_num = all_three_anno_num
            self.one_num_label.setText(f"Check One(Available/Remaining): {self.one_anno_num}/{self.all_one_anno_num}")
            self.two_num_label.setText(f"Check Two(Available/Remaining): {self.two_anno_num}/{self.all_two_anno_num}")
            self.three_num_label.setText(f"Chcek Three(Available/Remaining): {self.three_anno_num}/{self.all_three_anno_num}")
            self.has_one_anno = one_anno_num > 0
            self.has_two_anno = two_anno_num > 0
            self.has_three_anno = three_anno_num > 0
            self.video_name_input.setText(video_path.split('/')[-1].split('.')[0])
        
        if video is not None:
            self.load_video(video)
            self.progress.close()
        else:
            self.progress.close()
            self.smart_message("视频加载失败，请检查网络设置")
            return
        
    def load_video_async(self):
        self.progress = QProgressDialog("请等待，正在加载视频...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)  # 立即显示对话框
        self.progress.show()

        self.video_thread = self.request_video_async()
        self.video_thread.finished.connect(self.load_video_callback)
        self.video_thread.start()
    
    def load_video(self, video):
        if video is None:
            return -1
        self.frame_count = video.shape[0]
        self.sam_object_id = [0] * self.frame_count
        self.lang_anno = dict()
        self.tracking_points_sam = dict()
        self.ori_video = np.array(video)
        self.sam_obj_pos_label.setText("标注物体: 1/1")
        
        for i in range(self.frame_count):
            self.tracking_points_sam[i] = [dict(
                    pos=[], raw_pos=[], neg=[], raw_neg=[], labels=[]
            )]
        self.sam_object_id = [0] * self.frame_count

        self.progress_slider.setMaximum(self.frame_count - 1)
        self.progress_slider.show()
        self.frame_position_label.show()
        self.update_keyframe_bar()
        self.update_frame(0)
        self.progress_slider.setValue(0)
        # if self.mode == '语言标注':
        self.keyframe_bar.show()
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx+1}/{self.frame_count}")
        # self.pre_f_button.setDisabled(True)
        self.max_point_num = 0
        self.hist_num = dict()
        self.sam_frame_id = set()
        self.start_frame_to_object_id = dict()
        self.frame_contact = set()
        self.use_L = False
        
        if self.mode != '语言标注':
            self.check_re_anno()
            self.button_param_select.setCurrentIndex(0)
            self.is_finished_button.setChecked(False)
            self.is_finished_button.setDisabled(False)
            self.is_hard_sample_button.setChecked(False)
            self.is_hard_sample_button.setDisabled(False)
        
        self.seek_video()
        # for align the keyframe display length
        if 0 not in self.keyframes and self.is_first:
            self.mark_keyframe(debug=True)
            self.selected_keyframe = self.progress_slider.value()
            self.remove_keyframe()
            self.is_first = False
        
        # load global language annotation
        if self.mode == '语言标注':
            video_anno_json = self.video_2_lang
            # TODO check exist
            global_instruction_C = video_anno_json['instructionC'] if 'instructionC' in video_anno_json else ''
            if (0, 0) not in self.lang_anno or self.lang_anno[(0, 0)] == '':
                self.lang_anno[(0, 0)] = global_instruction_C

            desc = self.lang_anno[(0, 0)]
            self.video_lang_input.setText(f"视频描述: {desc}")
            self.preview_clip_lang_input.setText(self.get_clip_lang_anno())
        
        return 1
            
    def update_frame(self, frame_number):
        if  len(self.ori_video) == 0:
            self.smart_message('请先加载视频！')
            return
        frame = self.ori_video[frame_number]
        self.height, self.width, channel = frame.shape

        # Scale the image to fit QLabel
        label_width = self.video_label.width()
        label_height = self.video_label.height()
        self.scale_width = label_width / self.width
        self.scale_height = label_height / self.height
        self.scale = min(self.scale_width, self.scale_height)
        new_width = int(self.width * self.scale)
        new_height = int(self.height * self.scale)

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Update and reposition frame position label
        self.update_frame_position_label()
        self.last_frame = resized_frame
        
        self.draw_image()
                
    def seek_video(self):
        if self.last_frame is None:
            return 
        frame_number = self.progress_slider.value()
        self.update_frame(frame_number)
        self.cur_frame_idx = self.progress_slider.value()
        self.video_position_label.setText(f"Frame: {self.cur_frame_idx+1}/{self.frame_count}")
        
        self.sam_object_id[self.cur_frame_idx] = 0
        self.sam_obj_pos_label.setText(f"Object ID: {self.sam_object_id[self.cur_frame_idx]+1}/{len(self.tracking_points_sam[self.cur_frame_idx])}")
        self.sam_pre_button.setDisabled(True)
        self.sam_next_button.setDisabled(False)
        
        if (0, 0) in self.lang_anno:
            desc = self.lang_anno[(0, 0)]
            self.video_lang_input.setText(f"视频描述: {desc}")
        else:
            self.video_lang_input.setText('')
        
        anno_loc, (clip_text, prim, origin_text) = self.get_clip_description()
        anno_loc = anno_loc[1]
        if anno_loc is not None:
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {clip_text}")
        else:
            self.clip_lang_input.clear()
            
    def toggle_playback(self):
        if self.play_button.isChecked():
            self.play_button.setText("Pause")
            self.current_frame = self.progress_slider.value()
            self.timer.start(30)  # Set timer to update frame every 30 ms
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def update_frame_position_label(self):
        # Update the text of the label to show the current frame position
        frame_number = self.progress_slider.value()
        # check if the frame number has keyframe
        if frame_number in self.keyframes:
            keyframe_type = self.keyframes[frame_number]
            # keyframe_type = '开始' if keyframe_type.lower() == 'start' else '结束'
            # keyframe_type = keyframe_type if 
            if keyframe_type.lower() == 'start':
                keyframe_type = 'Start'
            elif keyframe_type.lower() == 'end':
                keyframe_type = 'End'
            elif keyframe_type.lower() == 'object_sep':
                keyframe_type = 'Sep'
            elif keyframe_type.lower() == 'object_contact':
                keyframe_type = 'Contact'
            
            self.frame_position_label.setText(f"Frame: {frame_number+1}({keyframe_type})")
        else:
            self.frame_position_label.setText(f"Frame: {frame_number+1}")

        # Calculate the position for the label above the slider handle
        slider_x = self.progress_slider.x()
        slider_width = self.progress_slider.width()
        # print(self.progress_slider.maximum(), self.progress_slider.minimum())
        value_ratio = frame_number / (self.progress_slider.maximum() - self.progress_slider.minimum())
        label_x = slider_x + int(value_ratio * slider_width) - self.frame_position_label.width() // 2
        
        # Set the position of the label
        label_y = self.progress_slider.y() - self.frame_position_label.height() - 4
        label_x = max(slider_x, min(label_x, slider_x + slider_width - self.frame_position_label.width()))
        self.frame_position_label.move(label_x, label_y)
        self.frame_position_label.show()  # Show the label

    def get_frame_position(self):
        current_position = self.progress_slider.value()
        return current_position
    
    def play_video(self):
        # if self.cap is not None:
        if self.current_frame < self.frame_count - 1:
            self.current_frame += 1
            self.update_frame(self.current_frame)
            self.progress_slider.setValue(self.current_frame)
        else:
            self.timer.stop()
            self.play_button.setChecked(False)
            self.play_button.setText("播放")
    
    # 自动播放视频，再点击停止播放
    def autoplayorstop(self):
        if self.is_stop:
            self.play_button.setText("暂停")
            self.current_frame = self.progress_slider.value()
            self.timer.start(30)
        else:
            self.play_button.setText("播放")
            self.timer.stop()

    def set_sam_config(self):
        tracking_points = self.tracking_points_sam
        select_frames = list(self.sam_frame_id)
        self.sam_config['video_path'] = self.video_path
        # print(self.video_path)
        self.sam_config['user'] = self.username
        self.sam_config["hard_sample_type"] = 'normal'
        
        if self.is_hard_sample_button.isChecked() and self.is_finished_button.isChecked():
            self.smart_message("困难样本不可标注为完成，请检查")
            return -1
        
        if self.is_finished_button.isChecked():
            self.sam_config['is_finished'] = True
            self.sam_config['is_hard_sample'] = False
            self.sam_config["hard_sample_type"] = self.is_hard_sample_button.text() 
            return 0
        
        if self.is_hard_sample_button.isChecked():
            self.sam_config['is_hard_sample'] = True
            self.sam_config['is_finished'] = False
            self.sam_config["hard_sample_type"] = self.is_hard_sample_button.text()
            return 0
        
        if len(select_frames) == 0:
            self.smart_message("请先标注物体")
            return -1
        
        if not self.use_L:
            self.smart_message("完成标注物体后，请按下L键确认物体绑定顺序")
            return -1
        
        if len(select_frames) > 1 and self.button_param_select.currentText() != '双向视频模式':
            self.smart_message("多帧模式仅支持双向视频模式，请检查")
            return -1
        
        for i in select_frames[1:]:
            if len(tracking_points[i]) != len(tracking_points[select_frames[0]]):
                self.smart_message(f"第{i}帧标注物体数量与第0帧不匹配, 请检查")
                return -1
        
        objects = list(set([i for i in self.start_frame_to_object_id.values()]))
        if len(tracking_points[select_frames[0]]) > 0 and len(tracking_points[select_frames[0]]) != len(objects):
            self.smart_message("标注物体数量与关键帧数量不匹配, 请检查")
            return -1
        
        if self.align_contact_frame_with_object_sep() == -1:
            return -1
        
        positive_points_all = {}
        negative_points_all = {}
        labels_all = {}
        
        if self.button_param_select.currentText() == 'Bidirectional Mode':
            direction = 'bidirection' 
        elif self.button_param_select.currentText() == 'Forward Mode':
            direction = 'forward'
        else:
            direction = 'backward'
        is_video = True
        
        for select_frame in select_frames:
            positive_points_all[select_frame] = {}
            negative_points_all[select_frame] = {}
            labels_all[select_frame] = {}
            frame_pts = tracking_points[select_frame]
            # select all objects
            for obj_id, obj_pts in enumerate(frame_pts):
                positive_points, negative_points, labels = [], [], []
                if obj_pts['raw_pos'] != []:
                    positive_points.extend([[pt.x(), pt.y()] for pt in obj_pts['raw_pos']])
                if obj_pts['raw_neg'] != []:
                    negative_points.extend([[pt.x(), pt.y()] for pt in obj_pts['raw_neg']])
                if (obj_pts['raw_pos'] != []) or (obj_pts['raw_neg'] != []):
                    labels.extend(obj_pts['labels'])
                
                positive_points_all[select_frame][obj_id] = positive_points
                negative_points_all[select_frame][obj_id] = negative_points
                labels_all[select_frame][obj_id] = labels
        
        self.sam_config['is_video'] = is_video
        self.sam_config['direction'] = direction
        self.sam_config['positive_points'] = positive_points_all
        self.sam_config['negative_points'] = negative_points_all
        self.sam_config['labels'] = labels_all
        self.sam_config['select_frames'] = select_frames
        self.sam_config['button_mode'] = self.button_mode
        self.sam_config['start_frame_2_obj_id'] = self.start_frame_to_object_id
        self.sam_config['frame_contact_2_obj_id'] = self.frame_contact_to_object_id
        self.sam_config['is_finished'] = False
        self.sam_config['is_hard_sample'] = False
        
        print(self.start_frame_to_object_id)
        print(self.frame_contact_to_object_id)
        
        return 0
    
    def smart_message(self, message, auto_close=True):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle('提示')
        msg.setText(message)
        msg.exec_()
    
    def save_sam_anno(self):
        res = self.set_sam_config()
        if res == -1:
            return -1
        self.progress = QProgressDialog("请等待，正在储存标注结果...", None, 0, 0, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.setMinimumDuration(0)
        self.progress.show()
        
        saved_config = self.sam_config.copy()
        saved_config['video_path'] = self.video_path
        res = save_anno(self.ip_address, self.port, self.save_path, self.sam_config.copy())
        try_time = 0
        while not res:
            if try_time > 3:
                break
            self.smart_message("保存失败，进行第{}次重试".format(try_time+1))
            time.sleep(4)
            res = save_anno(self.ip_address, self.port, self.save_path, self.sam_config.copy())
            try_time += 1
        
        if not res:
            self.progress.close()
            self.smart_message("保存失败，即将退出系统重启软件")
            drawback_video(self.ip_address, self.port, self.video_path, 'sam')
            sys.exit()
        
        self.progress.close()
        return 0
              
    def mousePressEvent(self, event: QMouseEvent):        
        if self.last_frame is None:
            return
        
        if self.mode == '语言标注':
            return
        
        if event.button() == Qt.LeftButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            gt_pos = self.get_align_point(pos.x(), pos.y())
            if gt_pos is None:
                return
            click_position = QPoint(gt_pos[0], gt_pos[1])
            original_position = QPoint(int(gt_pos[0]//self.scale), int(gt_pos[1]//self.scale))      
            sam_object_id = self.sam_object_id[self.progress_slider.value()]
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_pos'].append(original_position)
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos'].append(click_position)
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].append(1)
            self.sam_next_button.setDisabled(False)
            self.sam_frame_id.add(self.progress_slider.value())
                
        elif event.button() == Qt.RightButton and self.last_frame is not None:
            pos = self.video_label.mapFromGlobal(event.globalPos())
            gt_pos = self.get_align_point(pos.x(), pos.y())
            if gt_pos is None:
                return
            click_position = QPoint(gt_pos[0], gt_pos[1])
            original_position = QPoint(int(gt_pos[0]//self.scale), int(gt_pos[1]//self.scale))
            sam_object_id = self.sam_object_id[self.progress_slider.value()]
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg'].append(click_position)
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['raw_neg'].append(original_position)
            self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['labels'].append(-1)
            self.sam_next_button.setDisabled(False)
            self.sam_frame_id.add(self.progress_slider.value())
            
        self.draw_image()
    
    def get_align_point(self, x, y): 
        label_height, label_width = self.video_label.height(), self.video_label.width()
        resized_width = int(self.width * min(self.scale_width, self.scale_height))
        resized_height = int(self.height * min(self.scale_width, self.scale_height))
        offset_x = (label_width - resized_width) // 2
        offset_y = (label_height - resized_height) // 2
        x -= offset_x
        y -= offset_y
        
        gt_shape = self.last_frame.shape
        if x < 0 or y < 0 or x >= gt_shape[1] or y >= gt_shape[0]:
            return None
        
        return (x, y)
     
    def draw_image(self):
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()

        if self.mode != '语言标注':
            sam_object_id = self.sam_object_id[self.progress_slider.value()]
            if self.progress_slider.value() in self.tracking_points_sam:
                pos_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['pos']
                neg_click_position = self.tracking_points_sam[self.progress_slider.value()][sam_object_id]['neg']
            else:
                pos_click_position, neg_click_position = [], []
        else:
            pos_click_position, neg_click_position = [], []

        for point in pos_click_position:
            x, y = point.x(), point.y()
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        for point in neg_click_position:
            x, y = point.x(), point.y()
            cv2.circle(frame, (x, y), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        if len(pos_click_position)==0 and len(neg_click_position)==0:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def submit_description(self):
        self.add_video_description()

    def mark_keyframe(self, debug=False):
        current_frame = self.progress_slider.value()
        if self.key_frame_mode == 'start':
            # check if the last keyframe is 'end'
            if len(self.keyframes) > 0 and list(self.keyframes.values())[-1] == 'start':
                if not debug:
                    # self.is_stop = False
                    # self.autoplayorstop()
                    pass
                self.smart_message('请标注结束帧')
                return -1
            
            if len(self.keyframes) > 0:
                for i in list(self.lang_anno.keys()):
                    if i[0] <= current_frame and i[1] >= current_frame and i[0] != i[1]:
                        if not debug:
                            # self.is_stop = False
                            # self.autoplayorstop()
                            pass
                        self.smart_message('请勿在上一个视频段中标记')
                        return -1
            
            if current_frame in self.keyframes and self.keyframes[current_frame] == 'end':
                if not debug:
                    # self.is_stop = False
                    # self.autoplayorstop()
                    pass
                self.smart_message('请勿重复标记')
                return -1
            if not self.is_first:
                self.keyframes[current_frame] = 'start'
            self.update_keyframe_bar()
            self.update_frame_position_label()
            if not debug:
                # self.is_stop = True
                # self.autoplayorstop()
                pass
            
            self.last_start_frame = current_frame
        
        elif self.key_frame_mode == 'End':
            # check if the last keyframe is 'start'
            if len(self.keyframes) > 0 and list(self.keyframes.values())[-1] == 'end':
                if not debug:
                    # self.is_stop = False
                    # self.autoplayorstop()
                    pass
                self.smart_message('请标记开始帧')
                return -1
            
            if current_frame in self.keyframes and self.keyframes[current_frame] == 'start':
                if not debug:
                    # self.is_stop = False
                    # self.autoplayorstop()
                    pass
                self.smart_message('请勿重复标记')
                return -1
            
            if len(self.keyframes) > 0:
                for i in list(self.lang_anno.keys()):
                    if i[0] <= current_frame and i[1] >= current_frame:
                        if not debug:
                            # self.is_stop = False
                            # self.autoplayorstop()
                            pass
                        self.smart_message('请勿在上一个视频段中标记')
                        return -1

                    if self.last_start_frame <= i[0] and i[1] <= current_frame and i[0] != 0:
                        if not debug:
                            # self.is_stop = False
                            # self.autoplayorstop()
                            pass
                        self.keyframes.pop(self.last_start_frame)
                        self.update_keyframe_bar()
                        self.last_start_frame = None
                        self.smart_message('中间包含其他视频段')
                        return -1
            
            if len(self.keyframes) == 0:
                self.smart_message('请先标记开始帧')
                if not debug:
                    # self.is_stop = False
                    # self.autoplayorstop()
                    pass
                return -1
            
            self.keyframes[current_frame] = 'end'
            self.update_keyframe_bar()
            self.update_frame_position_label()
            if not debug:
                # self.is_stop = False
                # self.autoplayorstop()
                pass
            self.add_frame_discribtion()
        
        return 0
    
    def update_lang_anno(self):
        key_frame_list = sorted(self.keyframes.keys())
        key_pairs = []
        if len(key_frame_list) <= 1:
            self.smart_message('请先标记关键帧')
            return -1
        
        if len(key_frame_list) % 2 != 0:
            self.smart_message('请检查关键帧标记是否正确，必须是start和end交替出现')
            return -1
        
        for i in range(0, len(key_frame_list), 2):
            start_frame = key_frame_list[i]
            end_frame = key_frame_list[i+1]
            if self.keyframes[start_frame] != 'start' or self.keyframes[end_frame] != 'end':
                self.smart_message('请检查关键帧标记是否正确，必须是start和end交替出现')
                return -1
            key_pairs.append((start_frame, end_frame))
            
        for i in key_pairs:
            if i not in self.lang_anno:
                self.lang_anno[i] = (None, None, None)

    def remove_keyframe(self):
        if self.selected_keyframe is not None:
            frame_to_remove = self.selected_keyframe
            if frame_to_remove in self.keyframes:
                if self.keyframes[frame_to_remove] == 'object_contact':
                    self.frame_contact.remove(frame_to_remove)
                self.keyframes.pop(frame_to_remove)
                self.update_keyframe_bar()

    def update_keyframe_bar(self):
        # Clear the keyframe bar
        keyframe_image = QImage(self.keyframe_bar.width(), self.keyframe_bar.height(), QImage.Format_RGB32)
        keyframe_image.fill(Qt.gray)

        painter = QPainter(keyframe_image)
        for frame, key_type in self.keyframes.items():
            x_position = int((frame / self.frame_count) * self.keyframe_bar.width())
            color = QColor('red') if key_type == 'start' else QColor('blue')
            color = color if key_type != 'object_sep' else QColor('black')
            color = color if key_type != 'object_contact' else QColor('green')
            painter.fillRect(QRect(x_position, 0, 5, self.keyframe_bar.height()), color)
        painter.end()

        # Set the updated image to the QLabel
        self.keyframe_bar.setPixmap(QPixmap.fromImage(keyframe_image))
    
    def keyPressEvent(self, event):   
        key = event.key()
        if key == Qt.Key_A:
            self.pre_frame()
        elif key == Qt.Key_D:
            self.next_frame()
        elif key == Qt.Key_W and self.mode == '语言标注':
            self.key_frame_mode = 'start'
            self.is_stop = True
            self.mark_keyframe()
        elif key == Qt.Key_S and self.mode == '语言标注':
            self.key_frame_mode = 'End'
            self.is_stop = False
            self.mark_keyframe()
        elif key == Qt.Key_S and self.mode == '分割标注':
            self.add_object_sep()
            self.update_frame_position_label()
        elif key == Qt.Key_W and self.mode == '分割标注':
            self.add_object_contact_frame()
            self.update_frame_position_label()
            self.frame_contact.add(self.progress_slider.value())
        elif key == Qt.Key_Backspace and self.mode == '语言标注':
            self.selected_keyframe = self.progress_slider.value()
            # self.remove_keyframe()
            self.delete_keyframe()
            self.update_frame_position_label()
        elif key == Qt.Key_Backspace and self.mode == '分割标注':
            self.selected_keyframe = self.progress_slider.value()
            self.remove_keyframe()
            self.update_frame_position_label()
        elif key == Qt.Key_Return and self.mode == '语言标注':
            self.submit_description()
        elif key == Qt.Key_F:
            self.is_stop = not self.is_stop
            self.autoplayorstop()
        elif key == Qt.Key_L and self.mode == '语言标注':
            self.add_frame_discribtion()
        elif key == Qt.Key_L and self.mode == '分割标注':
            self.add_object_sep_2_id()
        elif key == Qt.Key_P:
            self.next_button.click()
        elif key == Qt.Key_E:
            self.sam_next_button.click()
    
    def add_object_sep(self):
        self.keyframes[self.progress_slider.value()] = 'object_sep'
        self.update_keyframe_bar()
    
    def add_object_contact_frame(self):
        self.keyframes[self.progress_slider.value()] = 'object_contact'
        self.update_keyframe_bar()
    
    def add_object_sep_2_id(self):
        self.use_L = True
        self.start_frame_to_object_id = dict()
        object_size = max([len(i) for i in self.tracking_points_sam.values()])
        key_frames = [k for (k, v) in self.keyframes.items() if v == 'object_sep']
        
        if object_size > 1:
            # if object_size != len(key_frames) + 1:
            #     self.smart_message('关键帧分割和物体标注数量不一致！')
            #     return
            dialog = ObjectAnnotationDialog(object_size, self.frame_count, key_frames, self)
            while True:
                if dialog.exec_() == QDialog.Accepted:
                    try:
                        self.start_frame_to_object_id = dialog.get_result()
                        break
                    except:
                        self.smart_message('有视频段未标注，请检查')
                        continue
                else:
                    return
        elif object_size == 1:
            # if len(key_frames) > 0:
            #     self.smart_message('关键帧分割和物体标注数量不一致！')
            #     return
            dialog = ObjectAnnotationDialog(object_size, self.frame_count, key_frames, self)
            while True:
                try:
                    if dialog.exec_() == QDialog.Accepted:
                        self.start_frame_to_object_id = dialog.get_result()
                        break
                except:
                    self.smart_message('有视频段未标注，请检查')
                    continue
                else:
                    return
            
        else:
            self.smart_message('无物体标注')
            return
    
    def align_contact_frame_with_object_sep(self):
        self.frame_contact_to_object_id = dict()
        # if len(self.frame_contact) != len(self.start_frame_to_object_id):
        #     print(self.frame_contact, self.start_frame_to_object_id)
        #     self.smart_message('接触帧数量和物体标注数量不一致')
        #     return -1

        self.frame_contact_to_object_id = dict()
        frame_contact_ids = sorted(list(self.frame_contact))
        sep_frame_ids = sorted(list(self.start_frame_to_object_id.keys()))
        
        for c_id, s_id in zip(frame_contact_ids, sep_frame_ids):
            self.frame_contact_to_object_id[c_id] = self.start_frame_to_object_id[s_id]
        
        return 0
    
    def delete_keyframe(self):
        info, lang = self.get_clip_description()
        if info[0] is not None:
            # remove the description
            idx, loc = info
            self.lang_anno.pop(loc)
            self.keyframes.pop(loc[0])
            self.keyframes.pop(loc[1])
            for i in self.video_2_lang['task_stepsC']:
                if lang[0] is None:
                    continue
                if self.video_2_lang['task_stepsC'][i] == lang[0]:
                    self.video_2_lang['task_stepsC'][i] = None
            self.update_keyframe_bar()
            self.clip_lang_input.clear()
            self.selected_keyframe = None
        
    def add_frame_discribtion(self):
        frame_number = self.progress_slider.value()
        if self.update_lang_anno() == -1:
            return
        key_pairs = list(self.lang_anno.keys())
        has_key = [i[0] <= frame_number <= i[1] for i in key_pairs].count(True) > 0
        if not has_key:
            self.smart_message('请先标记当前所在区域的起止帧')
            return

        # load the cached description
        anno_loc = [(idx, i) for idx, i in enumerate(key_pairs) if i[0] <= frame_number <= i[1] and i[0] != i[1]]
        if (0, 0) in self.lang_anno:
            anno_loc = [(i[0], i[1]) for i in anno_loc]
        
        if len(anno_loc) == 0:
            self.smart_message('请移动到所在区域的起止帧之间')
            return
        anno_id, anno_loc = anno_loc[0]
        
        if self.lang_anno[anno_loc][0] is not None:
            cached_lang, prim, origin_text = self.lang_anno[anno_loc]
        else:
            cached_lang, prim, origin_text = '', '', ''
        # Create a dialog to get the description from the user
        dialog = TextInputDialog(cached_lang, self, False, self.video_2_lang, origin_text=origin_text)
        if dialog.exec_() == QDialog.Accepted:
            select_gt_id = [i for i in self.video_2_lang['task_stepsC'] if self.video_2_lang['task_stepsC'][i] == cached_lang]
            if len(select_gt_id) > 0:
                self.video_2_lang['task_stepsC'][select_gt_id[0]] = None

            cached_lang = dialog.get_text()
            prim = dialog.get_prim()
            select_lang = dialog.get_select_lang()
            if select_lang != '空':
                self.video_2_lang['task_stepsC'][select_lang] = cached_lang
                
            self.lang_anno[anno_loc] = (cached_lang, prim, select_lang)
            self.clip_lang_input.setText(f"开始帧: {anno_loc[0]+1} | 结束帧: {anno_loc[1]+1}\n原子动作: {prim}\n动作描述: {cached_lang}")
        else:
            self.delete_keyframe()
            return
    
    def add_video_description(self):
        # Create a dialog to get the description from the user
        if (0, 0) in self.lang_anno:
            cached_lang = self.lang_anno[(0, 0)]
        else:
            cached_lang = ''
        
        dialog = TextInputDialog(cached_lang, self, True, self.video_2_lang)
        if dialog.exec_() == QDialog.Accepted:
            video_description = dialog.get_text()
            self.lang_anno[(0, 0)] = video_description
            self.video_lang_input.setText(f"视频描述: {video_description}")
        else:
            return

    def get_clip_description(self):
        # Get the description for the clip
        key_pairs = list(self.lang_anno.keys())
        frame_number = self.progress_slider.value()
        anno_loc = [(idx, i) for idx, i in enumerate(key_pairs) if i[0] <= frame_number <= i[1] and i[0] != i[1]]
        if len(anno_loc) > 0:
            anno_loc = anno_loc[0]
            return anno_loc, self.lang_anno[anno_loc[1]]
        return (None, None), (None, None, None)

    def get_clip_lang_anno(self):
        lang = self.video_2_lang['task_stepsC']
        out_text = ''
        for i in enumerate(lang):
            if i[1] not in BASE_CLIP_DES:
                out_text += f"{i[0]+1}: {i[1]}\n"
        return out_text.strip()


if __name__ == "__main__":
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass
    args = argparse.ArgumentParser()
    args.add_argument('--out_file', type=str, default='./annotation.pkl')
    args.add_argument('--sam_anno', type=str, default='./sam_anno.pkl')
    args.add_argument('--lang_anno', type=str, default='./lang_anno.pkl')
    args = args.parse_args()
    
    app = QApplication(sys.argv)
    player = VideoPlayer(args)
    player.resize(1100, 600)  # Adjusted size to accommodate the toolbar
    player.show()
    sys.exit(app.exec_())