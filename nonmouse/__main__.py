#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# NonMouse
# Author: Yuki Takeyama
# Date: 2023/04/09

import cv2
import time
import keyboard
import platform
import numpy as np
import mediapipe as mp
from pynput.mouse import Button, Controller

from nonmouse.args import *
from nonmouse.utils import *

mouse = Controller()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

pf = platform.system()
if pf == 'Windows':
    hotkey = 'Alt'
elif pf == 'Darwin':
    hotkey = 'Command'
elif pf == 'Linux':
    hotkey = 'XXX'              # hotkeyはLinuxでは無効


def main():
    cap_device, mode, kando, screenRes = tk_arg()
    dis = 0.7                           # くっつける距離の定義
    preX_4, preY_4 = 0, 0
    preX_8, preY_8 = 0, 0
    nowCli, preCli = 0, 0               # 現在、前回の左クリック状態
    norCli, prrCli = 0, 0               # 現在、前回の右クリック状態
    douCli = 0                          # ダブルクリック状態
    i, k, h = 0, 0, 0
    LiTx_4, LiTy_4 = [], []
    LiTx_8, LiTy_8, list0x, list0y, list1x, list1y, list4x, list4y, list6x, list6y, list8x, list8y, list12x, list12y = [
    ], [], [], [], [], [], [], [], [], [], [], [], [], []   # 移動平均用リスト
    moving_average = [[0] * 3 for _ in range(3)]
    cap_width = 1280
    cap_height = 720
    start, c_start = float('inf'), float('inf')
    c_text = 0
    # Webカメラ入力, 設定
    window_name = 'NonMouse'
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cfps = int(cap.get(cv2.CAP_PROP_FPS))
    if cfps < 30:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
        cfps = int(cap.get(cv2.CAP_PROP_FPS))
    # スムージング量（小さい:カーソルが小刻みに動く 大きい:遅延が大）
    ran = max(int(cfps/10), 1)
    hands = mp_hands.Hands(
        min_detection_confidence=0.8,   # 検出信頼度
        min_tracking_confidence=0.8,    # 追跡信頼度
        max_num_hands=1                 # 最大検出数
    )
    # メインループ ###############################################################################
    while cap.isOpened():
        p_s = time.perf_counter()
        success, image = cap.read()
        if not success:
            continue
        if mode == 1:                   # Mouse
            image = cv2.flip(image, 0)  # 上下反転
        elif mode == 2:                 # Touch
            image = cv2.flip(image, 1)  # 左右反転

        # 画像を水平方向に反転し、BGR画像をRGBに変換
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False   # 参照渡しのためにイメージを書き込み不可としてマーク
        results = hands.process(image)  # mediapipeの処理
        image.flags.writeable = True    # 画像に手のアノテーションを描画
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_height, image_width, _ = image.shape

        if results.multi_hand_landmarks:
            # 手の骨格描画
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if pf == 'Linux':           # Linuxだったら、常に動かす
                can = 1
                c_text = 0
            else:                       # Linuxじゃなかったら、keyboardからの入力を受け付ける
                if keyboard.is_pressed(hotkey):  # linuxではこの条件文に触れないように
                    can = 1
                    c_text = 0          # push hotkeyなし
                else:                   # 入力がなかったら、動かさない
                    can = 0
                    c_text = 1          # push hotkeyあり
                    # i = 0
            # グローバルホットキーが押されているとき ##################################################
            if can == 1:
                # print(hand_landmarks.landmark[0])
                # preX_8, preY_8に現在のマウス位置を代入 1回だけ実行
                if i == 0:
                    preX_4 = hand_landmarks.landmark[4].x
                    preY_4 = hand_landmarks.landmark[4].y
                    preX_8 = hand_landmarks.landmark[8].x
                    preY_8 = hand_landmarks.landmark[8].y
                    i += 1

                # 以下で使うランドマーク座標の移動平均計算
                landmark0 = [calculate_moving_average(hand_landmarks.landmark[0].x, ran, list0x), calculate_moving_average(
                    hand_landmarks.landmark[0].y, ran, list0y)]
                landmark1 = [calculate_moving_average(hand_landmarks.landmark[1].x, ran, list1x), calculate_moving_average(
                    hand_landmarks.landmark[1].y, ran, list1y)]
                landmark4 = [calculate_moving_average(hand_landmarks.landmark[4].x, ran, list4x), calculate_moving_average(
                    hand_landmarks.landmark[4].y, ran, list4y)]
                landmark6 = [calculate_moving_average(hand_landmarks.landmark[6].x, ran, list6x), calculate_moving_average(
                    hand_landmarks.landmark[6].y, ran, list6y)]
                landmark8 = [calculate_moving_average(hand_landmarks.landmark[8].x, ran, list8x), calculate_moving_average(
                    hand_landmarks.landmark[8].y, ran, list8y)]
                landmark12 = [calculate_moving_average(hand_landmarks.landmark[12].x, ran, list12x), calculate_moving_average(
                    hand_landmarks.landmark[12].y, ran, list12y)]

                posx, posy = mouse.position

                # 人差し指の先端をカーソルに対応
                # カメラ座標をマウス移動量に変換
                nowX_8 = calculate_moving_average(
                    hand_landmarks.landmark[8].x, ran, LiTx_8)
                nowY_8 = calculate_moving_average(
                    hand_landmarks.landmark[8].y, ran, LiTy_8)
                
                nowX_4 = calculate_moving_average(
                    hand_landmarks.landmark[4].x, ran, LiTx_4)
                nowY_4 = calculate_moving_average(
                    hand_landmarks.landmark[4].y, ran, LiTy_4)

                dx_4 = kando * (nowX_4 - preX_4) * image_width
                dy_4 = kando * (nowY_4 - preY_4) * image_height
                dx_8 = kando * (nowX_8 - preX_8) * image_width
                dy_8 = kando * (nowY_8 - preY_8) * image_height

                if pf == 'Windows' or pf == 'Linux':     # Windows,linuxの場合、マウス移動量に0.5を足して補正
                    dx_8 = dx_8+0.5
                    dy_8 = dy_8+0.5

                preX_4 = nowX_4
                preY_4 = nowY_4
                preX_8 = nowX_8
                preY_8 = nowY_8
                if posx+dx_8 < 0:  # カーソルがディスプレイから出て戻ってこなくなる問題の防止
                    dx_8 = -posx
                elif posx+dx_8 > screenRes[0]:
                    dx_8 = screenRes[0]-posx
                if posy+dy_8 < 0:
                    dy_8 = -posy
                elif posy+dy_8 > screenRes[1]:
                    dy_8 = screenRes[1]-posy

                mouse.move(dx_8, dy_8)
                draw_circle(image, hand_landmarks.landmark[8].x * image_width,
                            hand_landmarks.landmark[8].y * image_height, 8, (250, 0, 0))

                if abs(dx_4) > 10:
                    mouse.scroll(0, dx_4 / 8)
                    draw_circle(image, hand_landmarks.landmark[4].x * image_width,
                                hand_landmarks.landmark[4].y * image_height, 8, (0, 255, 0))

                preCli = nowCli
                prrCli = norCli

        # 表示 #################################################################################
        if c_text == 1:
            cv2.putText(image, f"Push {hotkey}", (20, 450),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(image, "cameraFPS:"+str(cfps), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        p_e = time.perf_counter()
        fps = str(int(1/(float(p_e)-float(p_s))))
        cv2.putText(image, "FPS:"+fps, (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        dst = cv2.resize(image, dsize=None, fx=0.4,
                         fy=0.4)         # HDの0.4倍で表示
        cv2.imshow(window_name, dst)
        if (cv2.waitKey(1) & 0xFF == 27) or (cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) == 0):
            break
    cap.release()


if __name__ == "__main__":
    main()
