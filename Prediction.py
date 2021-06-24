# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 07:05:55 2021

@author: sunil
"""


import cv2
from tensorflow.keras.models import load_model
import operator
import sys, os
import numpy as np
from tensorflow.keras.preprocessing import image




model = load_model("aslr.h5")



cap = cv2.VideoCapture(0)

while True:
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = (frame.shape[1])-10
    y2 = int(0.5*frame.shape[1])
    
    cv2.rectangle(frame, (x1-1,y1-1), (x2+1,y2+1), (255,0,0),1)
    
    
    roi = frame[y1:y2, x1:x2]
    
    
    roi = cv2.resize(roi, (64,64))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,test_image = cv2.threshold(roi, 155, 255, cv2.THRESH_BINARY)

    
    result = model.predict(test_image.reshape(1,64,64,1))   #(1,64,64,3)
    
    prediction = {
        "one" : result[0][0],
        "nothing" : result[0][1],
        "two" : result[0][2],
        "three" : result[0][3],
        "four" : result[0][4],
        "five": result[0][5],
        "six" : result[0][6],
        "seven" : result[0][7],
        "eight" : result[0][8],
        "nine" : result[0][9],
        "A" : result[0][10],
        "B" : result[0][11],
        "C" : result[0][12],
        "D" : result[0][13],
        "E" : result[0][14],
        "F" : result[0][15],
        "G" : result[0][16],
        "H" : result[0][17],
        "I" : result[0][18],
        "J" : result[0][19],
        "K" : result[0][20],
        "L" : result[0][21],
        "M" : result[0][22],
        "N" : result[0][23],
        "O" : result[0][24],
        "P" : result[0][25],
        "Q" : result[0][26],
        "R" : result[0][27],
        "S" : result[0][28],
        "T" : result[0][29],
        "U" : result[0][30],
        "V" : result[0][31],
        "W" : result[0][32],
        "X" : result[0][33],
        "Y" : result[0][34],
        "Z" : result[0][35],
        "best of luck" : result[0][36],
        "fuck you" : result[0][37],
        "i love you" : result[0][38],
        "space" : result[0][39]
    }
    
    predictions = sorted(prediction.items(), key = operator.itemgetter(1), reverse = True)
    
    
    cv2.putText(frame, predictions[0][0], (x1+100,y2+30), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    
    cv2.imshow("test_image", test_image)
    cv2.imshow("frame", frame)
    
    
    if cv2.waitKey(10) & 0xFF == 27:
        break
       
cap.release()
cv2.destroyAllWindows()