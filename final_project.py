# -*- coding: utf-8 -*-
"""
Created on Mon May 10 00:51:32 2021

@author: pankaj
"""

'''
Computer vision is an interdisciplinary scientific field that deals with how computers can gain
 high-level understanding from digital images or videos. From the perspective of engineering, it 
 seeks to understand and automate tasks that the human visual system can do.

'''

# importing the important libaries 
import cv2
import datetime
import imutils
import numpy as np   
from centroidtracker import CentroidTracker
from itertools import combinations
import math

# loading the mobilenetssd model from computer disk

protopath='C:/Users/pankaj/Downloads/inter/flask/MobileNetSSD_deploy.prototxt'
modelpath='C:/Users/pankaj/Downloads/inter/flask/MobileNetSSD_deploy.caffemodel'

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# tracker used for social distance and finding distance between two object
tracker = CentroidTracker(maxDisappeared=80, maxDistance=90)


def detail(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
            # detailing and dimension of rect formed
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))

def main():
    # video soucre
    cap = cv2.VideoCapture(r'C:\Users\pankaj\Downloads\041152577-paris-street-scene-bride-and-p_preview.mp4')
      # defining other thing for counting 
    fps_start_time = datetime.datetime.now()
    total_frames = 0
    lpc_count = 0
    opc_count = 0
    object_id_list = []
    # defining the dtail
    while True:
        ret, frame = cap.read()
        # seting up the size of frames
        frame = imutils.resize(frame, width=1000)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]
        #  seting up the shape of bob 
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 300)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = detail(boundingboxes, 0.3)
        
        #  setting of the cenroid and other dimension for detmining the centroid tracker
        centroid_dict = dict()
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            # determining the center part
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            
            # finding if object present in frames 
            if objectId not in object_id_list:
                object_id_list.append(objectId)
                # center distance
            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)
            # tracking the user id each person 
            text = "ID: {}".format(objectId)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
            #  detaining the red alert for people who break the social distance norms 
            
        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 75.0:
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)
                    
        # shape and size of rect form along people 
        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)  

        
        # count present on screen during the particular frames 
        lpc_count = len(objects)
        
        # total count the passed form that frames 
        opc_count = len(object_id_list)

        # writing the detail
        lpc_txt = "Person present: {}".format(lpc_count)
        opc_txt = "Total person cross: {}".format(opc_count)
        
        # puting the detail about style nd color of text
        cv2.putText(frame, lpc_txt, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
        cv2.putText(frame, opc_txt, (5, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0, 0), 1)

        # giving the name of frame 
        cv2.imshow("project", frame)
        key = cv2.waitKey(1)
        # assigning key for detroying the window
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()

