from ultralytics import YOLO
import cv2 as cv
import numpy as np

model=YOLO('yolov8n.pt')
cap=cv.VideoCapture(0)

ret,frame=cap.read()

result=model(frame)

coco=result[0].names
color_box=(0,255,0)
color_label=(0,0,255)
while True:
    ret,frame=cap.read()
    if ret:
        result=model(frame)
        boxes=result[0].boxes
        boxes_data=boxes.data
        detections=boxes_data.numpy()
        for obj in detections:
                x1,y1,x2,y2=map(int,obj[:4])
                pt1=(x1,y1)
                pt2=(x2,y2)
                confidence=obj[4]
                class_id=obj[5]
                if confidence>0.5:
                    label=f"{coco[class_id]}:{confidence:.2f}"
                    cv.rectangle(frame,pt1,pt2,color_box,2)
                    cv.putText(frame,label,(x1,y1-10),cv.FONT_HERSHEY_SIMPLEX,0.5,color_label,2)
        
        cv.imshow('Object Detection',frame)

        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    else:
         print("Video is unable to load please try again")
cap.release()
cv.destroyAllWindows()