import cv2
import uuid
from ultralytics import YOLO

model = YOLO("C:/Users/dasab/OneDrive/Desktop/sourab_best.pt")
print(model.names)
# define a video capture object
vid = cv2.VideoCapture(0)#"http://192.168.0.157:4747/video")

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frame=cv2.resize(frame,(640,640))
    results = model.predict(frame, conf=0.4,imgsz=640,device=0)
    dec=results[0].boxes.cls.tolist()
    if len(dec)>0:
        cv2.imshow('Detected', results[0].plot())
    # Display the resulting frame
    cv2.imshow('Live', results[0].plot())


    # if cv2.waitKey(1) & 0xFF == ord('p'):
    #     print("a")
    #     cv2.imwrite(f"E:/DefectScanner/6_3_23/vegitableDatasetImg/{str(uuid.uuid4())}.png",cv2.resize(frame,(640,640)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print("q")
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

# import torch
# torch.zeros(1).cuda()
# print(torch.cuda.device_count())

