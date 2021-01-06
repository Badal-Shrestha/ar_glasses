import cv2
from mtcnn import MTCNN

video_path = 0

cap = cv2.VideoCapture(video_path)

detector = MTCNN()

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(600,550))
    face_Data = detector.detect_faces(frame)

    for bounding_box in  face_Data:
        boxes = bounding_box.get("box")
        print(boxes)
        x = boxes[0] -20
        y = boxes[1] -20 
        w = boxes[0]+boxes[2] +20
        h = boxes[1]+boxes[3]+20
        cv2.rectangle(frame,(x,y),(w,h),(255,0,0),2)

    cv2.imshow("frame",frame)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()