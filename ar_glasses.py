import cv2
from mtcnn import MTCNN


cap = cv2.VideoCapture("1.mp4") # Path to video file
detector = MTCNN() # Intializing face detection 

IMG_NAME = "glass3.png" # path to glass image

# def blur_face(face_crop):
#     blur_face = cv2.GaussianBlur(face_crop,(45,45),30)


def resize_image(image, width = None, height=None,inter = cv2.INTER_AREA):
    '''
        This function resize dyanamically accordig to eyes cordinate
        param:
            image: glass image
            width: image width respective to eye corindate
            height: image height respective to eye corindate
            inter: cv2 resize method
        return: Dynamically resized glass image
    '''
    dim =None
    h,w,_ = image.shape

    if width is None and height is None:
        return image
    elif width is None:
        r = height / float(h)
        dim = (int(w*r),height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
   
    return resized

while True:
    rect,frame = cap.read()
    if rect:
        glasses = cv2.imread(IMG_NAME,-1)
        frame = cv2.resize(frame,(600,350))
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        bounding_box = detector.detect_faces(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
        try:
            for box in bounding_box:
                boxes = box.get("box")
                key_points = box.get('keypoints')

                left_eyes = key_points.get('left_eye')
                right_eyes = key_points.get('right_eye')
                ex = left_eyes[0] -50
                ey = left_eyes[1] - 55
                ew = right_eyes[0] - 35
                eh = right_eyes[1]


                glasses = resize_image(glasses,width=int((ew-ex)*1.9))
                gh, gw, _ = glasses.shape
                alpha_glass = glasses[:,:,3]/255.0
                alpha_1 = 1.0 - alpha_glass
                for c in range(0,3):
                    frame[ey:ey+gh,ex:ex+gw,c] = (alpha_glass*glasses[:,:,c]+ alpha_1* frame[ey:ey+gh,ex:ex+gw,c])

                # cv2.imshow("Crop ", crop_frame) 
                # for i in range(0, gw):
                #     for j in range(0, gh):
                #         #print(glasses[i, j]) #RGBA
                #         if glasses[i, j][3] != 0: # alpha 0
                #             frame[ey-45+i , ex-40 + j] = glasses[i, j]

            cv2.imwrite("Sample.png",frame)
            cv2.imshow("Frame",frame)
        except:
            pass
            # for key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# '''
cv2.destroyAllWindows()