import cv2 as cv

# Importing model

net = cv.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)

# accessing first webcam
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,720)


# creating a new window

cv.namedWindow('Frame')
# cv.setMouseCallback('Frame',click_button)

# Load classes
classes = []
with open("dnn_model/classes.txt","r") as fileObject:
    for className in fileObject.readlines():
        className = className.strip()
        classes.append(className)
        print(className)

while True:
    # creating frame
    ret, frame = cap.read()

    # detection start here
    (class_ids,scores,bboxes) = model.detect(frame)

    for class_id,score,bboxe in zip(class_ids,scores,bboxes):
        (x,y,w,h) = bboxe
        print(x,y,w,h)
        cv.putText(frame,str(classes[class_id]),(x,y-10),cv.FONT_HERSHEY_PLAIN,2,(200,0,50),2)
        cv.rectangle(frame,(x,y),(x+w,x+h),(200,0,50),)

    # printing class_ids,scores and bounded boxes
    # print("Class Ids",class_ids)
    # print("Scores", scores)
    # print("Bboxes", bboxes)
    print(classes)

    cv.imshow("Frame", frame)

    # to freze the screen but output will be a image
    # cv.waitKey(0)
#     To make dynamic or video
    cv.waitKey(1)
