from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
img = cv2.imread("D:/AI Project/YOLO/dog.jpg")

if img is None :
    print("image is not found !")
    exit()

results = model(img)
annotated_img = results[0].plot() 

cv2.imshow("Result" , annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()  

cv2.imwrite("output.jpg", annotated_img)