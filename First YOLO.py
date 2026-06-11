#open model

from ultralytics import YOLO
import cv2

#open camera

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

#initial setup variables

person_positions = {}
in_count = 0
out_count = 0
line_y = 300

#reed frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #edit frame

    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)
    frame = cv2.flip(frame, 1)

    #send frame to YOLO
    
    results = model.track(
        frame,
        persist=True,
        classes=[0], # person only
        conf=0.5,
        tracker="bytetrack.yaml"
        )
    
    annotated_frame = results[0].plot()

    # box size

    for box in results[0].boxes:
        track_id = int(box.id[0]) if box.id is not None else None
        
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        #track the person_position

        if track_id is not None:
            if track_id in person_positions:
                previous_y = person_positions[track_id]
                if previous_y < line_y and center_y >= line_y:
                    in_count += 1
                    print(f"count = {in_count}")
                elif previous_y > line_y and center_y <= line_y:
                    out_count += 1
                    print(f"count = {out_count}")

            person_positions[track_id] = center_y

        cv2.circle(
            annotated_frame,
            (center_x, center_y),
            5,
            (0, 0, 255),
            -1
        )
    
    #line_y = 300  remove it later

    cv2.line(
        annotated_frame,
        (0, line_y),
        (annotated_frame.shape[1], line_y),
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated_frame,
        f"IN : {in_count}",
        (20, 50),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2
    )
    cv2.putText(
        annotated_frame,
        f"OUT : {out_count}",
        (20, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("YOLOv8 Webcam", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()