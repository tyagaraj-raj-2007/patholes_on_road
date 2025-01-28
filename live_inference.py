import cv2 as cv
import time
import geocoder
import os

# Reading label names from obj.names file (assuming it contains 'pothole')
class_name = []
with open(os.path.join("project_files", 'obj.names'), 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]

# Importing model weights and config file and defining model parameters
net1 = cv.dnn.readNet('project_files/yolov4_tiny.weights', 'project_files/yolov4_tiny.cfg')

# Disable CUDA and use CPU for processing (fix for error)
net1.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV's CPU backend
net1.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)       # Use CPU target

model1 = cv.dnn_DetectionModel(net1)
model1.setInputParams(size=(640, 480), scale=1/255, swapRB=True)

# Defining the video source (use '0' for the default camera, or change if using an external one)
video_source = 0  # Change this to a different number if you have multiple cameras
cap = cv.VideoCapture(video_source)  # Open the camera stream

if not cap.isOpened():
    print("Error: Unable to open camera")
    exit()

width = cap.get(3)
height = cap.get(4)

# Create a VideoWriter to save the output (optional)
result = cv.VideoWriter('result.avi', 
                         cv.VideoWriter_fourcc(*'MJPG'),
                         10, (int(width), int(height)))

# Defining initial values for parameters
g = geocoder.ip('me')  # To get geolocation info (ensure internet access)
result_path = "pothole_coordinates"
starting_time = time.time()
Conf_threshold = 0.5
NMS_threshold = 0.4
frame_counter = 0
i = 0
b = 0

# Create the result directory if it doesn't exist
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Detection loop
while True:
    ret, frame = cap.read()
    frame_counter += 1
    if not ret:
        print("Failed to read frame from the camera.")
        break

    # Perform object detection with the model
    classes, scores, boxes = model1.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        label = "pothole"
        x, y, w, h = box
        recarea = w * h
        area = width * height

        # Drawing detection boxes on frame for detected potholes
        if len(scores) != 0 and scores[0] >= 0.7:
            if (recarea / area) <= 0.1 and box[1] < 600:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(frame, f"%{round(scores[0]*100, 2)} {label}", 
                           (box[0], box[1]-10), cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
                
                # Save image and GPS coordinates every time a pothole is detected
                if i == 0:
                    cv.imwrite(os.path.join(result_path, f'pothole{i}.jpg'), frame)
                    with open(os.path.join(result_path, f'pothole{i}.txt'), 'w') as f:
                        f.write(str(g.latlng))  # Write GPS coordinates to file
                    i += 1
                else:
                    if (time.time() - b) >= 2:  # Save every 2 seconds
                        cv.imwrite(os.path.join(result_path, f'pothole{i}.jpg'), frame)
                        with open(os.path.join(result_path, f'pothole{i}.txt'), 'w') as f:
                            f.write(str(g.latlng))  # Write GPS coordinates to file
                        b = time.time()
                        i += 1

    # Writing FPS on frame
    endingTime = time.time() - starting_time
    fps = frame_counter / endingTime
    cv.putText(frame, f'FPS: {fps:.2f}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)

    # Show the frame with detections
    cv.imshow('frame', frame)
    result.write(frame)

    # Break loop on 'q' key press
    key = cv.waitKey(1)
    if key == ord('q'):
        break

# Release video capture and writer, close windows
cap.release()
result.release()
cv.destroyAllWindows()
