from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
import csv


def eye_brow_distance(leye,reye):
    global points
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    return distq

def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ["angry" ,"disgust","scared", "sad", "surprised"]:
        stressed = 1
    else:
        stressed = 0
    return label,stressed
    
def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    stress_value = np.exp(-(normalized_value))
    return stress_value
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("best_fer_model.hdf5", compile=False)
cap = cv2.VideoCapture(0)
points = []

now = datetime.now()
current_time = 1
emotions_list = []
stress_list = []
recognized_time = []
img_name_array = []
numm = 0

while(True):

    
    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    
    
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

    #preprocessing the image
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    detections = detector(gray,0)
    for detection in detections:
        emotion,stressed = emotion_finder(detection,gray)
        cv2.putText(frame, emotion, (20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        shape = predictor(frame,detection)
        shape = face_utils.shape_to_np(shape)
           
        leyebrow = shape[lBegin:lEnd]
        reyebrow = shape[rBegin:rEnd]
            
        reyebrowhull = cv2.convexHull(reyebrow)
        leyebrowhull = cv2.convexHull(leyebrow)

        cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

        distq = eye_brow_distance(leyebrow[-1],reyebrow[0])
        stress_value = normalize_values(points,distq)
        stress_value = round(stress_value*100,2)
        if stressed == 1:
            cv2.putText(frame,"stress level: " + str(stress_value),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 5, 190), 2)
        else:
            stress_value = 0


    now = datetime.now()
    if current_time != int(now.strftime("%M")):
        try:
            numm += 1
            now = datetime.now()
            current_time = int(now.strftime("%M"))
            print("date and time =", current_time)
            emotions_list.append(emotion)
            stress_list.append(stress_value)
            now = datetime.now()
            recognized_time.append(str(now))
            img_name = "image_" + str(now) + ".png"
            img_name = img_name.replace(":","_")
            img_name_array.append(img_name)
            cv2.imwrite("data/" + img_name,frame)
        except:
            pass

        

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()


print(recognized_time)
print(emotions_list)
print(stress_list)

x = np.array(recognized_time)
y = np.array(stress_list)
y2 = np.array(emotions_list)


plt.title("Stress Level (%)")
plt.xlabel("Time")
plt.ylabel("Stress level")
plt.plot(x, y, color ="green")
plt.show()

plt.clf()


plt.title("Detected Emotion")
plt.xlabel("Time")
plt.ylabel("Emotion")
plt.plot(x, y2, color ="green")
plt.show()
plt.clf()


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

unique_emotions_list = unique(emotions_list)
print(unique_emotions_list)

unique_emotions_list1 = []
repeatation = []

for ue in unique_emotions_list:
    unique_emotions_list1.append(ue)
    repeatt = 0
    for ul in emotions_list:
        if ue == ul:
            repeatt += 1
    repeatation.append(repeatt)
    
yy = np.array(repeatation)


unique_emotions_labels = []

for i in range(len(yy)):
    unique_emotions_labels.append(unique_emotions_list1[i] + " : " + str(yy[i]))
    

plt.pie(yy, labels = unique_emotions_labels)
plt.legend(title = "Emotion repeatability")
plt.show() 
 

fieldnames = ["Time","Emotion","Stress","Image name"]


for i in range(len(recognized_time)):
    data2send = [recognized_time[i],stress_list[i],emotions_list[i],img_name_array[i]]
    data2write1=zip(fieldnames,data2send)
    data2write = dict(data2write1)
    with open('Stress_management.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(data2write)


