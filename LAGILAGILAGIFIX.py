import os
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from time import sleep
from ultralytics import YOLO


sender_email = 'aisafetyharita@gmail.com'
receiver_email = 'jhoenandasinulingga@gmail.com'
subject = 'Email Notifikasi'
server = smtplib.SMTP("smtp.gmail.com",587)
server.starttls()
server.login(sender_email, "jjopdhxzxogckrde")



cap = cv2.VideoCapture(0)
ret, frame = cap.read()

model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from RSTP")
        continue

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
            if int(class_id) == 1:
                # Prepare email content
                message = MIMEMultipart()
                message["From"] = sender_email
                message["To"] = receiver_email
                message["text"] = subject

                # Convert frame to JPEG image bytes
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()

                # Attach image as an application/octet-stream
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(img_bytes)
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', 'attachment', filename=f'captured_image_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
                message.attach(part)  
                server.sendmail(sender_email, receiver_email,message.as_string())
                sleep(60)

    cv2.imshow('YOLO', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
