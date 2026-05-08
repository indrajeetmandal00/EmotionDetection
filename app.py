from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import datetime
import random
import math
from collections import Counter
import os

app = Flask(__name__)

os.makedirs('captures', exist_ok=True)

# Load Models
model = load_model('model_file_30epochs.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

color_dict={
    0: (0, 0, 255),    # Red - Angry
    1: (0, 255, 100),  # Greenish - Disgust
    2: (255, 0, 255),  # Purple - Fear
    3: (0, 255, 0),    # Green - Happy
    4: (255, 255, 255),# White - Neutral
    5: (255, 0, 0),    # Blue - Sad
    6: (0, 255, 255)   # Yellow - Surprise
}

def generate_frames():
    video = cv2.VideoCapture(0)
    
    prev_time = 0
    frame_count = 0
    history = []
    max_history = 50
    data_stream = []
    smooth_box = None
    emotion_history = []
    conf_history = []
    heart_rate = 75
    bpm_history = []
    max_bpm_history = 50

    while True:
        success, frame = video.read()
        if not success:
            break

        frame_count += 1
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width, _ = frame.shape
        
        # Base Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 10), (320, 150), (0, 0, 0), -1)
        cv2.rectangle(overlay, (5, height - 250), (150, height - 10), (0, 0, 0), -1)
        
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Info Box
        cv2.rectangle(frame, (5, 10), (320, 150), (255, 255, 255), 1)
        cv2.putText(frame, "EMOTION ANALYSIS STUDIO", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"STATUS: ONLINE", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.2f}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"RES: {width}x{height}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        current_datetime = datetime.datetime.now().strftime("%Y.%m.%d / %H:%M:%S")
        cv2.putText(frame, f"TIME: {current_datetime}", (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        cv2.putText(frame, f"FACES DETECTED: {len(faces)}", (15, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255) if len(faces)>0 else (150, 150, 150), 1)

        # Event Log
        if frame_count % 30 == 0:
            if len(faces) > 0:
                data_stream.insert(0, f"Analysis: {len(faces)} Face(s)")
            else:
                data_stream.insert(0, "Scanning...")
            if len(data_stream) > 10:
                data_stream.pop()
        cv2.rectangle(frame, (5, height - 250), (150, height - 10), (255, 255, 255), 1)
        for i, log_str in enumerate(data_stream):
            y_pos = height - 20 - (i * 20)
            cv2.putText(frame, log_str, (15, y_pos), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 255), 1)

        # Insets Layout
        inset_w, inset_h = 160, 120
        pip_margin = 10
        
        # 1. Canny Edge PIP
        edges = cv2.Canny(gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_x = width - inset_w - pip_margin
        edge_y = height - inset_h - pip_margin
        
        cv2.rectangle(frame, (edge_x, edge_y), (edge_x + inset_w, edge_y + inset_h), (255, 255, 255), 1)
        frame[edge_y:edge_y+inset_h, edge_x:edge_x+inset_w] = cv2.resize(edges_colored, (inset_w, inset_h))
        cv2.rectangle(frame, (edge_x, edge_y - 15), (edge_x + inset_w, edge_y), (255, 255, 255), -1)
        cv2.putText(frame, "EDGE DETECT", (edge_x + 5, edge_y - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)

        # 2. CNN Model Input PIP
        input_x = edge_x - inset_w - pip_margin
        input_y = edge_y
        
        cv2.rectangle(frame, (input_x, input_y), (input_x + inset_w, input_y + inset_h), (255, 255, 255), 1)
        cv2.rectangle(frame, (input_x, input_y - 15), (input_x + inset_w, input_y), (255, 255, 255), -1)
        cv2.putText(frame, "CNN INPUT VIEW", (input_x + 5, input_y - 4), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 0), 1)
        cv2.putText(frame, "NO FACE", (input_x + 35, input_y + 65), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

        target_bpm = 75
        target_color = (255, 255, 255)
        
        if len(faces) > 0:
            faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
            x, y, w, h = faces[0]

            if smooth_box is None:
                smooth_box = [x, y, w, h]
            else:
                dist = math.hypot(smooth_box[0] - x, smooth_box[1] - y)
                if dist > 100:
                    smooth_box = [x, y, w, h]
                else:
                    alpha_box = 0.2
                    smooth_box[0] = smooth_box[0] * (1 - alpha_box) + x * alpha_box
                    smooth_box[1] = smooth_box[1] * (1 - alpha_box) + y * alpha_box
                    smooth_box[2] = smooth_box[2] * (1 - alpha_box) + w * alpha_box
                    smooth_box[3] = smooth_box[3] * (1 - alpha_box) + h * alpha_box
            
            sx, sy, sw, sh = [int(v) for v in smooth_box]

            sub_face_img=gray[y:y+h, x:x+w]
            resized=cv2.resize(sub_face_img,(48,48))
            
            input_view = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            input_view_resized = cv2.resize(input_view, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
            frame[input_y:input_y+inset_h, input_x:input_x+inset_w] = input_view_resized
            
            normalize=resized/255.0
            reshaped=np.reshape(normalize, (1, 48, 48, 1))
            
            result=model.predict(reshaped, verbose=0)
            label=np.argmax(result, axis=1)[0]
            confidence = np.max(result) * 100

            emotion_history.append(label)
            if len(emotion_history) > 10:
                emotion_history.pop(0)
            
            counts = Counter(emotion_history)
            smooth_label = counts.most_common(1)[0][0]
            
            conf_history.append(confidence)
            if len(conf_history) > 5:
                conf_history.pop(0)
            smooth_confidence = sum(conf_history) / len(conf_history)
            
            emotion = labels_dict[smooth_label]
            target_color = color_dict.get(smooth_label, (255, 255, 255))

            history.append(smooth_confidence)
            if len(history) > max_history:
                history.pop(0)

            if smooth_label in [0, 2]:
                target_bpm = 120 + random.randint(-5, 10)
            elif smooth_label == 6:
                target_bpm = 100 + random.randint(-5, 5)
            elif smooth_label == 3:
                target_bpm = 85 + random.randint(-2, 5)
            elif smooth_label == 5:
                target_bpm = 65 + random.randint(-2, 2)
            else:
                target_bpm = 75 + random.randint(-2, 2)

            t_thick = 2
            length = int(sw * 0.2) 
            
            cv2.line(frame, (sx, sy), (sx + length, sy), target_color, t_thick)
            cv2.line(frame, (sx, sy), (sx, sy + length), target_color, t_thick)
            cv2.line(frame, (sx + sw, sy), (sx + sw - length, sy), target_color, t_thick)
            cv2.line(frame, (sx + sw, sy), (sx + sw, sy + length), target_color, t_thick)
            cv2.line(frame, (sx, sy + sh), (sx + length, sy + sh), target_color, t_thick)
            cv2.line(frame, (sx, sy + sh), (sx, sy + sh - length), target_color, t_thick)
            cv2.line(frame, (sx + sw, sy + sh), (sx + sw - length, sy + sh), target_color, t_thick)
            cv2.line(frame, (sx + sw, sy + sh), (sx + sw, sy + sh - length), target_color, t_thick)
            cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), target_color, 1)

            cx, cy = sx + sw // 2, sy + sh // 2
            radius = int(max(sw, sh) // 1.5)
            start_angle = frame_count * 3 % 360
            end_angle = start_angle + 45
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle, end_angle, target_color, 2)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_angle + 180, end_angle + 180, target_color, 2)
            cv2.circle(frame, (cx, cy), 2, target_color, -1)

            panel_w, panel_h = 160, 95
            panel_x = sx + sw + 15
            panel_y = sy
            
            if panel_x + panel_w > width:
                panel_x = sx - panel_w - 15
                
            panel_overlay = frame.copy()
            cv2.rectangle(panel_overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
            frame = cv2.addWeighted(panel_overlay, 0.7, frame, 0.3, 0)
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), target_color, 1) 
            cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 20), target_color, -1)
            cv2.putText(frame, f"SUBJECT-01", (panel_x + 5, panel_y + 14), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)

            cv2.putText(frame, f"EMOTION:", (panel_x + 5, panel_y + 35), cv2.FONT_HERSHEY_PLAIN, 0.8, target_color, 1)
            cv2.putText(frame, f"{emotion.upper()}", (panel_x + 75, panel_y + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, target_color, 2)
            cv2.putText(frame, f"CONF:", (panel_x + 5, panel_y + 55), cv2.FONT_HERSHEY_PLAIN, 0.8, target_color, 1)
            cv2.putText(frame, f"{smooth_confidence:.1f}%", (panel_x + 55, panel_y + 55), cv2.FONT_HERSHEY_PLAIN, 1, target_color, 1)
            cv2.putText(frame, f"POS: {sx},{sy}", (panel_x + 5, panel_y + 70), cv2.FONT_HERSHEY_PLAIN, 0.8, target_color, 1)
            cv2.putText(frame, f"DIM: {sw}x{sh}", (panel_x + 5, panel_y + 85), cv2.FONT_HERSHEY_PLAIN, 0.8, target_color, 1)
            
            if panel_x > sx:
                cv2.line(frame, (sx+sw, sy+10), (panel_x, panel_y+10), target_color, 1)
            else:
                cv2.line(frame, (sx, sy+10), (panel_x + panel_w, panel_y+10), target_color, 1)
        else:
            smooth_box = None
            emotion_history.clear()
            conf_history.clear()

        heart_rate = heart_rate * 0.9 + target_bpm * 0.1
        bpm_history.append(heart_rate)
        if len(bpm_history) > max_bpm_history:
            bpm_history.pop(0)

        graph_w = 160
        graph_h = 60
        graph_x = width - graph_w - 10
        graph_y = 10
        
        graph_overlay = frame.copy()
        cv2.rectangle(graph_overlay, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(graph_overlay, 0.6, frame, 0.4, 0)
        g_color = target_color if len(faces) > 0 else (255, 255, 255)
        
        cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), g_color, 1)
        cv2.rectangle(frame, (graph_x, graph_y - 20), (graph_x + graph_w, graph_y), g_color, -1)
        cv2.putText(frame, "CONFIDENCE", (graph_x + 5, graph_y - 6), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        
        if len(history) > 1:
            for i in range(1, len(history)):
                p1_x = int(graph_x + (i - 1) * (graph_w / max_history))
                p1_y = int(graph_y + graph_h - (history[i - 1] / 100.0 * graph_h))
                p2_x = int(graph_x + i * (graph_w / max_history))
                p2_y = int(graph_y + graph_h - (history[i] / 100.0 * graph_h))
                cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), g_color, 2)

        bio_y = graph_y + graph_h + 30
        cv2.rectangle(graph_overlay, (graph_x, bio_y), (graph_x + graph_w, bio_y + graph_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(graph_overlay, 0.6, frame, 0.4, 0)
        bio_color = (255, 255, 255)
        
        cv2.rectangle(frame, (graph_x, bio_y), (graph_x + graph_w, bio_y + graph_h), bio_color, 1)
        cv2.rectangle(frame, (graph_x, bio_y - 20), (graph_x + graph_w, bio_y), bio_color, -1)
        cv2.putText(frame, "BIOMETRICS", (graph_x + 5, bio_y - 6), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 1)
        
        cv2.putText(frame, f"HR: {int(heart_rate)} BPM", (graph_x + 5, bio_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bio_color, 1)
        
        if len(bpm_history) > 1:
            for i in range(1, len(bpm_history)):
                val1 = max(0, min(1, (bpm_history[i-1] - 50) / 100.0))
                val2 = max(0, min(1, (bpm_history[i] - 50) / 100.0))
                
                p1_x = int(graph_x + (i - 1) * (graph_w / max_bpm_history))
                p1_y = int(bio_y + graph_h - (val1 * graph_h))
                p2_x = int(graph_x + i * (graph_w / max_bpm_history))
                p2_y = int(bio_y + graph_h - (val2 * graph_h))
                cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), bio_color, 1)

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the output frame in the byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)