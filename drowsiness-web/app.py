import time

import psycopg2
from psycopg2 import sql
from flask import Flask, render_template, Response
import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
from threading import Thread, Event

app = Flask(__name__)


@app.route('/')
def index():
    create_table()
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(detect_drowsiness(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/history')
def history():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""SELECT * FROM alarm_data""")
        alarm_data = cursor.fetchall()

        return render_template("history.html", alarm_data=alarm_data)

    except BaseException as ex:
        print(ex)


def detect_drowsiness():
    # Yüz ve gözleri algılamak için Haar Cascade sınıflandırıcılarını yükler.
    face_cascade = cv2.CascadeClassifier("static/haarcascade_frontalface_default.xml")
    left_eye_cascade = cv2.CascadeClassifier("static/haarcascade_lefteye_2splits.xml")
    right_eye_cascade = cv2.CascadeClassifier("static/haarcascade_righteye_2splits.xml")

    # Gözlerin açık/kapalı olup olmadığını tahmin etmek için derin öğrenme modelini yükler.
    model = load_model("static/best-model4Cat.h5")

    count = 0  # Gözlerin kapalı kaldığı süreyi saymak için sayaç.
    alarm_on = False  # Alarmın çalıyor olup olmadığını takip etmek için.
    alarm_sound = "static/alarm.mp3"  # Alarm ses dosyasının yolu.
    stop_event = Event()  # Alarmı durdurmak için kullanılan olay.

    cap = cv2.VideoCapture(0)  # Varsayılan kamera cihazından video yakalamak için.

    while True:
        ret, frame = cap.read()  # Kameradan bir kare okur.
        if not ret:
            break  # Eğer kare okunamıyorsa döngüden çık.

        height = frame.shape[0]  # Çerçevenin yüksekliğini al.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Çerçeveyi gri tonlamaya çevir.
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Yüzleri algıla.

        status1 = status2 = 1  # Varsayılan olarak gözlerin açık olduğunu varsay.

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)  # Yüzü çerçevele.
            roi_gray = gray[y:y + h, x:x + w]  # Yüz bölgesini gri tonlamada al.
            roi_color = frame[y:y + h, x:x + w]  # Yüz bölgesini renkli al.
            left_eye = left_eye_cascade.detectMultiScale(roi_gray)  # Sol gözü algıla.
            right_eye = right_eye_cascade.detectMultiScale(roi_gray)  # Sağ gözü algıla.

            for (x1, y1, w1, h1) in left_eye:
                cv2.rectangle(roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 1)  # Sol gözü çerçevele.
                eye1 = roi_gray[y1:y1 + h1, x1:x1 + w1]  # Sol göz bölgesini gri tonlamada al.
                eye1 = cv2.resize(eye1, (145, 145))  # Sol göz bölgesini yeniden boyutlandır.
                eye1 = eye1.astype('float') / 255.0  # Piksel değerlerini normalize et.
                eye1 = np.expand_dims(eye1, axis=-1)  # Göz görüntüsüne kanal boyutu ekle.
                eye1 = np.expand_dims(eye1, axis=0)  # Göz görüntüsüne batch boyutu ekle.
                pred1 = model.predict(eye1)  # Gözün açık mı kapalı mı olduğunu tahmin et.
                status1 = np.argmax(pred1)  # Tahmini sınıfı al (0: Kapalı, 1: Açık).
                break  # İlk tespit edilen sol gözü kullan.

            for (x2, y2, w2, h2) in right_eye:
                cv2.rectangle(roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 1)  # Sağ gözü çerçevele.
                eye2 = roi_gray[y2:y2 + h2, x2:x2 + w2]  # Sağ göz bölgesini gri tonlamada al.
                eye2 = cv2.resize(eye2, (145, 145))  # Sağ göz bölgesini yeniden boyutlandır.
                eye2 = eye2.astype('float') / 255.0  # Piksel değerlerini normalize et.
                eye2 = np.expand_dims(eye2, axis=-1)  # Göz görüntüsüne kanal boyutu ekle.
                eye2 = np.expand_dims(eye2, axis=0)  # Göz görüntüsüne batch boyutu ekle.
                pred2 = model.predict(eye2)  # Gözün açık mı kapalı mı olduğunu tahmin et.
                status2 = np.argmax(pred2)  # Tahmini sınıfı al (0: Kapalı, 1: Açık).
                break  # İlk tespit edilen sağ gözü kullan.

            if status1 == 0 and status2 == 0:  # Eğer her iki göz de kapalı ise.
                count += 1  # Kapalı göz sayacını artır.
                cv2.putText(frame, "GOZ KAPALI, SAYI: " + str(count), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1)
                if count >= 20:  # kameranın saniyede 25 kare çektiği vvarsayılırsa 1 sn de ki kontrol olacaktır.
                    cv2.putText(frame, "Uyku Alarm!!!", (100, height - 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                    if not alarm_on:
                        alarm_on = True  # Alarmın açık olduğunu belirt.
                        stop_event.clear()  # Alarmı durdurma olayını temizle.
                        t = Thread(target=start_alarm,
                                   args=(alarm_sound, stop_event))  # Alarmı çalmaya başlatacak bir thread oluştur.
                        t.daemon = True
                        t.start()

                        # Alarm tetiklendiğinde veritabanına veri yaz
                        alarm_time = time.strftime("%Y-%m-%d %H:%M:%S")
                        additional_data = "Uyku Algılandı"
                        insert_alarm_data(alarm_time, additional_data)

            else:  # Eğer gözler açık ise.
                cv2.putText(frame, "GOZ ACIK", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
                count = 0  # Kapalı göz sayacını sıfırla.
                stop_event.set()  # Alarmı durdurma olayını tetikle.
                alarm_on = False  # Alarmın kapalı olduğunu belirt.

        _, jpeg = cv2.imencode('.jpg', frame)  # Çerçeveyi JPEG formatında sıkıştır.
        frame_bytes = jpeg.tobytes()  # JPEG verisini byte dizisi olarak al.
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')  # HTTP yanıtı olarak çerçeveyi döndür.

    cap.release()


def start_alarm(sound, stop_event):
    while not stop_event.is_set():
        playsound(sound)
        time.sleep(1)


def create_table():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS alarm_data (
            id SERIAL PRIMARY KEY,
            alarm_time TIMESTAMP NOT NULL,
            additional_data TEXT
        );
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating table: {e}")


def insert_alarm_data(alarm_time, additional_data):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        insert_query = sql.SQL("""
            INSERT INTO alarm_data (alarm_time, additional_data)
            VALUES (%s, %s)
        """)
        cursor.execute(insert_query, (alarm_time, additional_data))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error inserting data: {e}")


def get_db_connection():
    return psycopg2.connect(
        dbname="drowsiness_detection",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )


if __name__ == '__main__':
    app.run(debug=True)
