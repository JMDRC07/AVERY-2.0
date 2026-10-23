from ultralytics import YOLO #INSTALAR LIBRERIA---------- pip install ultralytics
import cv2                   #INSTALAR LIBRERIA --------- pip instal cv2 (no recuerdo bien... si ya la tienes q bueno)  
import threading
import pyttsx3               #INSTALAR LIBRERIA---------- pip install pyttsx3

cv2.startWindowThread()  

detener_camara = threading.Event()
camara_activa = threading.Event()

_hilo_cam  = None
_modelo = None

_tts_lock = threading.Lock()

def hablar(texto):
    with _tts_lock:
        engine = pyttsx3.init()
        try:
            engine.setProperty("rate", 170)
            engine.say(texto)
            engine.runAndWait()
        finally:
            try:
                engine.stop()
            except Exception:
                pass
            del engine


# NUEVO: aquí la cámara guarda lo que ve en cada frame
_ultima_deteccion = {"personas": 0, "confianzas": []}
_lock = threading.Lock()   # para evitar conflictos entre hilos

def cam_loop(modelo):
    cap = cv2.VideoCapture(0)
    while not detener_camara.is_set():
        #captura el frame que se visualiza en ese momento
        ret, frame = cap.read()
        if not ret:
            break

        #carga el frame y hace la inferencia con el modelo en cuestion
        results = modelo(frame, verbose=False)
        # results[0].boxes es un objeto con .xyxy, .cls, .conf... sustitucion a la antigua opcion de pandas... mas limpio
        boxes = results[0].boxes

        personas = 0
        confianzas = []

        for box in boxes:
            cls = int(box.cls[0])       # clase
            if cls != 0:                # 0 = persona en COCO
                continue
            # filtramos los resultados de la inferencia solamente a los resultados 
            # que tengan la clase 0... que en este caso es la clase que detecta "Humanos"
            
            #ITERAMOS LOS OBJETOS QUE DETECTA EN EL FRAME PARA HACER ESPECIFICACIONES
            conf = float(box.conf[0])   # confianza
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # rectángulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # texto con confianza
            cv2.putText(
                frame, f"Humano {conf:.2f}", (x1, y1 - 15),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 3
            )
            personas += 1
            confianzas.append(conf)

        with _lock:
            _ultima_deteccion["personas"]   = personas
            _ultima_deteccion["confianzas"] = confianzas
        
        cv2.imshow("Detector de Humanos", frame)
        if cv2.waitKey(5) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    camara_activa.clear()
    detener_camara.clear()   # resetear para la próxima vez


def detectar():
    """
    Abre la cámara en un hilo aparte y retorna inmediatamente.
    Así el hilo del micrófono sigue funcionando.
    """
    if camara_activa.is_set():
        print("La camara ya esta corriendo.")
        return

    camara_activa.set()
    detener_camara.clear()

    # cargamos una sola vez el modelo de forma global
    global _modelo, _hilo_cam
    if _modelo is None:
        _modelo = YOLO("yolov8n.pt")   # si no existe, se asigna

    _hilo_cam = threading.Thread(target=cam_loop, args=(_modelo,), daemon=True)
    _hilo_cam.start()
    print("Camara lanzada en segundo plano. Sigo escuchando...")

def cerrar_camara():
    global _hilo_cam
    if camara_activa.is_set():
        detener_camara.set()
        print("Cerrando camara...")
        if _hilo_cam is not None:
            _hilo_cam.join(timeout=2)
            if _hilo_cam.is_alive():
                print("Advertencia: el hilo de camara no termino a tiempo")
            _hilo_cam = None
    else:
        print("La camara no está activa.")

def describe():
    #STRING CON LO QUE LA CAMARA DETECTA AL MOMENTO
    if not camara_activa.is_set():
        return "La camara no está activa. Di 'avery' para comenzar."
    with _lock:
        personas   = _ultima_deteccion["personas"]
        confianzas = _ultima_deteccion["confianzas"]

    if personas == 0:
        return "No veo ningun humano en este momento."
    elif personas == 1:
        return f"Veo 1 humano en frente de ti."
    else:
        promedio = sum(confianzas) / len(confianzas)
        return f"Veo que estas rodeado de personas. Veo {personas} humanos en frente de ti (confianza promedio {promedio:.2f})."
    

        


