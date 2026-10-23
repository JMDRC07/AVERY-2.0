#import speech_recognition as SR    #importar libreria
from rapidfuzz import fuzz         #importar libreria
import re, unicodedata, json
import os
os.environ["VOSK_LOG_LEVEL"] = "-1"   # -1 = silencio total
import pyaudio
import vosk
from ia_detect import  *


MODELO_PATH  = "vosk-model-small-es-0.42"
SAMPLE_RATE  = 16000
BLOQUE       = 4000 

if not os.path.isdir(MODELO_PATH):
    raise FileNotFoundError(f"Modelo no encontrado: {MODELO_PATH}")

# Solo estas frases reconocerá Vosk (más [unk] para lo demás)
GRAMMAR = [
    "avery",  "abrir", "hola",
    "que ves", "que vez", "que observas", "observas", "mirar", "miras",
    "cerrar", "cierra", "cierre", "cerra",
    "adios", "nos vemos", "chao", "apagar", "bye",
    "[unk]"
]




#usa variaciones por si no detecta exactamente lo mismo
VARIANTES = ["avery", "averi", "aberi", "haber i", "a veri", "abril"]

#VARIACIONES ORDEN PARA DETECTAR
VARIANTES2 = ["que vez", "que ves", "ves", "bes", "bez", "queves","que observas", "observas", "wachas", "mirar", "miras", "percibes"]

#VARIACIONES PARA CERRAR LA CAMARA
VARIANTES3 = ["cerrar", "cierra", "cierre", "cerra", "cerrado"]

#APAGAR EN SAFE
VARIANTES4 = ["adios", "nos vemos", "chao", "apagar", "bye"]

#NORMALIZA EL TEXTO QUITANDO CARACTERES RAROS Y CONVIRITIENDO A MINUSCULAS
def normalizar(t):
    t = t.lower()
    t = ''.join(c for c in unicodedata.normalize('NFD', t)
                if unicodedata.category(c) != 'Mn')
    t = re.sub(r'[^a-z0-9 ]', '', t)
    return t

#HACE LA COINCIDENCIA ENTRE EL TEXTO NORMALIZADO Y ALGUNA DE LAS DE VARIABLES PARA ALGUNA ORDEN ESPECIFICA
def coincide(texto, variantes, umbral=90):
    texto = normalizar(texto)
    for palabra in texto.split():
        for v in variantes:
            if fuzz.ratio(palabra, normalizar(v)) >= umbral:
                return True
    return False

model = vosk.Model(MODELO_PATH)
rec = vosk.KaldiRecognizer(model, SAMPLE_RATE, json.dumps(GRAMMAR, ensure_ascii=False))


audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=BLOQUE
)
stream.start_stream()

biem = "Despertando... di 'AVERI' para comenzar. Para terminar el programa di 'cerrar'."

print(biem)
hablar(biem)

#CODIGO DE PRUEBA #MICROFONO
try: 
    while True:
        try:

            # Si el bot está hablando, no escuches (evita eco)
            data = stream.read(BLOQUE, exception_on_overflow=False)
        except OSError as e:
            print("Error de audio:", e)
            continue

        if rec.AcceptWaveform(data):
            resultado = json.loads(rec.Result())
            texto = resultado.get("text", "").strip()

            if not texto or texto == "[unk]":
                continue

            # 2. Imprimir
            print("Dijiste:", texto)
            if coincide(texto, VARIANTES):
                txt = "Hola!, En que puedo ayudarte?"
                print(">>> BIENVENIDO <<<")
                detectar()
                hablar(txt)
            elif coincide(texto, VARIANTES2):
                print(">>> REPORTE DE LA CAMARA <<<")
                reporte = describe()
                print(reporte)
                hablar(reporte)
            elif coincide(texto, VARIANTES3):
                    txt = "Camara desactivada... Para volver di: 'AVERY', para salir di 'ADIOS'"
                    print(">>> CERRANDO CAMARA <<<")
                    cerrar_camara()
                    hablar(txt)

            elif coincide(texto, VARIANTES4):
                print(">>> BUEN DIA :D <<<")
                hablar("Cerrando programa, buen día")
                cerrar_camara()
                break
except KeyboardInterrupt:
    print("\nInterrumpido. Cerrando...")
    cerrar_camara()
except Exception as e:
    print("Error inesperado:", e)
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()

"""
#AUDIO PREGRABADO



with SR.AudioFile('C:\\Users\\jdelr\\OneDrive\\Escritorio\\SCRIPT\\audios\\avery_test.wav') as source:
    audio = r.listen(source)

    try:
        print('espere un momento, el audio se esta leyendo...')
        text = r.recognize_google(audio, language='es-ES')
        time.sleep(1.5)
        if "avery" in text:
            print("Redirigiendo, espere un minuto")
        print(text)
    except:
        print('No se logro reconocer el audio')"""
