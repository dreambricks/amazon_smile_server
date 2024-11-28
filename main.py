import pickle
import cv2
from utils import FaceLandmarks
import keyboard
import socket
import time
import parameters as pm
import serial
import threading

# Configurações de UDP
UDP_IP = "127.0.0.1"  # IP de destino
UDP_PORT = pm.UDP_PORT
UDP_PORT_SENDER = pm.UDP_PORT_SENDER # Porta de destino
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Cria o socket UDP

port = pm.SERIAL_PORT
baudrate = pm.SERIAL_BAUDRATE

# Variáveis de controle
draw_mask = False
draw_stats = False
terminate_program = False
analyzing = False  # Indica se a análise está ativa
start_time = None  # Armazena o tempo inicial da análise


def serial_reader():
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"Conectado à porta {port} com baudrate {baudrate}")
    except serial.SerialException as e:
        print(f"Erro ao abrir a porta serial: {e}")
        return

    try:
        while True:
            if ser.in_waiting > 0:
                mensagem = ser.readline().decode('utf-8').strip()
                print(f"Mensagem recebida: {mensagem}")
                serial_action(mensagem)

    except KeyboardInterrupt:
        print("Encerrando a leitura da porta serial.")
    finally:
        if ser.is_open:
            ser.close()
            print("Porta serial fechada.")

def serial_action(message):
    if message == "1":
        keyboard.press_and_release('m')

serial_thread = threading.Thread(target=serial_reader, daemon=True)
serial_thread.start()

# Função para escutar mensagens UDP
def listen_udp():
    global analyzing, start_time
    sock.bind(("0.0.0.0", UDP_PORT))  # Escuta todas as interfaces na porta definida
    while True:
        data, addr = sock.recvfrom(1024)  # Recebe mensagem
        if data.decode() == 'start':
            print("Recebido comando 'start'. Iniciando análise...")
            analyzing = True
            start_time = time.time()


# Inicializa o modelo e a captura de vídeo
fl = FaceLandmarks(static_image_mode=False)
emotions = ['happy', 'laughing', 'neutral']
emotions_pt = ['rindo', 'sorrindo', 'neutro']
emotions_idx_sorted = [1, 2, 0]

with open('./modeldb5', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(pm.CAM_INDEX)
ret, frame = cap.read()

# Thread para escutar mensagens UDP
import threading
udp_thread = threading.Thread(target=listen_udp)
udp_thread.daemon = True
udp_thread.start()

print('starting loop:', ret)
camera_fail_counter = 0
while not terminate_program and camera_fail_counter < 3:
    while ret:
        ret, frame = cap.read()
        if not ret:
            print("can't read the camera")
            break
        camera_fail_counter = 0

        # Se estiver analisando, verifica o tempo limite de 3 segundos
        if analyzing:
            face_landmarks = fl.get_face_landmarks(frame, draw=draw_mask)

            if len(face_landmarks) == 1404:
                output = model.predict_proba([face_landmarks])
                max_val = max(output[0])
                emotion = emotions_pt[output[0].tolist().index(max_val)]

                if draw_stats:
                    for idx, e_idx in enumerate(emotions_idx_sorted):
                        e = emotions_pt[idx]
                        text = f"{e} : {output[0][idx] * 100:.0f}%"
                        color = (0, 255, 0) if output[0][idx] == max_val else (0, 0, 255)
                        cv2.putText(frame,
                                    text,
                                    (10, frame.shape[0] - 10 - (len(emotions) - e_idx - 1) * 35),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    color,
                                    3)

                # Verifica se já passou o tempo limite de 3 segundos
                if time.time() - start_time >= 3:
                    print(f"Enviando emoção detectada: {emotion}")
                    sock.sendto(emotion.encode(), (UDP_IP, UDP_PORT_SENDER))  # Envia a emoção via UDP
                    analyzing = False  # Para a análise temporária

        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            print('normal exit')
            terminate_program = True
            break

    camera_fail_counter += 1

    # sleeps for 30 seconds
    time.sleep(30)

    print(f"Trying to re-initialize the camera for the {camera_fail_counter} time.")
    # try to re-initialize the camera
    cap = cv2.VideoCapture(pm.CAM_INDEX)
    ret, frame = cap.read()

if camera_fail_counter >= 3:
    print("Reinitialization of the camera didn't work. Finishing the program.")
else:
    print("Finishing the program")

cap.release()
cv2.destroyAllWindows()
