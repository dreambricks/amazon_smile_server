import pickle
import cv2
from utils import FaceLandmarks
import keyboard
import socket
import time

# Configurações de UDP
UDP_IP = "127.0.0.1"  # IP de destino
UDP_PORT = 5006        # Porta de destino
UDP_PORT_SENDER = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Cria o socket UDP

# Variáveis de controle
draw_mask = False
draw_stats = True
terminate_program = False
analyzing = False  # Indica se a análise está ativa
start_time = None  # Armazena o tempo inicial da análise

# Função callback de evento de teclado
def on_key_event(event):
    global draw_mask, draw_stats, terminate_program
    print(f"Key {event.name} was pressed")
    if event.name == 'm':
        draw_mask = not draw_mask
    elif event.name == 's':
        draw_stats = not draw_stats
    elif event.name == 'q':
        terminate_program = True

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

# Inicia o hook para eventos de teclado
keyboard.on_press(on_key_event)

# Inicializa o modelo e a captura de vídeo
fl = FaceLandmarks(static_image_mode=False)
emotions = ['happy', 'laughing', 'neutral']
emotions_pt = ['rindo', 'sorrindo', 'neutro']
emotions_idx_sorted = [1, 2, 0]

with open('./modeldb5', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Thread para escutar mensagens UDP
import threading
udp_thread = threading.Thread(target=listen_udp)
udp_thread.daemon = True
udp_thread.start()

while ret:
    ret, frame = cap.read()
    if terminate_program:
        break

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
        break

cap.release()
cv2.destroyAllWindows()
