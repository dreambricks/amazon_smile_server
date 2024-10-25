import cv2
import os


nome_da_pasta = "static/pedro/feliz"

os.makedirs(nome_da_pasta, exist_ok=True)

captura = cv2.VideoCapture(1)

if not captura.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

numero_de_frames = 300

gravando = False

print("Pressione 'r' para iniciar a gravação e 'q' para sair.")

while True:

    ret, frame = captura.read()

    if ret:
        cv2.imshow("Webcam", frame)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('r'):
            gravando = True
            print("Gravação iniciada...")

        elif tecla == ord('q'):
            print("Encerrando...")
            break

        if gravando and numero_de_frames > 0:
            caminho_do_arquivo = os.path.join(nome_da_pasta, f"frame_{numero_de_frames:04d}.png")
            cv2.imwrite(caminho_do_arquivo, frame)
            print(f"Frame salvo em {caminho_do_arquivo}")
            numero_de_frames -= 1

            if numero_de_frames == 0:
                print("Gravação finalizada.")
                gravando = False
    else:
        print("Falha ao capturar frame.")
        break

captura.release()
cv2.destroyAllWindows()
