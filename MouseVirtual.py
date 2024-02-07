# Importación de librerías necesarias
import cv2 # Importa OpenCV para el procesamiento de imágenes
import numpy as np
import SeguimientoManos as sm  # Módulo para el seguimiento de manos
import autopy  # Librería para controlar el mouse

# Configuración inicial de la cámara y la pantalla
anchocam, altocam = 640, 480  # Dimensiones de la cámara
cuadro = 100  # Área de interacción en la cámara
anchopanta, altopanta = autopy.screen.size()  # Dimensiones de la pantalla
sua = 5  # Factor de suavizado para el movimiento del mouse
pubix, pubiy = 0, 0  # Posiciones anteriores del mouse
cubix, cubiy = 0, 0  # Posiciones actuales del mouse

# Inicialización de la cámara
cap = cv2.VideoCapture(0)
cap.set(3, anchocam)  # Ancho de la cámara
cap.set(4, altocam)  # Alto de la cámara

# Inicialización del detector de manos
detector = sm.detectorManos(maxManos=1)  # Uso de una sola mano

click_presionado = False  # Estado del clic del mouse

# Función para mover el mouse en la pantalla
def mover_mouse(x1, y1, pubix, pubiy, sua, anchopanta, cuadro, anchocam, altocam, click_presionado):
    # Conversión de coordenadas de la cámara a coordenadas de la pantalla
    x3 = np.interp(x1, (cuadro, anchocam - cuadro), (0, anchopanta))
    y3 = np.interp(y1, (cuadro, altocam - cuadro), (0, altopanta))

    # Suavizado de movimientos del mouse
    cubix = pubix + (x3 - pubix) / sua
    cubiy = pubiy + (y3 - pubiy) / sua

    # Asegurarse de que el mouse no salga de la pantalla
    cubix = max(0, min(anchopanta - 1, cubix))
    cubiy = max(0, min(altopanta - 1, cubiy))

    # Mover el mouse en la pantalla
    if 0 <= cubix <= anchopanta - 1 and 0 <= cubiy <= altopanta - 1:
        autopy.mouse.move(anchopanta - cubix, cubiy)

    # Mantener presionado el clic si está activo
    if click_presionado:
        autopy.mouse.toggle(down=True)

    return cubix, cubiy

# Bucle principal
while True:
    # Lectura de la cámara
    ret, frame = cap.read()
    frame = detector.encontrarManos(frame)  # Detección de manos en el frame
    lista, bbox = detector.encontrarPosicion(frame)  # Obtención de la posición de la mano

    # Procesamiento de la posición de los dedos
    if len(lista) != 0:
        x1, y1 = lista[8][1:]  # Coordenadas del dedo índice
        x2, y2 = lista[12][1:]  # Coordenadas del dedo medio

        dedos = detector.dedosArriba()  # Estado de los dedos
        longitud, frame, linea = detector.distancia(8, 12, frame) if len(lista) > 12 else (0, frame, [])

        # Modo movimiento o arrastre
        if dedos[1] == 1 and dedos[2] == 0:  # Si solo el dedo índice está arriba
            if click_presionado:
                # Liberar el clic solo si estaba presionado
                autopy.mouse.toggle(down=False)
                click_presionado = False

            # Movimiento del mouse sin arrastrar
            cubix, cubiy = mover_mouse(x1, y1, pubix, pubiy, sua, anchopanta, cuadro, anchocam, altocam, False)
            pubix, pubiy = cubix, cubiy

        elif dedos[1] == 1 and dedos[2] == 1:  # Si ambos, dedo índice y medio, están arriba
            if longitud < 30:  # Si los dedos están cerca uno del otro
                if not click_presionado:
                    # Presionar el clic solo si no estaba ya presionado
                    autopy.mouse.toggle(down=True)
                    click_presionado = True

                # Movimiento del mouse mientras se arrastra
                cubix, cubiy = mover_mouse(x1, y1, pubix, pubiy, sua, anchopanta, cuadro, anchocam, altocam, True)
                pubix, pubiy = cubix, cubiy
            else:
                if click_presionado:
                    # Liberar el clic si los dedos ya no están cerca uno del otro
                    autopy.mouse.toggle(down=False)
                    click_presionado = False
        
        else:  # Si ninguna condición de clic o movimiento se cumple
            if click_presionado:
                # Liberar el clic si estaba presionado
                autopy.mouse.toggle(down=False)
                click_presionado = False

    cv2.imshow("Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
