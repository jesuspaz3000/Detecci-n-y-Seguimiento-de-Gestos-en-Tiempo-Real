import cv2  # Importa OpenCV para el procesamiento de imágenes
import mediapipe as mp  # Importa MediaPipe para detección de manos y puntos clave
import math

class detectorManos():
    def __init__(self, mode=False, maxManos=2, confDeteccion=0.5, confSegui=0.5):
        # Constructor de la clase que inicializa la detección de manos con configuraciones específicas
        self.mode = mode  # Modo de detección (estático o dinámico)
        self.maxManos = maxManos  # Número máximo de manos a detectar
        self.confDeteccion = confDeteccion  # Umbral de confianza para la detección
        self.confSegui = confSegui  # Umbral de confianza para el seguimiento

        # Inicializa la solución de manos de MediaPipe y el dibujante de los puntos y conexiones
        self.mpManos = mp.solutions.hands
        self.manos = self.mpManos.Hands(self.mode, self.maxManos, 
                                        model_complexity=1,
                                        min_detection_confidence=self.confDeteccion, 
                                        min_tracking_confidence=self.confSegui)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]  # Índices de las puntas de los dedos
        
    def dedosArriba(self):
        # Devuelve una lista indicando qué dedos están arriba (1) y cuáles no (0)
        dedos = []
        # Pulgar: Cambiar la lógica según sea necesario
        if self.lista[self.tip[0]][1] < self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        # Otros dedos
        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos

    def encontrarManos(self, frame, dibujar=True):
        # Procesa la imagen de entrada y dibuja las manos detectadas
        imgColor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convierte la imagen a RGB
        self.resultados = self.manos.process(imgColor)  # Procesa la imagen con MediaPipe Hands
        
        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    # Dibuja los puntos clave y las conexiones de la mano en la imagen
                    self.dibujo.draw_landmarks(frame, mano, self.mpManos.HAND_CONNECTIONS)
        return frame

    def encontrarPosicion(self, frame, manoNum=0, dibujar=True):
        # Encuentra la posición de los puntos clave de la mano en la imagen
        xlista = []
        ylista = []
        bbox = []  # Inicializa una lista para la caja delimitadora
        self.lista = []  # Inicializa la lista de puntos clave
        
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[manoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    # Dibuja un círculo en cada punto clave
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)
            
            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax
            if dibujar:
                # Dibuja la caja delimitadora alrededor de la mano
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        
        return self.lista, bbox  # Devuelve la lista de puntos clave y la caja delimitadora

    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        # Calcula la distancia entre dos puntos clave y dibuja una visualización en la imagen
        x1, y1 = self.lista[p1][1:]
        x2, y2 = self.lista[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if dibujar:
            # Dibuja la línea, los círculos en los puntos clave y el punto central
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)  # Calcula la distancia euclidiana
        
        return length, frame, [x1, y1, x2, y2, cx, cy]  # Devuelve la distancia y la imagen con las visualizaciones
