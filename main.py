import cv2

def movementv2():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        retStart, frameStart = cap.read()
        retEnd, frameEnd = cap.read()

        diff = cv2.absdiff(frameStart, frameEnd)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        #threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        test, threshold = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(threshold, None, iterations=10)
        contours, test1 = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frameStart, contours, -1, (255, 0, 0), 2)
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 700:
                continue
            cv2.rectangle(frameStart, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Frame', frameStart)
        frameStart = frameEnd
        retEnd, frameEnd = cap.read()

        if cv2.waitKey(30) & 0xFF == 27:  # Presiona Esc para salir
            break

    cap.release()
    cv2.destroyAllWindows()


def movementv1():
    # Use a breakpoint in the code line below to debug your script.

    # Inicializa el capturador de video con la cámara o un archivo de video
    cap = cv2.VideoCapture(0)  # Reemplaza 'video.mp4' con la ruta de tu video o 0 para la cámara

    # Inicializa el algoritmo de sustracción de fondo
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Aplica el algoritmo de sustracción de fondo al frame actual
        fgmask = fgbg.apply(frame)

        # Aplica una serie de operaciones morfológicas para eliminar el ruido
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, None)

        # Encuentra los contornos de los objetos en movimiento
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Dibuja los contornos en el frame original
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Puedes ajustar este valor según tu aplicación
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Muestra el frame original y el frame con los objetos en movimiento resaltados
        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', fgmask)

        if cv2.waitKey(30) & 0xFF == 27:  # Presiona Esc para salir
            break

    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    movementv2()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
