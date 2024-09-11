from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.path as mpath

def actualiza_mapa(modelo, matriz, frame):
    results = modelo(frame)
    # if you want all classes
    yolo_classes = list(modelo.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]
    for result in results: 
        try: 
            for mask, box in zip(result.masks.xy, result.boxes):
                
                points = np.int32([mask])
                color_number = classes_ids.index(int(box.cls[0]))
                #cv2.polylines(frame, points, True, colors[color_number], 1)
                #cv2.fillPoly(frame, points, colors[color_number])

                # Crear la máscara booleana
                height, width = matriz.shape
                x, y = np.meshgrid(np.arange(width), np.arange(height))
                x, y = x.flatten(), y.flatten()
                points_grid = np.vstack((x, y)).T

                path = mpath.Path(points[0])
                mask = path.contains_points(points_grid).reshape((height, width))

                # Sumar el valor a las celdas dentro de la polilínea
                matriz[mask] = color_number 
        except: 
            continue
    return matriz
