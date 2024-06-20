from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from numpy import savetxt
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.path as mpath


def actualiza_mapa(model, matriz, frame):
    results= model.predict(frame)
    # if you want all classes
    yolo_classes = list(model.names.values())
    classes_ids = [yolo_classes.index(clas) for clas in yolo_classes]

    colors = [
        (255, 0, 0),     # Rojo
        (0, 255, 0),     # Verde
        (0, 0, 255),     # Azul
        (255, 255, 0),   # Amarillo
        (0, 255, 255),   # Cian
        (255, 0, 255),   # Magenta
        (192, 192, 192), # Gris
        (255, 165, 0),   # Naranja
        (128, 0, 128),   # Púrpura
        (0, 128, 0)      # Verde oscuro
    ]

    for result in results: 
        for mask, box in zip(result.masks.xy, result.boxes):
            points = np.int32([mask])
            color_number = classes_ids.index(int(box.cls[0]))
            cv2.polylines(frame, points, True, colors[color_number], 1)
            cv2.fillPoly(frame, points, colors[color_number])

            # Crear la máscara booleana
            height, width = matriz.shape
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            x, y = x.flatten(), y.flatten()
            points_grid = np.vstack((x, y)).T

            path = mpath.Path(points[0])
            mask = path.contains_points(points_grid).reshape((height, width))

            # Sumar el valor a las celdas dentro de la polilínea
            matriz[mask] += color_number