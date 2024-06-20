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
import mapa
import movement

#se carga el modelo entrenado
model= YOLO("./best.pt")
colors = []
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

# Coordenadas iniciales de las patas del robot
robot_legs = [(480, 20), (440, 30), (480, 70), (440, 60)]
all_positions = []
all_positions.append(robot_legs)
izquierda=0
derecha=0
ultima_direccion=None
orientacion_patas='vertical'
# Simulación del movimiento del robot basado en la actualización del mapa de calor
directions = ['up', 'right', 'down', 'left']
current_direction_index = 0
videopath = "./tfg_maria.mp4"
capture=cv2.VideoCapture("./tfg_maria.mp4")
fotograma = 0

for step in range(25):  # Simula 25 pasos del robot
    # Actualizar el mapa de calor desde la cámara
    imagen = movement.obtener_fotograma(videopath, fotograma)
    
    # Mostrar la imagen
    fotograma_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    plt.imshow(fotograma_rgb)
    plt.axis('off')
    plt.title(f"Fotograma {fotograma}")
    plt.show()
    
    heatmap = np.zeros((480, 640), dtype=int)
    mapa.actualiza_mapa(model, heatmap, imagen)
        
    # Mover el robot
    new_positions = movement.move_robot(heatmap, robot_legs)
    if new_positions is None:
            # Mostrar mensaje
            print("Cambiar posición del robot")
            # Salir del bucle principal
            break

    # Actualizar la posición del robot
    robot_legs = new_positions

    all_positions.append(robot_legs.copy())

    # Visualizar el mapa, la nueva posición del robot y la trayectoria
    plt.figure(figsize=(15, 5))
    
    # Mostrar el mapa de calor
    plt.subplot(1, 2, 1)
    # Visualizar el mapa y la nueva posición del robot
    movement.visualize_map(heatmap, robot_legs)
    
    # Mostrar la trayectoria del robot sobre fondo blanco
    plt.subplot(1, 2, 2)
    plt.imshow(np.ones((480, 640, 3)))  # Fondo blanco
    for step_positions in all_positions:
        for leg in step_positions:
            plt.plot(leg[1], leg[0], 'bo', alpha=0.5)  # marcamos las trayectorias de las patas del robot
    plt.title("Trayectoria del Robot")
    
    plt.show()
    
    # Cambiar la dirección para la próxima iteración (esto es solo un ejemplo)
    current_direction_index = (current_direction_index + 1) % len(directions)
    
    # Verificar condición de parada
    if(robot_legs[3][0] <= 50 and robot_legs[3][1] >= 590):
        print("El robot ha acabado")
        break

    # Incrementar el fotograma a razón de 20 cada paso de robot
    fotograma += 5