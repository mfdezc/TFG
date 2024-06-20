import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# Función para verificar si una posición es válida
def is_valid_move(heatmap, new_positions):
    rows, cols = heatmap.shape
    for (r, c) in new_positions:
        if r < 0 or r >= rows or c < 0 or c >= cols or heatmap[r, c] in [0, 4, 5, 6]:
            return False
    return True

# Función para obtener un fotograma
def obtener_fotograma(video_path, numero_fotograma):
    """
    Obtiene un fotograma específico de un archivo de video.
    
    Args:
        video_path (str): Ruta al archivo de video.
        numero_fotograma (int): Número del fotograma que se desea obtener (empezando desde 0).
    
    Returns:
        frame (numpy.ndarray): El fotograma solicitado, o None si no se puede leer el fotograma.
    """
    # Abrir el archivo de video
    cap = cv2.VideoCapture(video_path)
    
    # Verificar si el video se abrió correctamente
    if not cap.isOpened():
        print(f"Error al abrir el archivo de video: {video_path}")
        return None
    
    # Establecer la posición del video al fotograma deseado
    cap.set(cv2.CAP_PROP_POS_FRAMES, numero_fotograma)
    
    # Leer el fotograma
    ret, frame = cap.read()
    frame = cv2.resize(frame, (480,640))
    
    # Liberar el objeto VideoCapture
    cap.release()
    
    if not ret:
        print(f"Error al leer el fotograma número {numero_fotograma}")
        return None
    
    return frame

# Función para mover el robot
# Variables globales para almacenar la dirección de movimiento anterior
ultima_direccion = None

def move_robot(heatmap, robot_legs):
    global izquierda
    global derecha
    global ultima_direccion
    global orientacion_patas

    # Definir las direcciones de movimiento
    move_vector_up = (-50, 0)
    move_vector_right = (0, 50)
    move_vector_left = (0, -50)

    # Intentar mover hacia arriba
    new_positions_up = [(r + move_vector_up[0], c + move_vector_up[1]) for (r, c) in robot_legs]
    if is_valid_move(heatmap, new_positions_up):
        print("Robot moved up")
        ultima_direccion = 'up'        
        return new_positions_up

    # Si no puede moverse hacia arriba, intentar mover hacia la derecha
    if ultima_direccion != 'left':
        new_positions_right = [(r + move_vector_right[0], c + move_vector_right[1]) for (r, c) in robot_legs]
        if is_valid_move(heatmap, new_positions_right):
            print("Robot moved right")
            ultima_direccion = 'right'
            return new_positions_right

    # Si no puede moverse hacia la derecha, intentar mover hacia la izquierda
    if ultima_direccion != 'right':
        new_positions_left = [(r + move_vector_left[0], c + move_vector_left[1]) for (r, c) in robot_legs]
        if is_valid_move(heatmap, new_positions_left):
            print("Robot moved left")
            ultima_direccion = 'left'
            return new_positions_left

    # Si no puede moverse en ninguna dirección, quedarse en el lugar
    print("El movimiento está bloqueado por obstáculos.")
    return None


# Función para visualizar el mapa y la posición del robot
def visualize_map(heatmap, robot_legs):
    plt.imshow(heatmap, interpolation='nearest')
    plt.title("Mapa de Calor y Posición del Robot")
    robot_legs_array = np.array(robot_legs)
    plt.scatter(robot_legs_array[:, 1], robot_legs_array[:, 0], c='red')
    plt.show()
