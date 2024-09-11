#!/usr/bin/env python3

import sys
import os
import cv2
import numpy as np
import rospy
import struct
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Twist, Vector3
from ultralytics import YOLO
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from tf.transformations import quaternion_from_euler
import tf2_ros
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import tf2_msgs
import extrae_fotograma
from sensor_msgs.msg import CompressedImage, Image
import threading
from scipy.ndimage import gaussian_filter
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from math import pow, atan2, sqrt
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import mapa

# Añadir el directorio actual al PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
image_data= None

def image_callback(data):
    global image_data
        # Guardar la imagen recibida y notificar al hilo de procesamiento
    with lock:
        image_data = data

# Inicializar ROS y sus nodos
rospy.init_node('heatmap_publisher', anonymous=True)
pub = rospy.Publisher('/heatmap_pointcloud', PointCloud2, queue_size=10)
marker_pub = rospy.Publisher('/visualization_marker', Marker, queue_size=10)
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
image_sub = rospy.Subscriber("/camera1/color/image_raw/compressed", CompressedImage, image_callback)
image_pub = rospy.Publisher("/camera/image_detected_c", Image, queue_size=10)
lock = threading.Lock()
bridge = CvBridge()

# Tamaño deseado para la imagen de visualización
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Imagen compuesta global
composite_image = None

# ID del marcador
marker_id = 0



#Funcion para mover al robot con una 2D navigation goal a un punto determinado
def send_goal(x, y):
    goal = PoseStamped()
    goal.header.frame_id = "odom"  # El marco de referencia del objetivo (puede variar)
    goal.header.stamp = rospy.Time.now()

    # Establecer la posición del objetivo
    goal.pose.position = Point(x, y, 0.0)

    # Calcular la orientación para girar 90 grados (sentido antihorario) en radianes
    q = quaternion_from_euler(0, 0, 0)  # 1.57 radianes = 90 grados

    # Establecer la orientación del objetivo
    goal.pose.orientation.x = q[0]
    goal.pose.orientation.y = q[1]
    goal.pose.orientation.z = q[2]
    goal.pose.orientation.w = q[3]

    # Publicar el objetivo en el topic /move_base_simple/goal
    goal_pub.publish(goal)
    rospy.loginfo(f"Enviando goal a ({x}, {y})")

# Función para convertir el mapa de calor a una imagen OpenCV
def heatmap_to_image(heatmap):
    # Redimensionar el mapa de calor
    heatmap_resized = cv2.resize(heatmap, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    heatmap_normalized = np.uint8(255 * heatmap_resized / np.max(heatmap_resized))

    heatmap_color = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_VIRIDIS)

    return heatmap_color


def image_to_pointcloud(heatmap, image, scale_factor): 
    points = []
    height, width, channels = image.shape

    # Crear un array vacío para almacenar los valores z
    z_values = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]  # Obtener los valores BGR de la imagen compuesta
            z = (0.299 * r + 0.587 * g + 0.114 * b) / 255.0  # Normalización basada en la luminosidad de los colores
            z_values[y, x] = z  # Guardar los valores z en el array

    smoothed_z_values = smooth_z_values(z_values, sigma=2.0) 

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]
            z = smoothed_z_values[y, x]  # Usar los valores z suavizados
            scaled_y = float(x) / width * 0.05 * 10.0 - 0.25  # Escalar el rango en x
            scaled_x = float(y) / height * scale_factor * 10.0  # Escalar el rango en y
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            # Crear un punto en la nube de puntos
            points.append([scaled_x, scaled_y, z * 0.15, rgb])
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'odom'

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('rgb', 12, PointField.UINT32, 1),
    ]

    pointcloud = pcl2.create_cloud(header, fields, points)
    return pointcloud

def smooth_z_values(z_values, sigma=1.0):
    """
    Suaviza los valores de z utilizando un filtro de promedio móvil.
    """
    return gaussian_filter(z_values, sigma=sigma)

#Función para publicar marcadores 
def publish_marker(x, y, color):
    global marker_id

    marker = Marker()
    marker.header.frame_id = "odom"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "basic_shapes"
    marker.id = marker_id
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.position.x = x
    marker.pose.position.y = y
    marker.pose.position.z = 0
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.025
    marker.scale.y = 0.025
    marker.scale.z = 0.025

    marker.color.r = color
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0

    marker.lifetime = rospy.Duration()

    marker_pub.publish(marker)
    marker_id += 1

    return x, y

#Función que transforma las coordenadas de la imagen a la nube de puntos
def image_to_ros_coordinates(mean, image_width, image_height, scale_factor):
    #x= leg[0]
    #y=leg[1]
    xm=mean[0]
    ym= mean[1]
    #ros_y = float(x) / (image_width) * 0.05 * 10.0 -0.25 # Escalar el rango en x
    #ros_x = float(y) / (image_height) * scale_factor * 10.0# Escalar el rango en y
    mean_y = float(xm) / (image_width) * 0.05 * 10.0 -0.25 # Escalar el rango en x
    mean_x = float(ym) / (image_height) * scale_factor * 10.0# Escalar el rango en y
    if image_height==480: 
        return mean_x, mean_y
    else:
        #ros_x = ros_x +(image_height/480)*0.33 -0.33
        mean_x = mean_x +(image_height/480)*0.33 -0.33
        return mean_x, mean_y
coordinatesLF= None
coordinatesRF= None
coordinatesLH= None
coordinatesRH= None
coordinatesbase=None
tf_published = False

#funcion que recibe las posiciones que publica el topico \tf
def tf_callback(msg):
    global coordinatesLF, coordinatesLH, coordinatesRF, coordinatesRH, coordinatesbase , tf_published
    global distanceLF, distanceLH, distanceRF, distanceRH
    if tf_published: 
        return
    else: 
        try:
            transb = tf_buffer.lookup_transform('odom', 'base_imu', rospy.Time())
            xbase = transb.transform.translation.x
            ybase= transb.transform.translation.y
            zbase = transb.transform.translation.z
            #rospy.loginfo(f"Coordenadas de 'LF_foot' en 'odom': ({xbase}, {ybase}, {zbase})")
            coordinatesbase=(xbase, ybase, zbase)
            # Obtener la transformación de 'LF_foot' a 'odom'
            transLF = tf_buffer.lookup_transform('odom', 'LF_FOOT', rospy.Time())
            xLF = transLF.transform.translation.x
            yLF = transLF.transform.translation.y
            zLF = transLF.transform.translation.z
            #rospy.loginfo(f"Coordenadas de 'LF_foot' en 'odom': ({xLF}, {yLF}, {zLF})")
            coordinatesLF=(xLF, yLF, zLF)
            # Obtener la transformación de 'LF_foot' a 'odom'
            transLH = tf_buffer.lookup_transform('odom', 'LH_FOOT', rospy.Time())
            xLH = transLH.transform.translation.x
            yLH = transLH.transform.translation.y
            zLH = transLH.transform.translation.z
            #rospy.loginfo(f"Coordenadas de 'LH_foot' en 'odom': ({xLH}, {yLH}, {zLH})")
            coordinatesLH=(xLH, yLH, zLH)
            # Obtener la transformación de 'LF_foot' a 'odom'
            transRF = tf_buffer.lookup_transform('odom', 'RF_FOOT', rospy.Time())
            xRF = transRF.transform.translation.x
            yRF = transRF.transform.translation.y
            zRF = transRF.transform.translation.z
            #rospy.loginfo(f"Coordenadas de 'RF_foot' en 'odom': ({xRF}, {yRF}, {zRF})")
            coordinatesRF=(xRF, yRF, zRF)
            # Obtener la transformación de 'LF_foot' a 'odom'
            transRH = tf_buffer.lookup_transform('odom', 'RH_FOOT', rospy.Time())
            xRH = transRH.transform.translation.x
            yRH = transRH.transform.translation.y
            zRH = transRH.transform.translation.z
            #rospy.loginfo(f"Coordenadas de 'RH_foot' en 'odom': ({xRH}, {yRH}, {zRH})")
            tf_published=True
            coordinatesRH=(xRH, yRH, zRH)
            distanceLF= xbase - xLF, yLF - ybase
            distanceLH= xbase - xLH, ybase - yLH
            distanceRF= xRF - xbase, yRF - ybase
            distanceRH= xRH - xbase, ybase - yRH
            # print(f'La distancia entre LF y el centro es de {distanceLF} en la nube')
            # print(f'La distancia entre LH y el centro es de {distanceLH} en la nube')
            # print(f'La distancia entre RF y el centro es de {distanceRF} en la nube')
            # print(f'La distancia entre RH y el centro es de {distanceRH} en la nube')
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"No se pudo obtener la transformación: {e}")  

#Funcion que convierte las coorddenadas proporcionadas por el \tf a pixeles
def ros_to_image_coordinates( image_width, image_height, scale_factor): #COMPROBAR ESTA FUNCION
    global coordinatesLF, coordinatesLH, coordinatesRF, coordinatesRH 
    global distanceLF, distanceLH, distanceRF, distanceRH
    global dLF_pixel, dLH_pixel, dRF_pixel, dRH_pixel 
    d1lf, d2lf = distanceLF
    d1lh, d2lh = distanceLH
    d1rf, d2rf = distanceRF
    d1rh, d2rh = distanceRH
    if d1lf<0: 
        d1lf=-d1lf
    if d2lf<0: 
        d2lf=-d2lf
    if d1rf<0: 
        d1rf=-d1rf
    if d2rf<0: 
        d2rf=-d2rf
    if d1lh<0: 
        d1lh=-d1lh
    if d2lh<0: 
        d2lh=-d2lh
    if d2rh<0: 
        d2rh=-d2rh
    if d1rh<0: 
        d1rh=-d1rh

    #print(f"Original d1: {d1}, Original d2: {d2}")

    if image_height == 480:
        d1LF = int(float(d2lf) * image_width / ( 0.05* 10.0))
        d2LF = int(float(d1lf) * image_height / (0.05 * 10.0))

        d1LH = int(float(d2lh) * image_width / (0.05 * 10.0))
        d2LH = int(float(d1lh) * image_height / (0.05 * 10.0))

        d1RF = int(float(d2rf) * image_width / (0.05 * 10.0))
        d2RF = int(float(d1rf) * image_height / (0.05 * 10.0))

        d1RH = int(float(d2rh) * image_width / (0.05 * 10.0))
        d2RH = int(float(d1rh) * image_height / (0.05 * 10.0))
    else:
        d2_adj = d2lf - (image_height / 480) * 0.33 + 0.33
        d1LF = int(float(d1lf) * image_width / (scale_factor2 * 10.0))
        d2LF = int(float(d2_adj) * image_height / (0.05 * 10.0))

        d2_adj = d2lh - (image_height / 480) * 0.33 + 0.33
        d1LH = int(float(d1lh) * image_width / (scale_factor2 * 10.0))
        d2LH = int(float(d2_adj) * image_height / (0.05 * 10.0))

        d2_adj = d2rf - (image_height / 480) * 0.33 + 0.33
        d1RF = int(float(d1rf) * image_width / (scale_factor2 * 10.0))
        d2RF = int(float(d2_adj) * image_height / (0.05 * 10.0))

        d2_adj = d2rh - (image_height / 480) * 0.33 + 0.33
        d1RH = int(float(d1rh) * image_width / (scale_factor2 * 10.0))
        d2RH = int(float(d2_adj) * image_height / (0.05 * 10.0))
    dLF_pixel=[d1LF, d2LF]  
    dLH_pixel=[d1LH, d2LH]  
    dRF_pixel=[d1RF, d2RF]  
    dRH_pixel=[d1RH, d2RH]  
    
    print(f"Distancia en píxeles LF: {d1LF}, {d2LF}")
    print(f"Distancia en píxeles LH: {d1LH}, {d2LH}")
    print(f"Distancia en píxeles RF: {d1RF}, {d2RF}")
    print(f"Distancia en píxeles RH: {d1RH}, {d2RH}")
    
#Función para mover el robot, comprobando que las patas no ocupen posiciones con obstáculos si es posible
def move_robot(heatmap): 
    global dLF_pixel, dLH_pixel, dRF_pixel, dRH_pixel 
    rows, cols = heatmap.shape
    print(rows, cols)
    dist_1=150
    dist_2=140
    r=150
    c=280
    for i in range(r, rows-150):
        for j in range(c, cols-200):
            if (0 <= i - dLH_pixel[0] < heatmap.shape[0] and 0 <= j - dLH_pixel[1] < heatmap.shape[1] and
            0 <= i - dLF_pixel[0] < heatmap.shape[0] and 0 <= j + dLF_pixel[1] < heatmap.shape[1] and
            0 <= i + dRF_pixel[0] < heatmap.shape[0] and 0 <= j + dRF_pixel[1] < heatmap.shape[1] and
            0 <= i + dRH_pixel[0] < heatmap.shape[0] and 0 <= j - dRH_pixel[1] < heatmap.shape[1] and
            heatmap[i - dLH_pixel[0]][j - dLH_pixel[1]] not in [4, 5, 6] and
            heatmap[i - dLF_pixel[0]][j + dLF_pixel[1]] not in [4, 5, 6] and
            heatmap[i + dRF_pixel[0]][j + dRF_pixel[1]] not in [4, 5, 6] and
            heatmap[i + dRH_pixel[0]][j - dRH_pixel[1]] not in [4, 5, 6]):
                g_x= i
                g_y= j
                goal_point=[g_x, g_y]
                pata1= [i-dLH_pixel[0], j-dLH_pixel[1]]#izquierda detras
                pata2= [i-dLF_pixel[0], j+dLF_pixel[1]]#izquierda delante
                pata3= [i+dRF_pixel[0], j+dRF_pixel[1]]#derecha delante
                pata4= [i+dRH_pixel[0], j-dRH_pixel[1]]
                print(f'posicion correcta establecida en {g_x}, {g_y} ') 
                return goal_point, pata1, pata2, pata3, pata4
                break
            else:
                pata1= [240-dLH_pixel[0], 320-dLH_pixel[1]]#izquierda detras
                pata2= [240-dLF_pixel[0], 320+dLF_pixel[1] ]#izquierda delante
                pata3= [240+dRF_pixel[0], 320+dRF_pixel[1]]#derecha delante
                pata4= [240+dRH_pixel[0], 320-dRH_pixel[1]]
                goal_point=[120, 160]
                continue
    print('no hay manera de evitarlos, moviendo al centro')
    return goal_point, pata1, pata2, pata3, pata4

def get_current_position():
    data = rospy.wait_for_message("/odom", Odometry)
    x = data.pose.pose.position.x
    y = data.pose.pose.position.y
    return x, y

def get_current_orientation():
    data = rospy.wait_for_message("/odom", Odometry)
    orientation_q = data.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
    return yaw


def calculate_velocity(x_goal, y_goal, x_current, y_current):
    global distance_y, distance_x
    # Ganancias del controlador
    K_linear = 0.6
    K_angular = 2.0

    # Calcular el error
    distance = ((x_goal - x_current)**2 + (y_goal - y_current)**2)**0.5
    angle_to_goal = atan2(y_goal - y_current, x_goal - x_current)
    distance_x = x_goal - x_current
    distance_y = y_goal - y_current
    
    # Velocidades deseadas
    current_angle = get_current_orientation()
    linear_velocity_x = K_linear * distance_x
    linear_velocity_y = K_linear * distance_y
    
    angular_velocity = K_angular * (angle_to_goal - current_angle)

    return linear_velocity_x, linear_velocity_y, angular_velocity


def move_robot_to_goal(x_goal, y_goal):
    
    rate = rospy.Rate(10)  # 10 Hz

    vel_msg = Twist()

    while not rospy.is_shutdown():
        # Supongamos que ya tenemos la posición actual del robot
        x_current, y_current = get_current_position()
        distance = ((x_goal - x_current)**2 + (y_goal - y_current)**2)**0.5
        
        

        linear_velocity_x, linear_velocity_y, angular_velocity = calculate_velocity(x_goal, y_goal, x_current, y_current)
        #print(f'La distancia es {linear_velocity_x, linear_velocity_y}')
        # Asignar velocidades calculadas
        vel_msg.linear.x = linear_velocity_x
        vel_msg.linear.y = linear_velocity_y
        vel_msg.angular.z = 0

        # Publicar la velocidad
        velocity_publisher.publish(vel_msg)

        # Verificar si estamos cerca del objetivo
        if distance < 0.05:  # Umbral de cercanía
            break




velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)
rospy.Subscriber('/tf', tf2_msgs.msg.TFMessage, tf_callback)
model = YOLO("/home/mariafdezc/3m/MODELO3M/weights/best.pt")
videopath = "/home/mariafdezc/videos:ARTUR/video_pruebas.mp4"
capture = cv2.VideoCapture(videopath)
rate = rospy.Rate(10)



scale_factor = 0.0
scale_factor2=0


legs=[]
heatmap_concatenado=[]
frame_index=130
mvto=0
#move_robot_to_goal(0,0)
ros_to_image_coordinates(image_width=640, image_height=480, scale_factor=0.05)

for index in range(4):
    
    #front_legs = [(0, 50), (0, 350)]
    mean=[]
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = capture.read()
    scale_factor2+=0.05
    scale_factor += 0.033
    #np_arr = np.frombuffer(image_data.data, np.uint8) #para las pruebas con el robot
    #cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #frame = cv2.resize(cv_image, (640, 480))
    heatmap = np.zeros((480, 640), dtype=np.float32)
    heatmap= mapa.actualiza_mapa(model, heatmap, frame) 
    heatmap_image = heatmap_to_image(heatmap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if composite_image is None:
        image_espejo= cv2.flip(heatmap_image, 0)
        composite_image = image_espejo
        print('hola', composite_image.shape)
    else:
        image_espejo= cv2.flip(heatmap_image, 0)
        composite_image = np.vstack((composite_image, image_espejo))
        print(composite_image.shape)
    image_height, image_width, _ = composite_image.shape
    print('Publicando nube')
    pointcloud_msg = image_to_pointcloud(heatmap, composite_image, scale_factor)
    rospy.loginfo('Publicando nube de puntos')
    pub.publish(pointcloud_msg)
    print('Nube publicada')
    print('ha entrado')
    mean, pata1, pata2, pata3, pata4= move_robot(heatmap)
    print('calculando coordenadas de patas')
    mean_x, mean_y= image_to_ros_coordinates(mean, image_width, image_height, scale_factor )
    print('Ha acabado de calcular')
    publish_marker(mean_x, mean_y, 0)
    publish_marker(mean_x+distanceLF[0], mean_y + distanceLF[1], 1)
    publish_marker(mean_x+distanceLH[0], mean_y + distanceLH[1], 1)
    publish_marker(mean_x-distanceRF[0], mean_y + distanceRF[1], 1)
    publish_marker(mean_x-distanceRH[0], mean_y + distanceRH[1], 1)
    if mvto==1: 
        move_robot_to_goal(mean_x, mean_y)
    else:
        send_goal(mean_x, mean_y)
    print('ha acabado de moverse')

    frame_index+=75#igual hay que cambiar este parametro para 

    
    

    legs=[]
capture.release()
cv2.destroyAllWindows()
rospy.spin()

