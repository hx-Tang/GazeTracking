"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import numpy as np
import dlib
import time
import math
import pandas as pd
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68

def _largest_face(dets):
    if len(dets) == 1:
        return 0

    face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]

    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area = face_areas[index]

    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index

def get_image_points_from_landmark_shape(landmark_shape):
    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None
    
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                (landmark_shape.part(30).x, landmark_shape.part(30).y),     # Nose tip
                                (landmark_shape.part(8).x, landmark_shape.part(8).y),     # Chin
                                (landmark_shape.part(36).x, landmark_shape.part(36).y),     # Left eye left corner
                                (landmark_shape.part(45).x, landmark_shape.part(45).y),     # Right eye right corne
                                (landmark_shape.part(48).x, landmark_shape.part(48).y),     # Left Mouth corner
                                (landmark_shape.part(54).x, landmark_shape.part(54).y)      # Right mouth corner
                            ], dtype="double")

    return 0, image_points
    
# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_image_points(img):
                            
    #gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )  # 图片调整为灰色
    dets = detector( img, 0 )

    if 0 == len( dets ):
        #print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)

    return get_image_points_from_landmark_shape(landmark_shape)


# 获取旋转向量和平移向量                        
def get_pose_estimation(img_size, image_points ):
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                             
                            ])
     
    # Camera internals
     
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
     
    #print("Camera Matrix :{}".format(camera_matrix))
     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
 
    #print("Rotation Vector:\n {}".format(rotation_vector))
    #print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs

# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)
    
    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    #print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    
    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    
	# 单位转换：将弧度转换为度
    Y = int((pitch/math.pi)*180)
    X = int((yaw/math.pi)*180)
    Z = int((roll/math.pi)*180)
    
    return 0, Y, X, Z

def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None
    
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie, image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None
    
        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)
        return 0, pitch, yaw, roll
    
    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None


if __name__ == '__main__':
    data = np.array(['', 'Face_Y', 'Face_X', 'Face_Z', 'Eye', 'Concentration level'])
    count = 0
    temp_data = np.array([0, 0, 0, 0, '', 0])
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        count = count + 1
        temp_data[0]=count
        start_time = time.time()
        # We get a new frame from the webcam
        ret, frame = webcam.read()
        if ret != True:
            print('read frame failed!')
            continue
        size = frame.shape

        if size[0] > 700:
            h = size[0] / 3
            w = size[1] / 3
            frame = cv2.resize(frame, (int( w ), int( h )), interpolation=cv2.INTER_CUBIC )
            size = frame.shape
     
        ret, image_points = get_image_points(frame)
        if ret != 0:
            print('get_image_points failed')
            temp_data[1] = 1000
            temp_data[2] = 1000
            temp_data[3] = 1000
            temp_data[5] = 0
            data = np.vstack((data,temp_data))
            continue
        
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)
        if ret != True:
            print('get_pose_estimation failed')
            continue
       
        
        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)
        temp_data[1] = pitch
        temp_data[2] = yaw
        temp_data[3] = roll
        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose
         
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
         
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
         
         
        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        # We send this frame to GazeTracking to analyze it
        gaze.refresh(frame)

        frame = gaze.annotated_frame()
        text = ""

        # if gaze.is_blinking():
        #     text = "Blinking"

        if gaze.vertical_ratio() and gaze.horizontal_ratio():
            print('v'+str(gaze.vertical_ratio())+'    h'+str(gaze.horizontal_ratio()))

        if gaze.is_right():
            text = "Looking right"
            temp_data[4] = 'right'
        elif gaze.is_left():
            text = "Looking left"
            temp_data[4] = 'left'
        elif gaze.is_up():
            text = "Looking up"
            temp_data[4] = 'up'
        elif gaze.is_down():
            text = "Looking down"
            temp_data[4] = 'down'
        elif gaze.is_center():
            text = "Looking center"
            temp_data[4] = 'center'
        else:
            temp_data[4] = 'no'

        cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

        left_pupil = gaze.pupil_left_coords()
        right_pupil = gaze.pupil_right_coords()
        cv2.line(frame, p1, p2, (255,0,0), 2)
        cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
        cv2.putText(frame, euler_angle_str, (0, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1 )
        cv2.imshow("Demo", frame)
        if int(temp_data[3]) > 10 or int(temp_data[3]) < -10:
            temp_data[5] = 1
        elif int(temp_data[2]) > 40 or int(temp_data[2]) < -40:
            temp_data[5] = 2
        elif temp_data[4] == 'no':
            temp_data[5] = 3
        elif temp_data[4] == 'left' or temp_data[4] == 'right':
            temp_data[5] = 4
        elif temp_data[4] == 'up' or temp_data[4] == 'down':
            temp_data[5] = 5
        else:
            temp_data[5] = 6
        if int(temp_data[2]) < -40 and temp_data[4] == 'right':
            temp_data[5] = 6
        if int(temp_data[2]) < -40 and temp_data[4] == 'right':
            temp_data[5] = 6
        if int(temp_data[1]) > 0 and temp_data[4] == 'down':
            temp_data[5] = 6
        if int(temp_data[1]) < 0 and temp_data[4] == 'up':
            temp_data[5] = 6
        data = np.vstack((data,temp_data))
        used_time = time.time() - start_time
        print("used_time:{} sec".format(round(used_time, 3)))
        if cv2.waitKey(1) == 27:
            break
    print(data)
    data1 = pd.DataFrame(data)
    data1.to_csv('data1.csv', header = False, index = False)
    webcam.release()
    cv2.destroyAllWindows()
