# neural net
from keras.models import model_from_json
from keras.optimizers import Adam

# video imports
from imutils import face_utils
from imutils.video import VideoStream
import imutils
import cv2
import dlib

# various imports
import playsound
import time
from threading import Thread
import numpy as np
import math


# 3-D coordinates to be used for perspective n point algo
# Antropometric constant values of the human head.
# Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"

# X-Y-Z with X pointing forward and Y on the left.
# The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0])  # 0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0])  # 4
P3D_MENTON = np.float32([0.0, 0.0, -122.7])  # 8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0])  # 12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0])  # 16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0])  # 17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0])  # 26
P3D_SELLION = np.float32([0.0, 0.0, 0.0])  # 27
P3D_NOSE = np.float32([21.1, 0.0, -48.0])  # 30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0])  # 33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5, -5.0])  # 36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5, -5.0])  # 39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5, -5.0])  # 42
P3D_LEFT_EYE = np.float32([-20.0, 65.5, -5.0])  # 45
# P3D_LIP_RIGHT = np.float32([-20.0, 65.5, -5.0])  # 48
# P3D_LIP_LEFT = npnp.float32([-20.0, 65.5, -5.0])  # 54
P3D_STOMION = np.float32([10.0, 0.0, -75.0])  # 62

# The points to track
# These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
ALL_POINTS = list(range(0, 68))  # Used for debug only

# Distortion coefficients
camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

# This matrix contains the 3D points of the
# 11 landmarks we want to find. It has been
# obtained from antrophometric measurement
# on the human head.
landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                           P3D_GONION_RIGHT,
                           P3D_MENTON,
                           P3D_GONION_LEFT,
                           P3D_LEFT_SIDE,
                           P3D_FRONTAL_BREADTH_RIGHT,
                           P3D_FRONTAL_BREADTH_LEFT,
                           P3D_SELLION,
                           P3D_NOSE,
                           P3D_SUB_NOSE,
                           P3D_RIGHT_EYE,
                           P3D_RIGHT_TEAR,
                           P3D_LEFT_TEAR,
                           P3D_LEFT_EYE,
                           P3D_STOMION])


# histogram of oriented gradients and svm face detection
face_hog = dlib.get_frontal_face_detector()
# facial feature predictions
predictor = dlib.shape_predictor('facial-landmarks/shape_predictor_68_face_landmarks.dat')


def detect_hog(img, face_detector):
    '''
    takes an image and returns a list of rectangle dimensions
    in the form of ((x,y), (width, height))
    '''
    # detect faces in grayscale
    # 1 is number of times to upsample image (helps to detect smaller faces)
    rects = face_detector(img, 1)
    return rects


def draw_roi(eye, image):
    '''
    takes the indexes of an eye and returns the
    region of interest from the image
    '''
    x, y, w, h = cv2.boundingRect(np.array([eye]))
    h = w
    y = y - h
    x = x - w // 2
    # draw a box around the eye in the frame
    cv2.rectangle(image, (x, y), (x + (w * 2), y + (h * 2)), (255, 0, 0), 2)
    roi = image[y:y + (h * 2), x:x + (w * 2)]
    roi = imutils.resize(roi, width=24, height=24, inter=cv2.INTER_CUBIC)
    return roi


def crop_eyes(frame, camera_matrix):
    '''
    outputs the left and right eye images from a frame
    '''
    # detect the face at grayscale image
    rect = detect_hog(frame, face_hog)

    # if the face detector doesnt detect face
    # return None, else if detects more than one faces
    # keep the bigger and if it is only one keep one dim
    if len(rect) == 0:
        return None
    elif len(rect) > 1:
        face = rect[0]
    elif len(rect) == 1:
        [face] = rect

    # determine the facial landmarks for the face region
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)



    # if the face detector is not in the frame then stop detecting
    # to avoid boundry errors
    (f_x, f_y, f_w, f_h) = face_utils.rect_to_bb(face)

    # for pnp algorithm
    landmarks_2D = returnLandmarks(frame, f_x, f_y, f_x + f_w, f_y + f_h, points_to_return=TRACKED_POINTS)
    for point in landmarks_2D:
                cv2.circle(frame, (point[0], point[1]), 2, (0, 0, 255), -1)

    # Applying the PnP solver to find the 3D pose
    # of the head from the 2D position of the
    # landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings
    # points from the model coordinate system to the camera coordinate system.
    # tvec - Output translation vector.
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix, camera_distortion)
    rvec_matrix = cv2.Rodrigues(rvec)[0]
    proj_matrix = np.hstack((rvec_matrix, tvec))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    cv2.putText(frame, "Tilt: {:.2f}".format(int(pitch)), (100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Now we project the 3D points into the image plane
    # Creating a 3-axis to be used as reference in the image.
    axis = np.float32([[50, 0, 0],
                       [0, 50, 0],
                       [0, 0, 50]])
    imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

    # Drawing the three axis on the image frame.
    # The opencv colors are defined as BGR colors such as:
    # (a, b, c) >> Blue = a, Green = b and Red = c
    # Our axis/color convention is X=R, Y=G, Z=B
    sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
    cv2.line(frame, sellion_xy, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
    cv2.line(frame, sellion_xy, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # BLUE
    cv2.line(frame, sellion_xy, tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # RED

    # define frame boundaries
    h, w = frame.shape[:2]
    # if face detection limits not within the boundaries, return nothing
    if (f_x or f_y or (f_x + f_w) or (f_x + f_h)) not in range(0, w):
        return None
    else:
        # grab indexes of facial landmarks for left and right eye, respectively
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']

        # extract the left and right eye indexes
        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        # get ROI (region of interest) for the eyes
        left_eye_image = draw_roi(left_eye, frame)
        right_eye_image = draw_roi(right_eye, frame)

        # if it doesnt detect left or right eye return None
        if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
            return None

        # return left and right eye
        return left_eye_image, right_eye_image


def returnLandmarks(inputImg, roiX, roiY, roiW, roiH, points_to_return=range(0, 68)):
    # Creating a dlib rectangle and finding the landmarks
    dlib_rectangle = dlib.rectangle(left=int(roiX), top=int(roiY), right=int(roiW), bottom=int(roiH))
    dlib_landmarks = predictor(inputImg, dlib_rectangle)

    # It selects only the landmarks that
    # have been indicated in the input parameter "points_to_return".
    # It can be used in solvePnP() to estimate the 3D pose.
    landmarks = np.zeros((len(points_to_return), 2), dtype=np.float32)
    counter = 0
    for point in points_to_return:
        landmarks[counter] = [dlib_landmarks.parts()[point].x, dlib_landmarks.parts()[point].y]
        counter += 1

    return landmarks


def cnn_predict(img, model):
    '''
    Uses pre trained eye open/close detection model
    and outputs state of eye
    '''
    # convert to greyscale and scale between 0 and 1
    img = np.dot(np.array(img, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    # keras needs (row, width, height, channel)
    # expand to add row dimension
    img = np.expand_dims(img, axis=0)
    classes = model.predict_classes(img, batch_size=10)
    return classes[0][0]


def load_model(model_path, weight_path):
    '''
    loads the pre trained eye model
    '''
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    # load model architecture
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    # compile the loaded model
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def main():
    counter = 0
    alarm_on = False
    max_frame = 40

    # open the camera,load the cnn model
    # imutils videostream will thread the frame reading separately
    # from the frame processing
    cap = cv2.VideoCapture('drowsy_car2.mov')
    model = load_model('model_test.json', 'weight_test.h5')
    writer = None
    while True:
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
        frame = imutils.resize(frame, width=600)

        # For head tilt calculations
        # Obtaining the CAM dimension
        h, w = frame.shape[:2]
        cam_w = w
        cam_h = h

        # Defining the camera matrix.
        # To have better result it is necessary to find the focal
        # length of the camera. fx/fy are the focal lengths (in pixels)
        # and cx/cy are the optical centres. These values can be obtained
        # roughly by approximation, for example in a 640x480 camera:
        # cx = 640/2 = 320
        # cy = 480/2 = 240
        # fx = fy = cx/tan(60/2 * pi / 180) = 554.26
        c_x = cam_w / 2
        c_y = cam_h / 2
        f_x = c_x / np.tan(60 / 2 * np.pi / 180)
        f_y = f_x

        # Estimated camera matrix values.
        camera_matrix = np.float32([[f_x, 0.0, c_x],
                                    [0.0, f_y, c_y],
                                    [0.0, 0.0, 1.0]])

        # detect eyes
        eyes = crop_eyes(frame, camera_matrix)
        # dont try to detect if no eyes are there
        if eyes is None:
            if writer is not None:
                writer.write(frame)
            continue
        else:
            left_eye, right_eye = eyes

            if (left_eye.shape[0:2] or right_eye.shape[0:2]) != (24, 24):
                if writer is not None:
                    writer.write(frame)
                continue
            else:
                # predict if the eye is open or closed
                state_left = cnn_predict(left_eye, model)
                state_right = cnn_predict(right_eye, model)

                if state_left <= 0.5 and state_right <= 0.5:
                    counter += 1
                    cv2.putText(frame, "Closed", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, "For: {:.2f}".format(counter), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # sound alarm if closed too long
                    if counter >= max_frame:
                        # if the alarm is not on, turn it on
                        if not alarm_on:
                            alarm_on = True
                            t = Thread(target=sound_alarm,
                                       args=('siren.wav',))
                            t.deamon = True
                            t.start()

                            # draw an alarm on the frame
                        cv2.putText(frame, "YOU'RE GOING TO CRASH!", (100, 300),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                else:
                    counter = 0
                    alarm_on = False
                    cv2.putText(frame, "Opened", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # write the frame to a file
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter('output_tilt.avi', fourcc, 60,
                                     (frame.shape[1], frame.shape[0]), True)

        # if the writer is not None, write the frame with recognized
        # faces t odisk
        if writer is not None:
            writer.write(frame)

    cap.release()
    writer.release()


if __name__ == '__main__':
    main()
