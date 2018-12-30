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


def crop_eyes(frame):
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
    max_frame = 20

    # open the camera,load the cnn model
    # imutils videostream will thread the frame reading separately
    # from the frame processing
    vs = VideoStream(src=0).start()
    model = load_model('model_test.json', 'weight_test.h5')
    time.sleep(2.0)

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # detect eyes
        eyes = crop_eyes(frame)
        # dont try to detect if no eyes are there
        if eyes is None:
            continue
        else:
            left_eye, right_eye = eyes

            if (left_eye.shape[0:2] or right_eye.shape[0:2]) != (24, 24):
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    counter = 0
                    alarm_on = False
                    cv2.putText(frame, "Opened", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # show the frame
        # imshow wil decrease fps dramatically
        # final product will not need this and react faster
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == '__main__':
    main()
