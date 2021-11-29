import os
import numpy as np
import cv2
from dlclive import DLCLive, Processor
import tensorflow as tf
import matplotlib.pyplot as plt

total_points = []

def inference_emotion(video_path, model_path):
    print(video_path)
    vcap = cv2.VideoCapture(video_path)
    dlc_model_path = model_path + '/DLC_prj_DOG_resnet_50_iteration-0_shuffle-1'
    lstm_model_path = model_path + '/best_model.h5'
    RESIZED_RATIO = 0.3
    FRAME_LENGTH = 100
    NUMBER_OF_POINTS = 30
    fps = vcap.get(cv2.CAP_PROP_FPS)
    width, height = int(vcap.get(3) * RESIZED_RATIO), int(vcap.get(4) * RESIZED_RATIO)
    #print(fps, width, height)

    while (vcap.isOpened()):
        ret, image = vcap.read()
        global total_points
        if ret:
            image = cv2.resize(image, dsize=(0,0), fx=RESIZED_RATIO, fy=RESIZED_RATIO, interpolation=cv2.INTER_LINEAR)
            #print(vcap.get(1))
            if (int(vcap.get(1)) % int(fps / 5)) == 0:
                video_points = find_keypoints(dlc_model_path, width, height, image)
                total_points.append(video_points)
        else:
            print('All frames are read!')
            break

    #print(total_points)
    res = predict_emotion(lstm_model_path, total_points, FRAME_LENGTH, NUMBER_OF_POINTS)
    return res


def find_keypoints(dlc_model_path, width, height, image):
    print('keypoint detection')
    video_points = []
    dlc_proc = Processor()
    dlc_live = DLCLive(dlc_model_path, processor=dlc_proc)
    dlc_live.init_inference(image)
    points = dlc_live.get_pose(image)
    for i in points:
        video_points.append(i[0] / width)
        video_points.append(i[1] / height)
    return video_points


def predict_emotion(lstm_model_path, total_points, FRAME_LENGTH, NUMBER_OF_POINTS):
    if len(total_points) >= FRAME_LENGTH:
        total_points = total_points[:FRAME_LENGTH]
    else:
        for i in range(FRAME_LENGTH - len(total_points)):
            total_points.append([0 for i in range(NUMBER_OF_POINTS)])
    total_points = np.array(total_points)
    total_points = np.reshape(total_points, ((1,) + total_points.shape))
    print(total_points[0])
    print(total_points.shape)

    trained_lstm_model = tf.keras.models.load_model(lstm_model_path)
    prediction = trained_lstm_model.predict(total_points)
    predicted_emotion = np.argmax(prediction)



    if predicted_emotion == 0:
        y = '공격성'
    elif predicted_emotion == 1:
        y = '공포'
    elif predicted_emotion == 2:
        y = '불안/슬픔'
    elif predicted_emotion == 3:
        y = '편안/안정'
    elif predicted_emotion == 4:
        y = '행복/즐거움'
    else:
        y = '화남/불쾌'

    return prediction, y

def test_keypoint_detection(image_path, model_path):
    img = cv2.imread(image_path)
    dlc_model_path = model_path + '/DLC_prj_DOG_resnet_50_iteration-0_shuffle-1'
    dlc_proc = Processor()
    dlc_live = DLCLive(dlc_model_path, processor=dlc_proc)
    dlc_live.init_inference(img)
    points = dlc_live.get_pose(img)
    for i in range(15):
        img = cv2.line(img, tuple(np.int64(points[i][:2])), tuple(np.int64(points[i][:2])), (255, 0, 0), 10)
    print(img.shape)
    cv2.imshow("res", img)
    cv2.waitKey(0)

if __name__ == '__main__':
    #video_path = '/mnt/efs/fs1/test.MOV'
    #model_path = '/mnt/efs/fs1/models'
    video_path = 'C:/Users/duri1994/anaconda3/envs/dlc-live/files/test.MOV'
    model_path = 'C:/Users/duri1994/anaconda3/envs/dlc-live/files/models'
    image_path = 'C:/Users/duri1994/anaconda3/envs/dlc-live/files/test.jpg'


    #print(tf.VERSION)
    prediction, res = inference_emotion(video_path, model_path)
    #test_keypoint_detection(image_path, model_path)

    print(prediction, res)