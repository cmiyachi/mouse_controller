
import cv2
import numpy as np
import utils
import logging as log
import time
import csv

from model import FaceDetectorModel, FaceLandmarksModel, HeadPoseModel, GazeModel

from mouse_controller import MouseController
from argparse import ArgumentParser

log.basicConfig(level=log.DEBUG)

def build_argparser():
    
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file. Use CAM to use webcam stream")
    
    # all arguments are optional below
    parser.add_argument("-mf", "--model_facedetector", required=False, type=str, default=None,
                        help="Path to face detector model.")
    parser.add_argument("-ml", "--model_facelm", required=False, type=str, default=None,
                        help="Path to face landmarks detector model.")
    parser.add_argument("-mh", "--model_headpose", required=False, type=str, default=None,
                        help="Path to head pose detector model.")
    parser.add_argument("-mg", "--model_gaze", required=False, type=str, default=None,
                        help="Path to gaze detector model.")

                                                          
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering")

    return parser


def get_models(args):
    model_facedetector = args.model_facedetector
    model_facelm = args.model_facelm
    model_headpose = args.model_headpose
    model_gaze = args.model_gaze

    if model_facedetector:
        facedetector = FaceDetectorModel(model_path=model_facedetector, device=args.device)
        # facedetector = FaceDetectionModel(model_path=model_facedetector, device=args.device)
    else:
        facedetector = FaceDetectorModel(device=args.device)
        # facedetector = FaceDetectionModel(device=args.device)
    
    if model_facelm:
        facelm = FaceLandmarksModel(model_path=model_facelm, device=args.device)
    else:
        facelm = FaceLandmarksModel(device=args.device)
    
    if model_headpose:
        headpose = HeadPoseModel(model_path=model_headpose, device=args.device)
    else:
        headpose = HeadPoseModel(device=args.device)

    if model_gaze:
        gaze = GazeModel(model_path=model_gaze, device=args.device)
    else:
        gaze = GazeModel()

    return facedetector, facelm, headpose, gaze


def gaze_pointer_controller(args, facedetector, facelm, headpose, gaze):

    mouse_controller = MouseController(precision='high', speed='fast')

    # Handle input type:
    inference_time_face = []
    inference_time_landmarks = []
    inference_time_headpose = []
    inference_time_gaze = []
    if args.input != 'CAM':
        try:
            input_stream = cv2.VideoCapture(args.input)
            length = int(input_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            webcamera = False

            if length > 1:
                single_image_mode = False
            else:
                single_image_mode = True

        except:
            print('Not supported file format.')
            exit()

    else:
        input_stream = cv2.VideoCapture(0)
        single_image_mode = False
        webcamera = True

    if not single_image_mode:
        count = 0
        while(input_stream.isOpened()):
        
            # Read the next frame:
            flag, frame = input_stream.read()

            if not flag:
                break

            frame_count = 20
            if count % frame_count == 0:

                start = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                face_crop, detection = facedetector.get_face_crop(frame, args)
                finish_face_detector_time = time.time()
                face_detector_time = round(finish_face_detector_time-start,5)
                inference_time_face.append(face_detector_time)

                right_eye, left_eye = facelm.get_eyes_coordinates(face_crop)

                right_eye_crop, left_eye_crop, right_eye_coords, left_eye_coords = utils.get_eyes_crops(face_crop, right_eye,left_eye)
                finish_eyes_coordinates = time.time()
                eyes_detector_time = round(finish_eyes_coordinates-finish_face_detector_time,5)
                inference_time_landmarks.append(eyes_detector_time)

                headpose_angles = headpose.get_headpose_angles(face_crop)
                finish_headpose_angles = time.time()
                headpose_detector_time = round(finish_headpose_angles-finish_eyes_coordinates,5)
                inference_time_headpose.append(headpose_detector_time)

                (x_movement, y_movement), gaze_vector = gaze.get_gaze(right_eye_crop, left_eye_crop, headpose_angles)
                finish_gaze_detection_time = time.time()
                gaze_detector_time = round(finish_gaze_detection_time-finish_headpose_angles,5)
                inference_time_gaze.append(gaze_detector_time)


                # because Ubuntu doesn't move the mouse much, draw graphics
                
                frame = cv2.rectangle(frame,(detection[0],detection[1]),(detection[2],detection[3]),color=(0,255,0), thickness=5)

                right_eye_coords = [right_eye_coords[0]+detection[1], right_eye_coords[1]+detection[1], right_eye_coords[2]+detection[0], right_eye_coords[3]+detection[0]]
                left_eye_coords = [left_eye_coords[0]+detection[1], left_eye_coords[1]+detection[1],left_eye_coords[2]+detection[0], left_eye_coords[3]+detection[0]]
                frame = cv2.rectangle(frame,(right_eye_coords[2],right_eye_coords[1]),(right_eye_coords[3],right_eye_coords[0]),color=(255,0,0),thickness=5)
                frame = cv2.rectangle(frame,(left_eye_coords[2],left_eye_coords[1]),(left_eye_coords[3],left_eye_coords[0]),color=(255,0,0),thickness=5)

            
                x_r_eye = int(right_eye[0]*face_crop.shape[1]+detection[0])
                y_r_eye = int(right_eye[1]*face_crop.shape[0]+detection[1])
                x_r_shift, y_r_shift = int(x_r_eye+gaze_vector[0]*100), int(y_r_eye-gaze_vector[1]*100)

                x_l_eye = int(left_eye[0]*face_crop.shape[1]+detection[0])
                y_l_eye = int(left_eye[1]*face_crop.shape[0]+detection[1])
                x_l_shift, y_l_shift = int(x_l_eye+gaze_vector[0]*100), int(y_l_eye-gaze_vector[1]*100)

                frame = cv2.arrowedLine(frame, (x_r_eye, y_r_eye), (x_r_shift, y_r_shift), (0, 0, 255), 2)
                frame = cv2.arrowedLine(frame, (x_l_eye, y_l_eye), (x_l_shift, y_l_shift), (0, 0, 255), 2)


                frame = cv2.putText(frame, 'Yaw: '+str(headpose_angles[0])+' '+'Pitch: '+str(headpose_angles[1])+' '+'Roll: '+str(headpose_angles[2]),(15,20),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,0),2)

      
                cv2.namedWindow('Miyachi',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Miyachi', 600,400)
                cv2.imshow('Miyachi', frame)

                mouse_controller.move(x_movement,y_movement)
            count = count + 1

        input_stream.release()
        with open('benchmarks.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Face Detector','Eyes Detector', 'Headpose Detector', 'Gaze Detector'])
            for i in range(len(inference_time_face)):
                writer.writerow([inference_time_face[i], inference_time_landmarks[i], inference_time_headpose[i], inference_time_gaze[i]])

    cv2.destroyAllWindows()


def main():
   
    # Read command line args
    args = build_argparser().parse_args()

    # Load the models
    facedetector, facelm, headpose, gaze = get_models(args)

    # Do the work
    gaze_pointer_controller(args, facedetector, facelm, headpose, gaze)


if __name__ == '__main__':
    main()
