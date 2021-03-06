'''
Class to manipulate models used in gaze pointer controller.
'''

import os
import sys
import logging as log
import utils
import math
import time
from openvino.inference_engine import IENetwork, IECore

FACE_DETECTION_MODEL = 'intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
HEAD_POSE_MODEL = 'intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
FACE_LANDMARKS_MODEL = 'intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml'
GAZE_ESTIMATION_MODEL = 'intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml'

# FACE_DETECTION_MODEL = 'intel/face-detection-adas-0001/FP16/face-detection-adas-0001.xml'
# HEAD_POSE_MODEL = 'intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml'
# FACE_LANDMARKS_MODEL = 'intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml'
# GAZE_ESTIMATION_MODEL = 'intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml'

class BaseModel:

    def __init__(self, device):
        self.device = device
        self.net = None
        self.plugin = None
        self.input_blob = None
        self.out_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        self.outputs = None


    def load_model(self, model, cpu_extension=None, plugin=None):
        start_time = time.time()
   
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"

        if not plugin:
            log.info("Initializing plugin for model {} in {} device...".format(self.__class__.__name__,self.device))
            self.plugin = IECore()
        else:
            self.plugin = plugin

        if cpu_extension and 'CPU' in self.device:
            self.plugin.add_extension(cpu_extension, "CPU")

        self.net = IECore().read_network(model=model_xml, weights=model_bin) #IENetwork(model=model_xml, weights=model_bin)

        # If applicable, add a CPU extension to self.plugin
        if "CPU" in self.device:
            supported_layers = self.plugin.query_network(self.net, "CPU")
            not_supported_layers = [layer for layer in self.net.layers.keys() if layer not in supported_layers]

            if len(not_supported_layers) != 0:
                log.error("Layer not supported")
                sys.exit(1)

        self.net_plugin = self.plugin.load_network(network=self.net, device_name=self.device)

        self.input_blob = next(iter(self.net.inputs))
        self.out_blob = next(iter(self.net.outputs))
        finish_time = time.time()
        log.info("Model {} load time: {} seconds.".format(self.__class__.__name__, round(finish_time-start_time,4)))
        return self.plugin

    def predict(self, image, request_id=0):

        preprocessed_image = self.preprocess_input(image)
        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, 
                                                                inputs={self.input_blob: preprocessed_image})

        return self.net_plugin

    def check_model(self):
        pass

    def preprocess_input(self, image):

        input_shape = self.net.inputs[self.input_blob].shape
        preprocessed_image = utils.handle_image(image, input_shape[3], input_shape[2])

        return preprocessed_image

    def wait(self, request_id=0):
        status = self.net_plugin.requests[request_id].wait(-1)

        return status

    def get_output(self, request_id=0):

        self.outputs = self.net_plugin.requests[request_id].outputs[self.out_blob]

        return self.outputs


class FaceDetectorModel(BaseModel):

    def __init__(self, model_path=FACE_DETECTION_MODEL, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)
    
    def preprocess_output(self):

        return self.outputs[0,0]

    def get_face_crop(self, frame, args):

        threshold = args.prob_threshold

        try:
            self.predict(frame)
            self.wait()
            self.get_output()
            output = self.preprocess_output()
            
        except:
            return None

        if len(output)>0:
            detection = []
            for o in output:
                if o[2] > threshold:
                    xmin = o[3]
                    ymin = o[4]
                    xmax = o[5]
                    ymax = o[6]
                    detection.append([xmin, ymin, xmax, ymax])

        detection = detection[0] # just use first face


        w = frame.shape[1]
        h = frame.shape[0]
        detection = [int(detection[0]*w), int(detection[1]*h), int(detection[2]*w), int(detection[3]*h)]

        return frame[detection[1]:detection[3], detection[0]:detection[2]], detection

class HeadPoseModel(BaseModel):

    def __init__(self, model_path=HEAD_POSE_MODEL, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)


    def get_output(self, request_id=0):
        self.outputs = self.net_plugin.requests[request_id].outputs

        return self.outputs


    def preprocess_output(self):
        return [self.outputs['angle_y_fc'][0,0], self.outputs['angle_p_fc'][0,0], self.outputs['angle_r_fc'][0,0]]


    def get_headpose_angles(self, face_crop):

        try:
            self.predict(face_crop)
            self.wait()
            self.get_output()
            output = self.preprocess_output()
        except:
            return None

        return output


class FaceLandmarksModel(BaseModel):

    def __init__(self, model_path=FACE_LANDMARKS_MODEL, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)
    
    def preprocess_output(self):

        return self.outputs[0,:,0,0]

    def get_eyes_coordinates(self, face_crop):

        try:
            self.predict(face_crop)
            self.wait()
            self.get_output()
            output = self.preprocess_output()
        except:
            return None

        if len(output) > 0:
            right_eye = (output[0], output[1])
            left_eye = (output[2], output[3])

        return right_eye, left_eye


class GazeModel(BaseModel):
    def __init__(self, model_path=GAZE_ESTIMATION_MODEL, device='CPU'):
        super().__init__(device=device)
        self.load_model(model_path)

        self.input_name = [i for i in self.net_plugin.inputs.keys()]
        self.input_shape = self.net_plugin.inputs[self.input_name[1]].shape
    
    def predict(self, left_eye_crop, right_eye_crop, headpose_angles, request_id=0):
        preprocessed_left_eye_crop, preprocessed_right_eye_crop = self.preprocess_input(left_eye_crop, right_eye_crop)

        self.infer_request_handle = self.net_plugin.start_async(request_id=request_id, 
                                                                inputs={'head_pose_angles':headpose_angles, 
                                                                'left_eye_image':preprocessed_left_eye_crop, 
                                                                'right_eye_image':preprocessed_right_eye_crop})

        return self.net_plugin

    def preprocess_input(self, left_eye_crop, right_eye_crop):

        preprocessed_left_eye_crop = utils.handle_image(left_eye_crop, self.input_shape[3], self.input_shape[2])
        preprocessed_right_eye_crop = utils.handle_image(right_eye_crop, self.input_shape[3], self.input_shape[2])

        return preprocessed_left_eye_crop, preprocessed_right_eye_crop

    def get_output(self, request_id=0):
        
        self.outputs = self.net_plugin.requests[request_id].outputs
        return self.outputs

    def preprocess_output(self, headpose_angles):
        
        angle_r_fc = headpose_angles[2]
        roll_cosine = math.cos(angle_r_fc*math.pi/180.0)
        roll_sine = math.sin(angle_r_fc*math.pi/180.0)

        gaze_vector = self.outputs['gaze_vector'][0]

        x_movement = gaze_vector[0] * roll_cosine + gaze_vector[1] * roll_sine
        y_movement = -gaze_vector[0] *  roll_sine+ gaze_vector[1] * roll_cosine
                
        return (x_movement, y_movement), gaze_vector

    def get_gaze(self, right_eye_crop, left_eye_crop, headpose_angles):

        try:
            self.predict(left_eye_crop, right_eye_crop, headpose_angles)
            self.wait()
            self.get_output()
            
            (x_movement, y_movement), gaze_vector = self.preprocess_output(headpose_angles)
            
        except:
            return (0,0), 0

        return (x_movement, y_movement), gaze_vector