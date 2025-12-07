import os

import cv2
import pyzed.sl as sl


class ZedImage():

    def __init__(self, onnx_path, threshold=70):

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.sdk_verbose = 1
        self.init_params.camera_fps = 100
        self.init_params.camera_resolution = sl.RESOLUTION.VGA
        status = self.zed.open(self.init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f'ZED failed to open: {repr(status)}')

        self.runtime_params = sl.RuntimeParameters()
        self.output_img_zed = sl.Mat()
        self.thresh = threshold

        if os.path.exists(onnx_path):
            print(f'[INFO] Found model at: {onnx_path}')
            self.obj_param = sl.ObjectDetectionParameters()
            self.obj_param.enable_tracking = True
            self.obj_param.enable_segmentation = True
            self.obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_YOLOLIKE_BOX_OBJECTS
            self.obj_param.custom_onnx_file = onnx_path

            res = sl.Resolution(width=640, height=640)
            self.obj_param.custom_onnx_dynamic_input_shape = res
            
            if self.obj_param.enable_tracking :
                positional_tracking_param = sl.PositionalTrackingParameters()
                #positional_tracking_param.set_as_static = True
                self.zed.enable_positional_tracking(positional_tracking_param)

            status = self.zed.enable_object_detection(self.obj_param)
            if status == sl.ERROR_CODE.SUCCESS:
                self.objects = sl.Objects()
                self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
                self.obj_runtime_param.detection_confidence_threshold = threshold
                self.zed.set_object_detection_runtime_parameters(self.obj_runtime_param)
                self.obj_detection_enabled = True
                print('[INFO] Custom YOLO Detection enabled.')
            else:
                print(f'[WARN] Failed to enable object detection: {repr(status)}')
                print('[WARN] Continuing without object detection.')

    def get_frame(self):
        """Get the color frame, detected frame and 3D pos."""
        if self.zed.grab(self.runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None, None, None

        # Retrieve left RGBA from ZED
        self.zed.retrieve_image(self.output_img_zed, sl.VIEW.LEFT)
        raw_image = self.output_img_zed.get_data()
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2BGR)
        det_image = raw_image.copy()
        positions_3d = [-1.0, -1.0, -1.0]
        if self.objects:
            self.zed.retrieve_objects(self.objects, self.obj_runtime_param)
            best_obj = max(self.objects.object_list, key=lambda o: o.confidence, default=None)
            if best_obj is not None and best_obj.confidence > self.thresh:
                bb = best_obj.bounding_box_2d
                pt1 = (int(bb[0][0]), int(bb[0][1]))
                pt2 = (int(bb[2][0]), int(bb[2][1]))

                cv2.rectangle(det_image, pt1, pt2, (0, 255, 0), 2)
                cls_txt = f'Moving Ball: ({best_obj.confidence:.2f})'
                cv2.putText(det_image, cls_txt, (pt1[0], pt1[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                positions_3d = [best_obj.position[0], best_obj.position[1], best_obj.position[2]]

        return raw_image, det_image, positions_3d

    def close(self):
        self.zed.disable_object_detection()
        self.zed.close()
