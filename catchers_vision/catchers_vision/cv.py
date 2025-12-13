import cv2
import numpy as np


class image_processor():
    """Image processing class for images in ROS."""

    def __init__(self):
        """Image Processing class."""
        self.lower_tennis = np.array([15, 65, 50])
        self.upper_tennis = np.array([35, 255, 200])

        self.lower_green = np.array([50, 50, 86])
        self.upper_green = np.array([86, 255, 169])

        #Not a great color threshold # noqa: E26
        self.lower_red = np.array([0, 30, 30])
        self.upper_red = np.array([3, 255, 210])

        self.lower_orange = np.array([0, 185, 100])
        self.upper_orange = np.array([15, 255, 255])

        self.depth_scale = 0.001

    def convert_color(self, frame):
        """
        Convert the color space from BGR to HSV.

        Parameters
        ----------
        frame: np.array
            Image frame to be converted

        Returns
        -------
        hsv: np.array
            Image in HSV color space

        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv

    def threshold_ball(self, image, frame, ball):
        """
        Create a color mask of specifed ball type.

        Parameters
        ----------
        image : np.array
            BGR color space version of the image

        frame : np.array
            HSV color space version of the image

        ball : string
            Describes the type of ball thresholded for

        Returns
        -------
        bitwise : np.array
            Bitwise conjunction of the image and it's mask

        mask : np.array
            Color thresholded binary representation of the image

        """
        if ball == 'tennis':
            mask = cv2.inRange(frame, self.lower_tennis, self.upper_tennis)
        elif ball == 'green':
            mask = cv2.inRange(frame, self.lower_green, self.upper_green)
        elif ball == 'red':
            mask = cv2.inRange(frame, self.lower_red, self.upper_red)
        else:
            mask = cv2.inRange(frame, self.lower_orange, self.upper_orange)

        return cv2.bitwise_and(
            image, image, mask=mask
        ), mask

    def color_threshold(self, color_img, depth_img, intr, ball):
        """
        Create a color mask of specifed ball type.

        Parameters
        ----------
        color_img : np.array
            BGR color space version of the image

        depth_img : np.array
            HSV color space version of the image

        intr : (fx, fy, cx, cy)
            Inherent values to determine camera distances

        ball : string
            Describes the type of ball thresholded for

        Returns
        -------
        centroid : np.array[3]
            Location of the x, y, and z of the ball

        image_processed : np.array
            Image with mask, or outlined ball

        """
        frame = self.capture_frame(color_img, depth_img)
        if frame is not None:
            gauss = cv2.GaussianBlur(frame, (11, 11), 0)
            frame_HSV = self.convert_color(gauss)
            frame_green, mask = self.threshold_ball(frame, frame_HSV, ball)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            (cx, cy, cz), cvt_image = self.find_ball(mask, frame_green, depth_img, intr)
            if len(cvt_image.shape) == 3:
                image_processed = cvt_image
            else:
                image_processed = mask
            if cz != -1:
                return np.array([float(cx), float(cy), float(cz)]), image_processed
            else:
                return np.array([-1.0, -1.0, -1.0]), image_processed
        else:
            return None, None

    def find_ball(self, mask, cvt_image, depth_img, intr):
        """
        Locate centroid of ball in 3d space.

        Parameters
        ----------
        mask : np.array
            color masked version of the image

        cvt_image : np.array
            HSV color space version of the image

        depth_img : np.array
            array of depth values

        intr : (fx, fy, cx, cy)
            Inherent values to determine camera distances

        Returns
        -------
        point_3d : np.array[3]
            Location of the x, y, and z of the ball

        cvt_image : np.array
            Return image with the contours drawn on it

        """
        # qqret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        empty_img = np.zeros((h, w, 3), dtype=np.uint8)
        if len(contours) < 1:
            return np.array([-1, -1, -1]), empty_img

        elif len(contours) >= 1:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < 15:
                return np.array([-1, -1, -1]), empty_img

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                return (-1, -1, -1), empty_img

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            point_3d = self.depth_extract(cx, cy, depth_img, intr)

            cv2.drawContours(cvt_image, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(cvt_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(
                cvt_image,
                f'({round(point_3d[0], 3)},{round(point_3d[1], 3)},{round(point_3d[2], 3)})',
                (cx + 10, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2
            )
            return point_3d, cvt_image

    def capture_frame(self, color_img, depth_img):
        """
        Capture frame function to clip depth.

        Parameters
        ----------
        color_img : np.array
            color version of the image

        depth_img : np.array
            array of depth values

        Returns
        -------
        bg_removed : np.array
            version of the image where depth is clipped

        """
        if color_img is None or depth_img is None:
            return None

        clipping_distance = 8 / self.depth_scale
        depth_image_3d = np.dstack((depth_img, depth_img, depth_img))
        grey_color = 153

        bg_removed = np.where(
            (depth_image_3d > clipping_distance) |
            (depth_image_3d <= 0),
            grey_color,
            color_img
        )

        return bg_removed

    def yolo_find_ball(self, results, class_names):
        """
        Filter detection boxes for Moving Balls.

        Parameters
        ----------
        results : Results()[]
            list of results from yolo model

        class_names : dict{int: string}
            HSV color space version of the image

        Returns
        -------
        cx : int
            x location of the ball

        cy : int
            y location of the ball

        """
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names[cls_id]

                if cls_name == 'Moving Ball':
                    xywh = box.xywh[0].cpu().numpy().astype(int)
                    cx, cy, w, h = xywh
                    conf = float(box.conf[0])
                    if conf > 0.70:
                        return cx, cy
            return None, None
        else:
            return None, None

    def depth_extract(self, cx, cy, depth_img, intr):
        """
        Extract depth from a point in an image with intrinsics.

        Parameters
        ----------
        cx : int
            x-coordinate of ball centroid

        cy : int
            y-coordinate of ball centroid

        depth_img : np.array
            array of depth values

        intr: (fx, fy, cx, cy)
            Inherent values to determine camera distances

        Returns
        -------
        centroid : np.aray
            3d center point of the array

        """
        if cx is not None:
            depth = depth_img[cy, cx] * 0.001
            fx, fy, cx0, cy0 = intr
            X = (cx - cx0) * depth / fx
            Y = (cy - cy0) * depth / fy
            Z = depth
            return np.array([X, Y, Z])
        else:
            return [-1.0, -1.0, -1.0]
