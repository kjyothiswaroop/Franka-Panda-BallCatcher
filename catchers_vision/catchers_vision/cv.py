import cv2
import numpy as np


class image_processor():

    def __init__(self):
        """Image Processing class."""
        self.lower_tennis = np.array([15, 65, 50])
        self.upper_tennis = np.array([35, 255, 200])

        self.lower_green = np.array([50, 150, 0])
        self.upper_green = np.array([80, 255, 170])

        #Not a great color threshold # noqa: E26
        self.lower_red = np.array([0, 30, 30])
        self.upper_red = np.array([3, 255, 210])

        self.lower_orange = np.array([0, 185, 100])
        self.upper_orange = np.array([15, 255, 255])

        self.depth_scale = 0.001

    def convert_color(self, frame):
        """Convert color space from bgr to hsv."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return hsv

    def threshold_ball(self, image, frame, ball):
        """Isolate ball."""
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
        """Check frame for ball and publishes if present."""
        while True:
            frame = self.capture_frame(color_img, depth_img)
            if frame is not None:
                frame_HSV = self.convert_color(frame)
                frame_green, mask = self.threshold_ball(frame, frame_HSV, ball)
                (cx, cy, cz), pnt = self.find_ball(mask, depth_img, intr)
                if cz != -1:
                    return np.array([float(cx), float(cy), float(cz)]), mask
                else:
                    return np.array([-1.0, -1.0, -1.0]), mask
            else:
                return None

    def find_ball(self, mask, depth_img, intr):
        """Locate centroid of ball in 3d space."""
        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)

        if len(contours) < 1:
            return np.array([-1, -1, -1]), np.array([0, 0])

        elif len(contours) >= 1:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            if area < 20:
                return np.array([-1, -1, -1]), np.array([0, 0])
            perimeter = cv2.arcLength(cnt, True)

            if perimeter != 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.6:
                    return (-1, -1, -1), np.array([0, 0])

            M = cv2.moments(cnt)
            if M['m00'] == 0:
                return (-1, -1, -1), np.array([0, 0])

            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            depth = depth_img[cy, cx] * 0.001
            fx, fy, cx0, cy0 = intr
            X = (cx - cx0) * depth / fx
            Y = (cy - cy0) * depth / fy
            Z = depth
            point_3d = np.array([X, Y, Z])

            return point_3d, np.array([cx, cy])

    def capture_frame(self, color_img, depth_img):
        """Capture frame function to clip depth."""
        if color_img is None or depth_img is None:
            return None

        clipping_distance = 4 / self.depth_scale
        depth_image_3d = np.dstack((depth_img, depth_img, depth_img))
        grey_color = 153

        bg_removed = np.where(
            (depth_image_3d > clipping_distance) |
            (depth_image_3d <= 0),
            grey_color,
            color_img
        )

        return bg_removed
