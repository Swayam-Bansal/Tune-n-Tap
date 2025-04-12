import cv2
from cv2.typing import MatLike

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the hands detection model in mediapipe
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.2,
)

# draw the hands tracing landmarks on the live feed
def draw_landmarks(image: MatLike, hand_landmarks: mp_hands.HandLandmark):
    # Draw the hand annotations on the image.
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
    )

class Detect_Piano():
    def __init__(self):
        self.piano_contour = None
        self.pianoKeys_points = []

        self.img_wrap = self.paper_warp = np.zeros((*(200, 200), 3), dtype=np.uint8)
        self.img_warp_bin = np.zeros((*(200, 200), 1), dtype=np.uint8)

        # to reduce latency and to have a smoother experience
        self.gray_frame = None
        self.blur_frame = None
        self.binary_frame = None

    def detect_piano(self, frame: MatLike, blur: int = 5, lower_canny_threshold: int = 50, upper_canny_threshold: int = 150):
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.blur_frame = cv2.GaussianBlur(self.gray_frame, (blur, blur), 0)
        self.binary_frame = cv2.Canny(self.blur_frame, lower_canny_threshold, upper_canny_threshold, L2gradient=True)

        kernel = np.ones((4,4),np.uint8)
        self.binary_frame = cv2.morphologyEx(self.binary_frame, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(self.binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # No contour was found
        if not contours:
            self.piano_contour = None
            print("Error: Could not find any contours in the frame.")
            return
        

        # Find the largest contour
        self.piano_contour = find_largest_rectangle(contours, 0.02)

        # No contour was found
        if self.piano_contour is None:
            print("Error: Could not find any contours in the frame, please try to focus camera to the piano.")
            return
        
        corner_points_piano = self.piano_contour.copy().reshape((4, 2))
        corner_points_piano = np.array(corner_points_piano, dtype=np.float32)

        # Sort the points in clockwise order
        corner_points_piano = sorted(corner_points_piano, key=lambda x: (x[0], x[1]))

        top_left = corner_points_piano[0]
        bottom_left = corner_points_piano[1]
        bottom_right = corner_points_piano[2]
        top_right = corner_points_piano[3]

        # Calculate the width and height of the rectangle
        width = int(max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left)
        ))

        height = int(max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left)
        ))

        # Define the destination points for the perspective transform
        dst_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        corner_points_piano = np.array(corner_points_piano, dtype=np.float32)

        # get the transformation matrix
        transformation_matrix = cv2.getPerspectiveTransform(corner_points_piano, dst_points)

        # Perform the perspective transform
        self.img_wrap = cv2.warpPerspective(frame, transformation_matrix, (width, height))

        # Convert the warped image to grayscale
        wrap_img = cv2.cvtColor(self.img_wrap, cv2.COLOR_BGR2GRAY)
        wrap_img = cv2.GaussianBlur(wrap_img, (blur, blur), 0)
        wrap_img = cv2.adaptiveThreshold(wrap_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((5,5),np.uint8)
        wrap_img = cv2.morphologyEx(wrap_img, cv2.MORPH_OPEN, kernel)

        #remove unwanted noise
        # wrap_img = add_inside_border

        kernel = np.ones((3,3),np.uint8)
        wrap_img = cv2.morphologyEx(wrap_img, cv2.MORPH_DILATE, kernel)

        self.img_warp_bin = wrap_img

        contour, _ = cv2.findContours(self.img_warp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # No contour was found
        if not contour:
            self.pianoKeyS_points = []
            print("Error: Could not find any contours in the frame.")
            return
        
        # Finds centers of points
        points = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filters small leftovers 
            if w > 6 and h > 6:
                points.append((x+w/2, y+h/2))
        
        # No points detected
        if not points:
            print("Error: Could not find any points in the frame.")
            return
        
        # Reshapes points for transform
        points = np.array(points, np.float32).reshape(-1, 1, 2)

        # Computes inverse matrix
        inverse_matrix = np.linalg.inv(transformation_matrix)

        corner_points_piano = cv2.perspectiveTransform(points, inverse_matrix)
        corner_points_piano = corner_points_piano.reshape(-1, 2)

        corner_points_piano = [tuple(row) for row in corner_points_piano.astype(object)]
        self.pianoKeys_points = sorted(corner_points_piano, key=lambda x: (x[0], x[1])) 


    def draw_contour(self, img: MatLike):
        """Draws the detected contour on the image"""
        if self.piano_contour is None:
            return

        # Draws green contour
        cv2.drawContours(img, [self.piano_contour], -1, (0, 255, 0), 2)
        # Draws red vetices
        for point in self.piano_contour:
            cv2.circle(img, tuple(point[0]), 5, (0, 0, 255), -1)


    def draw_points(self, img: MatLike, labeled: bool = True):
        """Draws original points, if exists"""
        if self.pianoKeys_points is None:
            return
        
        # Draws circles
        for point in self.pianoKeys_points:
            cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
            
        if labeled:
            # Define the text and properties
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 0, 0)
            thickness = 2
            
            i = 0
            # Draws text
            for point in self.pianoKeys_points:
                text = str(i)
                position = (int(point[0]-10), int(point[1]-10))
                cv2.putText(img, text, position, font, font_scale, color, thickness)
                i+=1
                       

def find_largest_rectangle(contours: MatLike, epsilon_mult: float) -> MatLike:
    """Finds largest contour by bounding box"""
    max_area = 0
    largest_rect = None
    for contour in contours:
        approx = approximate_contour(contour, epsilon_mult)
        area = cv2.contourArea(approx)

        if area > max_area and cv2.isContourConvex(approx) and len(approx) == 4:
            max_area = area
            largest_rect = approx

    return largest_rect


def approximate_contour(contour: MatLike, epsilon_mult: float) -> MatLike:
    """Approximates the contour to reduce the number of points"""
    epsilon = epsilon_mult * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    return approx
        

        