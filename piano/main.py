import cv2
import detection as detection
import control as control
import numpy as np

def main():
    window = cv2.VideoCapture(1)

    # make an instance of the Detect_Piano class
    piano_det = detection.Detect_Piano()

    ctrls = control.ControlPanel()

    # Uncomment the following lines to set the camera resolution
    # window.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # window.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Check if the camera opened successfully
    if not window.isOpened():
        print("Error: Could not open camera.")
        return
    
    was_frozen = False

    while True:
        # Read a frame from the camera
        ret, frame = window.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # flips the frame to create a mirror effect
        frame = cv2.flip(frame, 1)

        if not ctrls.freeze_points.get():
            # Detect piano points
            piano_det.detect_piano(frame, ctrls.blur.get()*2+1, ctrls.canny_ths1.get(), ctrls.canny_ths2.get())
            wasFrozen = False
        else:
            if wasFrozen == False:
                wasFrozen = True

        # Draws points and landmarks on original frame
        piano_det.draw_contour(frame)
        piano_det.draw_points(frame)

        # Converts biary from greyscale to rgb to stack them with color image
        main_bin_rgb = cv2.cvtColor(piano_det.binary_frame ,cv2.COLOR_GRAY2BGR)
        warped_bin_rgb = cv2.cvtColor(piano_det.img_warp_bin ,cv2.COLOR_GRAY2BGR)

        print("Shape of piano_det.paper_warp:", piano_det.paper_warp.shape)
        print("Shape of warped_bin_rgb:", warped_bin_rgb.shape)

        # Stacks images
        display_main = np.hstack((frame, main_bin_rgb))
        display_warped = np.hstack((piano_det.paper_warp, warped_bin_rgb))
        
        # Displays images
        cv2.imshow("Main", display_main)
        cv2.imshow("Warped", display_warped)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    window.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()