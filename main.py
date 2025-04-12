import cv2
import detection

def main():
    window = cv2.VideoCapture(1)

    # Uncomment the following lines to set the camera resolution
    # window.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # window.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    # Check if the camera opened successfully
    if not window.isOpened():
        print("Error: Could not open camera.")
        return
    
    while True:
        # Read a frame from the camera
        ret, frame = window.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # flips the frame to create a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame to detect hands
        results = detection.hands.process(rgb_frame)

        # If hands are detected, draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                detection.draw_landmarks(frame, hand_landmarks)

        # Display the frame in a window
        cv2.imshow("Camera Feed", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    window.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()