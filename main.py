import cv2

def main():
    window = cv2.VideoCapture(1)

    window.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    window.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

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

        # Display the frame in a window
        cv2.imshow("Camera Feed", frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    window.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()