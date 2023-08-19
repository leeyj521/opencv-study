import cv2
import numpy as np

def extract_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
    corners = np.int0(corners)
    return corners.reshape(-1, 2)

def estimate_rotation_matrix(corners):
    src_pts = np.float32(corners)
    dst_pts = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])
    M, _ = cv2.findHomography(src_pts, dst_pts)
    return M

def canny_img(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 5000, 1500, apertureSize=5, L2gradient=True)
    return canny


# Initialize video capture
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Extract corners from the current frame
    corners = extract_corners(frame)

    if len(corners) == 4:
        # Estimate the rotation matrix
        M = estimate_rotation_matrix(corners)

        # Print the estimated rotation matrix
        print("Estimated Rotation Matrix:")
        print(M)

        # Display the corners on the frame
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
