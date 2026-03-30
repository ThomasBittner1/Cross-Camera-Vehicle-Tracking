import cv2
import numpy as np

# List to store coordinates
points = []


def draw_polygon(event, x, y, flags, param):
    global points

    # Left click to add a point
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})")

    # Right click to reset points if you mess up
    elif event == cv2.EVENT_RBUTTONDOWN:
        points = []
        print("Resetting points.")


def main():
    global points
    video_path = r"AICity22_Track1_MTMC_Tracking\test\S06\c041\vdo.avi"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video. Check the path.")
        return

    ret, img = cap.read()
    cap.release()

    if not ret or img is None:
        print("Error: Could not read the first frame from the video.")
        return

    cv2.namedWindow("Polygon Mask Drawer")
    cv2.setMouseCallback("Polygon Mask Drawer", draw_polygon)

    print("\n--- INSTRUCTIONS ---")
    print("1. Left-Click to place points.")
    print("2. Right-Click to clear points.")
    print("3. Press 'q' to quit and print the final array.")
    print("---------------------\n")

    while True:
        display_img = img.copy()

        # Draw lines between points
        if len(points) > 0:
            # Draw dots at each click
            for pt in points:
                cv2.circle(display_img, pt, 4, (0, 255, 0), -1)

            # Draw lines connecting the dots
            if len(points) > 1:
                cv2.polylines(display_img, [np.array(points)], isClosed=False, color=(255, 0, 0), thickness=2)

        cv2.imshow("Polygon Mask Drawer", display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # Final Output
    print("\nFinal Polygon Coordinates:")
    print(points)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
