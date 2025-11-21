import cv2

def main():

    video_filepath = "/mnt/Storage/Daihatsu/video_frames.avi"

    # Read homography (should be computed once with findHomography)
    
    cap = cv2.VideoCapture(video_filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / 10)

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        
        # Read data
        ret, frame = cap.read()
        if not ret:
            break

        # Crop: 2880 × 1860 ---> 2880 x 1440

        # Rescale---> 640 x 320

        # Run inference

        # Show raw binary mask (must be normalized so we can use homography)

        # Convert raw binary mask to BEV using homography

        # Process BEV to extract lane points

        # Show BEV masks (debugging purpose)

        # Process lane points to get curve parameters of the road (lane offset, yaw angle, curvature)

        # Show BEV vis with the curve parameters and sliding windows and basically everything that helps us debug

    cap.release()

if __name__ == "__main__":
    main()