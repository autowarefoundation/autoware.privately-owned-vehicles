import cv2

def main():

    video_filepath = "/mnt/Storage/Daihatsu/video_frames.avi"
    
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

        # Crop + rescale: 2880 × 1860 ---> 2880 x 1440 ---> 640 x 320

        # Run inference

        # Show raw binary mask

        # Convert raw binary mask to BEV

        # Process BEV to extract lane points

        # Process lane points to get curve parameters of the road (lane offset, yaw angle, curvature)

    cap.release()

if __name__ == "__main__":
    main()