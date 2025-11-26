import cv2
import argparse
from pathlib import Path


def draw_bbox(image_path):
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]

    label_path = Path(image_path.replace("images", "labels").replace(".jpg", ".txt"))
    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip malformed lines
            cls, x, y, w, h = map(float, parts)

            # Convert to pixel coordinates
            x_center_px = int(x * img_w)
            y_center_px = int(y * img_h)
            w_px = int(w * img_w)
            h_px = int(h * img_h)

            # Get top-left and bottom-right corners
            x1 = int(x_center_px - w_px / 2)
            y1 = int(y_center_px - h_px / 2)
            x2 = int(x_center_px + w_px / 2)
            y2 = int(y_center_px + h_px / 2)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    # Show image
    cv2.imshow("Image with bbox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="image path")
    args = parser.parse_args()

    image = args.image_path
    draw_bbox(image)
