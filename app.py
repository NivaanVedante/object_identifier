import cv2
import argparse
from ultralytics import YOLO

# -------------------- ARGUMENT PARSER --------------------
parser = argparse.ArgumentParser(description="Live or File Video Object Detection with YOLO and optional tracking")
parser.add_argument('--source', type=str, default='webcam',
                    help="Input source: 'webcam' or path to video file like 'input.mp4'")
args = parser.parse_args()

# -------------------- LOAD YOLO --------------------
model = YOLO("yolo12n.pt")  # Replace with your own model if needed

# -------------------- TRACKING SETUP --------------------
tracking = False
tracker = None
bbox = None
frame = None

def select_object(event, x, y, flags, param):
    global tracking, tracker, bbox, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = cv2.selectROI("Select Object", frame, fromCenter=False)
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, bbox)
        tracking = True

# -------------------- VIDEO INPUT --------------------
input_source = 0 if args.source == 'webcam' else args.source
cap = cv2.VideoCapture(input_source)

cv2.namedWindow("Live Tracking")
cv2.setMouseCallback("Live Tracking", select_object)

# -------------------- VIDEO WRITER SETUP --------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# -------------------- MAIN LOOP --------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip only if webcam
    if args.source == 'webcam':
        frame = cv2.flip(frame, 1)

    results = model(frame)
    annotated_frame = results[0].plot()

    # Optional object tracking
    if tracking:
        success, bbox = tracker.update(annotated_frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(annotated_frame, "Tracking Failed!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    out.write(annotated_frame)
    cv2.imshow("Live Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
