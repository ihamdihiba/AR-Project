import cv2
import easyocr

print("Loading EasyOCR model (English)...")
reader = easyocr.Reader(['en'], gpu=False)
print("EasyOCR loaded successfully!")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("\nWebcam is running!")
print("Hold text or a pill box in front of the camera.")
print("Press 'S' to scan a snapshot, or 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w, _ = frame.shape
    
    # Draw alignment reticle in the center
    box_x1, box_y1 = w // 4, h // 4
    box_x2, box_y2 = 3 * w // 4, 3 * h // 4
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
    cv2.putText(frame, "Align Box Here & Press 'S' to Scan", (box_x1, box_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("AR Medication Scanner - Camera Test", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("\n--- Scanning Region of Interest ---")
        # Crop to the reticle area
        roi = frame[box_y1:box_y2, box_x1:box_x2]
        
        # Run EasyOCR on the cropped image
        results = reader.readtext(roi)
        
        if not results:
            print("No text detected. Try adjusting lighting or bringing the box closer.")
        else:
            print("Detected text:")
            for bbox, text, prob in results:
                print(f"  -> '{text}' (Confidence: {prob:.2f})")
        print("-----------------------------------\n")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()