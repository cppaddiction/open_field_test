import cv2

cap = cv2.VideoCapture('vid.avi')
backSub = cv2.createBackgroundSubtractorMOG2()
fourcc = cv2.VideoWriter_fourcc(*'aviv')

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(fps, width, height)

out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while cap.isOpened():
    # Capture video frame-by-frame
    ret, frame = cap.read()
    frame_out = frame.copy()
    if ret:
        # Apply background subtraction
        fg_mask = backSub.apply(frame)

        # Find contours
        contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # Apply global threshold to remove shadows
        retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

        # Set the kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # Apply erosion
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

        min_contour_area = 500  # Define your minimum area threshold
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]  # Filter contours

        for cnt in large_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            frame_out = cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 0, 200), 3)

        out.write(frame_out)
