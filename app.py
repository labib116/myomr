
import cv2
import numpy as np
from imutils import contours

# --- 1. Define the Answer Key and Configuration ---
# The correct answers for all 60 questions.
# 0=A, 1=B, 2=C, 3=D
ANSWER_KEY = {
    0: 1, 1: 3, 2: 0, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0,
    10: 1, 11: 3, 12: 0, 13: 2, 14: 1, 15: 0, 16: 3, 17: 2, 18: 1, 19: 0,
    20: 1, 21: 2, 22: 0, 23: 3, 24: 1, 25: 0, 26: 2, 27: 3, 28: 1, 29: 0,
    30: 3, 31: 2, 32: 1, 33: 0, 34: 1, 35: 2, 36: 0, 37: 3, 38: 1, 39: 2,
    40: 0, 41: 1, 42: 3, 43: 2, 44: 0, 45: 1, 46: 3, 47: 0, 48: 2, 49: 1,
    50: 3, 51: 0, 52: 1, 53: 2, 54: 3, 55: 0, 56: 1, 57: 2, 58: 3, 59: 0
}

# --- 2. Image Processing Helper Functions ---

def order_points(pts):
    """
    Sorts a list of 4 points to be in top-left, top-right,
    bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """
    Applies a perspective transform to obtain a top-down view of the image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def find_document_corners(image):
    """
    Finds and correctly orders the four black square markers using a robust
    adaptive thresholding method, suitable for varied lighting conditions.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    thresh = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    corner_contours = []
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4 and cv2.contourArea(c) > 100:
                 corner_contours.append(approx)
                 if len(corner_contours) == 4:
                     break
    
    if len(corner_contours) != 4:
         print(f"Error: Found {len(corner_contours)} corner markers instead of 4. Check image quality.")
         cv2.imshow("DEBUG: Corner Detection Threshold", cv2.resize(thresh, (thresh.shape[1]//2, thresh.shape[0]//2)))
         cv2.waitKey(0)
         return None

    points = []
    for c in corner_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))
        else:
            points.append(tuple(c[0][0]))

    points = sorted(points, key=lambda p: p[1])
    top_points = sorted(points[:2], key=lambda p: p[0])
    bottom_points = sorted(points[2:], key=lambda p: p[0])

    ordered_points = np.array([
        top_points[0], top_points[1], bottom_points[1], bottom_points[0]
    ], dtype="float32")

    return ordered_points

# --- 3. Main Grading Logic ---

def process_omr_sheet(image_path):
    """
    Main function using a self-calibrating, dynamic threshold for robust grading.
    """
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    height, width, _ = original_image.shape
    target_height = 1000.0
    ratio = target_height / float(height)
    image = cv2.resize(original_image, (int(width * ratio), int(target_height)))

    corners = find_document_corners(image)
    if corners is None:
        return
    
    paper = four_point_transform(original_image, corners / ratio)
    warped = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(warped, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 2)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_bubbles = []
    for c in cnts:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0 and area > 100:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.8:
                all_bubbles.append(c)

    if len(all_bubbles) < 240:
        print(f"Warning: Found fewer bubbles than expected ({len(all_bubbles)}). Results may be incomplete.")

    page_width = paper.shape[1]
    col1_bubbles, col2_bubbles, col3_bubbles = [], [], []

    for bubble in all_bubbles:
        M = cv2.moments(bubble)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        if cx < page_width / 3:
            col1_bubbles.append(bubble)
        elif cx < (page_width / 3) * 2:
            col2_bubbles.append(bubble)
        else:
            col3_bubbles.append(bubble)

    sorted_col1 = contours.sort_contours(col1_bubbles, method="top-to-bottom")[0] if col1_bubbles else []
    sorted_col2 = contours.sort_contours(col2_bubbles, method="top-to-bottom")[0] if col2_bubbles else []
    sorted_col3 = contours.sort_contours(col3_bubbles, method="top-to-bottom")[0] if col3_bubbles else []

    question_map = { 0: sorted_col1, 1: sorted_col2, 2: sorted_col3 }
    total_correct = 0

    # --- UPGRADED GRADING LOGIC: DYNAMIC THRESHOLD CALIBRATION ---
    grad_thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 1. Collect pixel data from all bubbles
    pixel_counts = []
    for bubble in all_bubbles:
        mask = np.zeros(grad_thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [bubble], -1, 255, -1)
        mask = cv2.bitwise_and(grad_thresh, grad_thresh, mask=mask)
        pixel_counts.append(cv2.countNonZero(mask))

    # 2. Statistically determine the fill threshold
    if len(pixel_counts) > 0:
        mean_pixels = np.mean(pixel_counts)
        std_pixels = np.std(pixel_counts)
        # A bubble is "filled" if its pixel count is significantly higher than the average
        fill_threshold = mean_pixels + 1.5 * std_pixels
        print(f"Dynamic Fill Threshold Calculated: {fill_threshold:.2f}")
    else:
        fill_threshold = 99999 # A high value if no bubbles were found

    for col_idx in range(3):
        bubbles_in_col = question_map[col_idx]
        for i in range(0, len(bubbles_in_col), 4):
            question_num = (col_idx * 20) + (i // 4)
            current_bubbles = bubbles_in_col[i:i+4]
            if len(current_bubbles) != 4:
                continue

            sorted_choices = contours.sort_contours(current_bubbles, method="left-to-right")[0]
            
            marked_bubbles = []
            for j, choice_bubble in enumerate(sorted_choices):
                mask = np.zeros(grad_thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [choice_bubble], -1, 255, -1)
                mask = cv2.bitwise_and(grad_thresh, grad_thresh, mask=mask)
                total = cv2.countNonZero(mask)
                
                if total > fill_threshold:
                    marked_bubbles.append(j)
            
            color = (0, 0, 255)
            correct_answer_idx = ANSWER_KEY.get(question_num)
            student_answer_idx = -1

            if len(marked_bubbles) == 1:
                student_answer_idx = marked_bubbles[0]
                if correct_answer_idx == student_answer_idx:
                    total_correct += 1
                    color = (0, 255, 0)

            if correct_answer_idx is not None:
                cv2.drawContours(paper, [sorted_choices[correct_answer_idx]], -1, color, 3)

    score = (total_correct / 60.0) * 100
    print(f"--- OMR Grading Results ---")
    print(f"Total Questions: 60")
    print(f"Correct Answers: {total_correct}")
    print(f"Score: {score:.2f}%")
    print("----------------------------")

    result_image = cv2.resize(paper, (paper.shape[1] // 2, paper.shape[0] // 2))
    cv2.imshow("Graded Sheet", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == "__main__":
    image_file = 'demo2.jpg'
    process_omr_sheet(image_file)


