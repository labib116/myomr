import cv2
import numpy as np
from imutils import contours

# --- 1. Define the Answer Key ---
ANSWER_KEY = {
    0: 1, 1: 3, 2: 0, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0,
    10: 1, 11: 3, 12: 0, 13: 2, 14: 1, 15: 0, 16: 3, 17: 2, 18: 1, 19: 0,
    20: 1, 21: 2, 22: 0, 23: 3, 24: 1, 25: 0, 26: 2, 27: 3, 28: 1, 29: 0,
    30: 3, 31: 2, 32: 1, 33: 0, 34: 1, 35: 2, 36: 0, 37: 3, 38: 1, 39: 2,
    40: 0, 41: 1, 42: 3, 43: 2, 44: 0, 45: 1, 46: 3, 47: 0, 48: 2, 49: 1,
    50: 3, 51: 0, 52: 1, 53: 2, 54: 3, 55: 0, 56: 1, 57: 2, 58: 3, 59: 0
}

# --- 2. UPGRADED Alignment Functions using Fiducial Markers ---

def reorder(myPoints):
    """Sorts corner points into a consistent order: tl, tr, bl, br."""
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 2), np.float32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] # Top-Left
    myPointsNew[3] = myPoints[np.argmax(add)] # Bottom-Right
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # Top-Right
    myPointsNew[2] = myPoints[np.argmax(diff)] # Bottom-Left
    return myPointsNew

def find_fiducial_markers(image):
    """
    Finds the four black square fiducial markers for robust alignment.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    marker_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        
        if 0.8 <= aspect_ratio <= 1.2 and 100 < area < 10000:
             marker_contours.append(c)
    
    if len(marker_contours) < 4:
        print(f"Error: Found only {len(marker_contours)} fiducial markers. Cannot align.")
        return None

    marker_contours = sorted(marker_contours, key=cv2.contourArea, reverse=True)[:4]
    
    points = []
    for c in marker_contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append([cX, cY])
        else:
            (x,y,w,h) = cv2.boundingRect(c)
            points.append([x + w//2, y + h//2])

    if len(points) != 4:
        print(f"Error: Could not determine center for all 4 markers. Found {len(points)}.")
        return None

    points = np.array(points, dtype="int32").reshape(4,2)
    
    return reorder(points)


def four_point_transform(image, pts):
    """Applies a perspective transform to get a top-down view."""
    (tl, tr, bl, br) = pts
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1]], dtype="float32")
        
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))


# --- 3. Bubble Detection and Grading Logic ---

def get_intensity_threshold(intensities, min_jump=15):
    """
    Analyzes all bubble intensities to find the best threshold to separate
    marked (dark) from unmarked (light) bubbles.
    """
    if not intensities: return 150 
    intensities.sort()
    jumps = [(intensities[i+1] - intensities[i], i) for i in range(len(intensities)-1)]
    if not jumps: return np.mean(intensities) if intensities else 150
    best_jump, best_index = max(jumps, key=lambda item: item[0])
    if best_jump > min_jump: return intensities[best_index] + best_jump / 2
    else: return np.mean(intensities) - 5


def group_bubbles_into_rows(bubbles, tolerance=10):
    """
    Groups bubbles within a single column into rows.
    """
    if not bubbles: return []
    bubbles = contours.sort_contours(bubbles, method="top-to-bottom")[0]
    rows = []
    current_row = [bubbles[0]]
    for i in range(1, len(bubbles)):
        (x, y, w, h) = cv2.boundingRect(bubbles[i])
        (prev_x, prev_y, prev_w, prev_h) = cv2.boundingRect(current_row[0])
        if abs(y - prev_y) < tolerance: current_row.append(bubbles[i])
        else:
            rows.append(current_row)
            current_row = [bubbles[i]]
    rows.append(current_row)
    return rows


def process_omr_sheet(image_path):
    """Main processing function with fiducial marker detection and robust positional grading."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    corners = find_fiducial_markers(original_image)
    
    if corners is None:
        print("Could not find 4 fiducial markers. Aborting.")
        return
        
    paper = four_point_transform(original_image, corners)
    
    h, w = paper.shape[:2]
    crop_margin_y = int(h * 0.1) 
    crop_margin_x = int(w * 0.05) 
    
    cropped_paper = paper[crop_margin_y:h-crop_margin_y, crop_margin_x:w-crop_margin_x]
    
    warped_color = cropped_paper.copy()
    warped_gray = cv2.cvtColor(cropped_paper, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(warped_gray, 230, 255, cv2.THRESH_BINARY_INV)
    
    cv2.imshow("Debugging: Cropped Threshold Image", thresh)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    all_bubbles = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # --- FINAL FIX: Relaxing the aspect ratio filter ---
        # Widened the range to be more forgiving of hand-drawn marks.
        if w >= 10 and h >= 10 and ar >= 0.7 and ar <= 1.4:
            all_bubbles.append(c)

    print(f"INFO: Found {len(all_bubbles)} bubble contours.")
    
    total_correct = 0
    
    if len(all_bubbles) > 0:
        all_bubble_intensities = []
        for c in all_bubbles:
            mask = np.zeros(warped_gray.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mean_val = cv2.mean(warped_gray, mask=mask)[0]
            all_bubble_intensities.append(mean_val)
            
        grading_threshold = get_intensity_threshold(all_bubble_intensities)
        print(f"Dynamic Intensity Threshold calculated: {grading_threshold:.2f}")

        page_width = warped_gray.shape[1]
        col1, col2, col3 = [], [], []

        for c in all_bubbles:
            (x, y, w, h) = cv2.boundingRect(c)
            if x < page_width / 3: col1.append(c)
            elif x < page_width / 3 * 2: col2.append(c)
            else: col3.append(c)
        
        for col_idx, column in enumerate([col1, col2, col3]):
            question_rows = group_bubbles_into_rows(column)
            
            if not question_rows: continue

            first_row_y = cv2.boundingRect(question_rows[0][0])[1]
            last_row_y = cv2.boundingRect(question_rows[-1][0])[1]
            approx_row_height = (last_row_y - first_row_y) / 19.0 if len(question_rows) > 1 else 30 

            for row in question_rows:
                row_y = cv2.boundingRect(row[0])[1]
                
                est_row_index = int(((row_y - first_row_y) / approx_row_height) + 0.5)
                question_num = (col_idx * 20) + est_row_index

                if len(row) != 4:
                    print(f"Skipping malformed row for Question ~{question_num+1} (found {len(row)} bubbles).")
                    for bubble in row:
                        (x, y, w, h) = cv2.boundingRect(bubble)
                        cv2.rectangle(warped_color, (x, y), (x + w, y + h), (255, 255, 0), 2)
                    continue
                
                question_bubbles = contours.sort_contours(row, method="left-to-right")[0]
                
                marked_choice = -1
                marked_count = 0
                
                for choice_idx, bubble_contour in enumerate(question_bubbles):
                    mask = np.zeros(warped_gray.shape, dtype="uint8")
                    cv2.drawContours(mask, [bubble_contour], -1, 255, -1)
                    bubble_intensity = cv2.mean(warped_gray, mask=mask)[0]

                    if bubble_intensity < grading_threshold:
                        marked_choice = choice_idx
                        marked_count += 1

                color = (0, 0, 255) 
                correct_answer_idx = ANSWER_KEY.get(question_num)

                if marked_count == 1: 
                    if marked_choice == correct_answer_idx:
                        total_correct += 1
                        color = (0, 255, 0)

                if correct_answer_idx is not None and correct_answer_idx < len(question_bubbles):
                    cv2.drawContours(warped_color, [question_bubbles[correct_answer_idx]], -1, color, 3)

    score = (total_correct / 60.0) * 100 if len(ANSWER_KEY) > 0 else 0
    print("\n--- OMR Grading Results ---")
    print(f"Correct Answers: {total_correct} / 60")
    print(f"Score: {score:.2f}%")
    print("----------------------------")

    cv2.putText(warped_color, f"Score: {score:.2f}%", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    h, w = warped_color.shape[:2]
    max_height = 800
    if h > max_height:
        ratio = max_height / h
        result_image = cv2.resize(warped_color, (int(w * ratio), max_height))
    else:
        result_image = warped_color

    cv2.imshow("Graded Sheet", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = 'demo_marked.jpg' 
    process_omr_sheet(image_file)


