import cv2
import numpy as np
from imutils import contours
import os
import csv
import logging

# --- New Function: Setup Logging ---
def setup_logging():
    """Configures the logging system to output to both a file and the console."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("omr_grader.log"), # Log file
            logging.StreamHandler()                # Console output
        ]
    )

# --- 1. Define the Answer Key and Choice Mapping ---
ANSWER_KEY = {
    0: 1, 1: 3, 2: 0, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0,
    10: 1, 11: 3, 12: 0, 13: 2, 14: 1, 15: 0, 16: 3, 17: 2, 18: 1, 19: 0,
    20: 1, 21: 2, 22: [0, 2], 23: 3, 24: 1, 25: 0, 26: 2, 27: 3, 28: 1, 29: 0,
    30: 3, 31: 2, 32: 1, 33: 0, 34: 1, 35: 2, 36: 0, 37: 3, 38: 1, 39: 2,
    40: 0, 41: 1, 42: 3, 43: 2, 44: 0, 45: 1, 46: 3, 47: 0, 48: 2, 49: 1,
    50: 3, 51: 0, 52: 1, 53: 2, 54: 3, 55: 0, 56: 1, 57: 2, 58: 3, 59: 0
}
CHOICE_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


# --- 2. Alignment Functions (Unchanged) ---
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 2), np.float32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def find_fiducial_markers(image):
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
        logging.error(f"Found only {len(marker_contours)} fiducial markers. Cannot align.")
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
            (x, y, w, h) = cv2.boundingRect(c)
            points.append([x + w // 2, y + h // 2])
    if len(points) != 4:
        logging.error(f"Could not determine center for all 4 markers. Found {len(points)}.")
        return None
    return reorder(np.array(points, dtype="float32"))

def four_point_transform(image, pts):
    (tl, tr, bl, br) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]
    ], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

# --- 3. Bubble Detection Helpers (Unchanged) ---
def get_intensity_threshold(intensities, min_jump=15):
    if not intensities: return 150
    intensities.sort()
    jumps = [(intensities[i + 1] - intensities[i], i) for i in range(len(intensities) - 1)]
    if not jumps: return np.mean(intensities) if intensities else 150
    best_jump, best_index = max(jumps, key=lambda item: item[0])
    if best_jump > min_jump: return intensities[best_index] + best_jump / 2
    else: return np.mean(intensities) - 5

def group_bubbles_into_rows(bubbles, tolerance=10):
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

# --- 4. Separator Line Finder (Unchanged) ---
def find_horizontal_separator(image):
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > w_img * 0.8 and h < 20:
            return y + (h // 2)
    return -1

# --- 5. Roll Number Decoding (Unchanged) ---
def decode_roll_number(roll_section_image, output_image_to_draw, y_offset):
    # This function remains unchanged as per previous logic.
    if roll_section_image is None or roll_section_image.size == 0: return "N/A"
    h, w = roll_section_image.shape[:2]
    crop_y_start = int(h * 0.315); bubble_area = roll_section_image[crop_y_start:, :]
    gray = cv2.cvtColor(bubble_area, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c); ar = w / float(h)
        if 10 < w < 50 and 10 < h < 50 and 0.7 <= ar <= 1.3: bubble_contours.append(c)
    logging.info(f"Found {len(bubble_contours)} potential bubble contours in roll number section.")
    if len(bubble_contours) < 50: return "Not Enough Bubbles Found"
    bubble_contours, _ = contours.sort_contours(bubble_contours, method="left-to-right")
    columns = []; current_column = [bubble_contours[0]]
    avg_bubble_width = np.mean([cv2.boundingRect(c)[2] for c in bubble_contours])
    for i in range(1, len(bubble_contours)):
        prev_x, _, prev_w, _ = cv2.boundingRect(bubble_contours[i-1]); curr_x, _, _, _ = cv2.boundingRect(bubble_contours[i])
        if (curr_x - (prev_x + prev_w)) > (avg_bubble_width * 0.4): columns.append(current_column); current_column = [bubble_contours[i]]
        else: current_column.append(bubble_contours[i])
    columns.append(current_column)
    logging.info(f"Clustered roll number bubbles into {len(columns)} columns.")
    if len(columns) == 6: bubble_columns = columns[1:]
    elif len(columns) == 5: bubble_columns = columns
    else: logging.error(f"Expected 5 or 6 columns for roll number, but found {len(columns)}."); return "Column Count Error"
    decoded_roll = []
    all_bubble_intensities = [cv2.mean(gray, mask=cv2.drawContours(np.zeros(gray.shape, dtype="uint8"), [c], -1, 255, -1))[0] for c in bubble_contours]
    grading_threshold = get_intensity_threshold(all_bubble_intensities)
    logging.info(f"Roll number grading threshold: {grading_threshold:.2f}")
    for column in bubble_columns:
        column_sorted, _ = contours.sort_contours(column, method="top-to-bottom")
        if not (9 <= len(column_sorted) <= 11): logging.warning(f"A roll number column has {len(column_sorted)} bubbles. Skipping."); decoded_roll.append("X"); continue
        marked_count = 0; marked_digit = -1
        for i, bubble_c in enumerate(column_sorted):
            mask = np.zeros(gray.shape, dtype="uint8"); cv2.drawContours(mask, [bubble_c], -1, 255, -1)
            if cv2.mean(gray, mask=mask)[0] < grading_threshold: marked_digit = i; marked_count += 1
        if marked_count == 1: decoded_roll.append(str(marked_digit))
        else: decoded_roll.append("E")
    return "".join(decoded_roll)

# --- Function to Write Results to CSV (MODIFIED) ---
def write_results_to_csv(image_path, roll_number, student_answers, score):
    """Saves the detailed grading results to a CSV file named after the student's roll number."""
    
    # --- MODIFICATION: Name the CSV file based on the student's roll number ---
    # Check if the roll number is valid (e.g., 5 digits, no error characters).
    if roll_number and roll_number.isdigit() and len(roll_number) == 5:
        csv_filename = f"{roll_number}_results.csv"
    else:
        # Fallback to the original image name if the roll number is invalid
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        csv_filename = f"{base_name}_error_results.csv" # Add 'error' to signify a problem
        logging.warning(f"Invalid roll number '{roll_number}'. Saving results using image filename: {csv_filename}")

    logging.info(f"Writing results to {csv_filename}")
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Roll Number', roll_number])
            writer.writerow(['Score (%)', f"{score:.2f}"])
            writer.writerow([])
            writer.writerow(['Question Number', 'Student Answer', 'Correct Answer', 'Result'])
            for answer in student_answers:
                writer.writerow([
                    answer['question'],
                    answer['student_answer'],
                    answer['correct_answer'],
                    answer['result']
                ])
    except IOError as e:
        logging.error(f"Could not write to CSV file {csv_filename}. Reason: {e}")

# --- 6. Main Processing Function (Unchanged) ---
def process_omr_sheet(image_path):
    setup_logging()
    original_image = cv2.imread(image_path)
    if original_image is None:
        logging.error(f"Could not load image from {image_path}")
        return

    corners = find_fiducial_markers(original_image)
    if corners is None: return
        
    paper = four_point_transform(original_image, corners)
    h_paper, w_paper = paper.shape[:2]
    crop_margin_y = int(h_paper * 0.05); crop_margin_x = int(w_paper * 0.05)
    cropped_paper = paper[crop_margin_y:h_paper-crop_margin_y, crop_margin_x:w_paper-crop_margin_x]
    
    h_crop, w_crop, _ = cropped_paper.shape
    split_y = find_horizontal_separator(cropped_paper)
    
    if split_y == -1:
        logging.warning("Could not find horizontal separator. Falling back to percentage split.")
        split_y = int(h_crop * 0.22)
    else:
        logging.info(f"Found horizontal separator at y={split_y}")

    roll_section = cropped_paper[0:split_y, :]; mcq_section = cropped_paper[split_y + 15:, :]
    output_image = cropped_paper.copy()

    roll_number = decode_roll_number(roll_section, output_image, 0)
    logging.info(f"--- Decoded Roll Number: {roll_number} ---")

    mcq_gray = cv2.cvtColor(mcq_section, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(mcq_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_bubbles = [c for c in cnts if 15 <= cv2.boundingRect(c)[2] <= 50 and 15 <= cv2.boundingRect(c)[3] <= 50]
    logging.info(f"Found {len(all_bubbles)} bubble contours in MCQ section.")
    
    total_correct = 0
    student_answers_data = []
    
    if len(all_bubbles) >= 230:
        all_bubble_intensities = [cv2.mean(mcq_gray, mask=cv2.drawContours(np.zeros(mcq_gray.shape, dtype="uint8"), [c], -1, 255, -1))[0] for c in all_bubbles]
        grading_threshold = get_intensity_threshold(all_bubble_intensities)
        logging.info(f"Dynamic Intensity Threshold calculated: {grading_threshold:.2f}")

        page_width = mcq_gray.shape[1]
        col1, col2, col3 = [], [], []
        for c in all_bubbles:
            x = cv2.boundingRect(c)[0]
            if x < page_width / 3: col1.append(c)
            elif x < page_width * 2 / 3: col2.append(c)
            else: col3.append(c)
        
        for col_idx, column in enumerate([col1, col2, col3]):
            question_rows = group_bubbles_into_rows(column)
            if not question_rows: continue
            question_rows.sort(key=lambda r: cv2.boundingRect(r[0])[1])
            if len(question_rows) != 20: logging.warning(f"Column {col_idx+1} has {len(question_rows)} rows, expected 20.")
            
            for row_idx, row in enumerate(question_rows):
                question_num = (col_idx * 20) + row_idx
                if len(row) != 4: continue
                
                question_bubbles = contours.sort_contours(row, method="left-to-right")[0]
                marked_choice_idx = -1; marked_count = 0
                for choice_idx, bubble_contour in enumerate(question_bubbles):
                    mask = cv2.drawContours(np.zeros(mcq_gray.shape, dtype="uint8"), [bubble_contour], -1, 255, -1)
                    if cv2.mean(mcq_gray, mask=mask)[0] < grading_threshold:
                        marked_choice_idx = choice_idx; marked_count += 1

                color = (0, 0, 255); is_correct = False
                correct_answers = ANSWER_KEY.get(question_num)

                if marked_count == 1:
                    if isinstance(correct_answers, list) and marked_choice_idx in correct_answers: is_correct = True
                    elif marked_choice_idx == correct_answers: is_correct = True
                
                if is_correct: total_correct += 1; color = (0, 255, 0)
                
                student_answer_char = CHOICE_MAP.get(marked_choice_idx, 'Blank') if marked_count == 1 else ('Error' if marked_count > 1 else 'Blank')
                if isinstance(correct_answers, list): correct_answer_char = ", ".join([CHOICE_MAP.get(ans) for ans in correct_answers])
                else: correct_answer_char = CHOICE_MAP.get(correct_answers, 'N/A')
                
                student_answers_data.append({
                    'question': question_num + 1, 'student_answer': student_answer_char,
                    'correct_answer': correct_answer_char, 'result': 'Correct' if is_correct else 'Incorrect'
                })

                if correct_answers is not None:
                    answers_to_draw = correct_answers if isinstance(correct_answers, list) else [correct_answers]
                    for answer_idx in answers_to_draw:
                        if answer_idx < len(question_bubbles):
                            (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[answer_idx])
                            cv2.rectangle(output_image, (x_b, y_b + split_y + 15), (x_b + w_b, y_b + h_b + split_y + 15), color, 3)

    score = (total_correct / 60.0) * 100 if len(ANSWER_KEY) > 0 else 0
    logging.info(f"--- OMR Grading Results ---\nCorrect Answers: {total_correct} / 60\nScore: {score:.2f}%")
    
    write_results_to_csv(image_path, roll_number, student_answers_data, score)
    
    cv2.putText(output_image, f"Roll: {roll_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
    cv2.putText(output_image, f"Score: {score:.2f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    h_final, w_final = output_image.shape[:2]
    max_height = 900
    if h_final > max_height:
        ratio = max_height / h_final
        result_image = cv2.resize(output_image, (int(w_final * ratio), max_height))
    else:
        result_image = output_image

    cv2.imshow("Graded Sheet", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = 'im2.jpg' 
    process_omr_sheet(image_file)


