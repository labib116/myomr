import cv2
import numpy as np
from imutils import contours
import os
import csv
import logging

# --- New Function: Setup Logging ---
def setup_logging():
    """Configures the logging system to output to both a file and the console."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("omr_grader.log"),
                logging.StreamHandler()
            ]
        )

# --- 1. Define the Answer Key and Choice Mapping ---
ANSWER_KEY = {
    0: 1, 1: 3, 2: 0, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0, 10: 1, 11: 3, 12: 0, 13: 2, 14: 1,
    15: 0, 16: 3, 17: 2, 18: 1, 19: 0, 20: 1, 21: 2, 22: [0, 2], 23: 3, 24: 1, 25: 0, 26: 2, 27: 3, 28: 1, 29: 0,
    30: 3, 31: 2, 32: 1, 33: 0, 34: 1, 35: 2, 36: 0, 37: 3, 38: 1, 39: 2, 40: 0, 41: 1, 42: 3, 43: 2, 44: 0,
    45: 1, 46: 3, 47: 0, 48: 2, 49: 1, 50: 3, 51: 0, 52: 1, 53: 2, 54: 3, 55: 0, 56: 1, 57: 2, 58: 3, 59: 0
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
            cX = int(M["m10"] / M["m00"]); cY = int(M["m01"] / M["m00"])
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
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)); widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)); heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]], dtype="float32")
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
    rows = []; current_row = [bubbles[0]]
    for i in range(1, len(bubbles)):
        (x, y, w, h) = cv2.boundingRect(bubbles[i])
        (prev_x, prev_y, prev_w, prev_h) = cv2.boundingRect(current_row[0])
        if abs(y - prev_y) < tolerance: current_row.append(bubbles[i])
        else:
            rows.append(current_row); current_row = [bubbles[i]]
    rows.append(current_row)
    return rows

# --- 4. Separator Logic (Unchanged) ---
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

def find_vertical_separators(image, debug=False):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    if debug:
        cv2.imshow("Debug: Vertical Separator Threshold", cv2.resize(thresh, (w // 2, h // 2)))
        cv2.imshow("Debug: Detected Vertical Lines", cv2.resize(detected_lines, (w // 2, h // 2)))
        cv2.waitKey(0)
    cnts, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    logging.info(f"Found {len(cnts)} potential line contours. Analyzing each...")
    height_threshold = h * 0.5
    for i, c in enumerate(cnts):
        x, y, w_c, h_c = cv2.boundingRect(c)
        logging.info(f"  - Contour {i}: height={h_c}, width={w_c}. (Height threshold is {height_threshold:.2f})")
        if h_c > height_threshold and w_c < 30:
            lines.append(x + w_c // 2)
            logging.info(f"    -> ACCEPTED. Appending x-coordinate {x + w_c // 2}.")
        else:
            logging.info(f"    -> REJECTED.")
    if len(lines) > 0:
        lines.sort()
        clusters = []
        current_cluster = [lines[0]]
        for i in range(1, len(lines)):
            if lines[i] - current_cluster[-1] < 50:
                current_cluster.append(lines[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [lines[i]]
        clusters.append(current_cluster)
        separator_xs = [sum(cluster) // len(cluster) for cluster in clusters]
        logging.info(f"Found {len(separator_xs)} vertical separator clusters at x={separator_xs}")
        if len(separator_xs) >= 3:
            logging.info(f"Successfully identified 3+ separators. Using the first three.")
            separator_xs.sort()
            return separator_xs[0], separator_xs[1], separator_xs[2]
    logging.warning(f"Could not find 3 clear vertical separator lines. Found {len(lines)} accepted candidates which clustered into {len(separator_xs) if 'separator_xs' in locals() else 0} groups.")
    return None, None, None

# --- 5. Roll Number Decoding (MODIFIED with Debug Visuals) ---
def decode_roll_number(roll_section_image, output_image_to_draw, y_offset):
    if roll_section_image is None or roll_section_image.size == 0: return "N/A"
    
    h, w = roll_section_image.shape[:2]
    crop_y_start = int(h * 0.315)
    bubble_area = roll_section_image[crop_y_start:, :]
    
    # --- ADDITION: Create a debug image for the cropped bubble area ---
    debug_bubble_area = bubble_area.copy()

    gray = cv2.cvtColor(bubble_area, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    
    # --- ADDITION: Show the thresholded image for debugging ---
    cv2.imshow("Debug: Roll Number Threshold", thresh)

    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []

    # --- ADDITION: First, draw ALL found contours in RED ---
    for c in cnts:
        (x, y, w_c, h_c) = cv2.boundingRect(c)
        cv2.rectangle(debug_bubble_area, (x, y), (x + w_c, y + h_c), (0, 0, 255), 2) # Red for all candidates

    # --- EXISTING LOGIC with added drawing for accepted contours ---
    for c in cnts:
        (x, y, w_c, h_c) = cv2.boundingRect(c); ar = w_c / float(h_c)
        # This is the core logic you didn't want changed.
        if 10 < w_c < 50 and 10 < h_c < 50 and 0.7 <= ar <= 1.3:
            bubble_contours.append(c)
            # --- ADDITION: Draw accepted contours over in GREEN ---
            cv2.rectangle(debug_bubble_area, (x, y), (x + w_c, y + h_c), (0, 255, 0), 2) # Green for accepted

    # --- ADDITION: Show the final debug image with classifications ---
    cv2.imshow("Debug: Roll Number Contours (Red=All, Green=Accepted)", debug_bubble_area)

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

# --- 6. MCQ Section Grading Function (Unchanged from previous version) ---
def grade_mcq_section(section_image, question_offset, output_image, section_x_offset, section_y_offset):
    if section_image is None or section_image.size == 0: return [], 0
    h_sec, w_sec = section_image.shape[:2]
    debug_section_image = section_image.copy()
    section_image = section_image[int(h_sec * 0.05):, :]
    gray = cv2.cvtColor(section_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubble_contours = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c); ar = w / float(h)
        if 8 < w < 60 and 8 < h < 60 and 0.2 < ar < 5.0:
            bubble_contours.append(c)
            cv2.rectangle(debug_section_image, (x, y + int(h_sec * 0.05)), (x + w, y + h + int(h_sec * 0.05)), (0, 255, 255), 2)
    if not bubble_contours:
        logging.warning(f"MCQ Section {question_offset//15 + 1}: No contours found after filtering.")
        return [], 0
    bubble_contours, _ = contours.sort_contours(bubble_contours, method="left-to-right")
    columns = []; current_column = [bubble_contours[0]]
    avg_bubble_width = np.mean([cv2.boundingRect(c)[2] for c in bubble_contours])
    for i in range(1, len(bubble_contours)):
        prev_x, _, prev_w, _ = cv2.boundingRect(bubble_contours[i-1])
        curr_x, _, _, _ = cv2.boundingRect(bubble_contours[i])
        if (curr_x - (prev_x + prev_w)) > (avg_bubble_width * 0.3):
             columns.append(current_column)
             current_column = [bubble_contours[i]]
        else:
            current_column.append(bubble_contours[i])
    columns.append(current_column)
    logging.info(f"MCQ Section {question_offset//15 + 1}: Clustered into {len(columns)} columns.")
    if len(columns) == 5:
        bubble_columns = columns[1:]
        logging.info("  -> Found 5 columns, assuming first is question numbers. Discarding it.")
    elif len(columns) == 4:
        bubble_columns = columns
        logging.info("  -> Found 4 columns, assuming they are the answer bubbles.")
    else:
        logging.error(f"MCQ Section {question_offset//15 + 1}: Expected 4 or 5 columns, found {len(columns)}.")
        cv2.imshow(f"FAILED Section {question_offset//15 + 1}", debug_section_image)
        return [], 0
    all_bubbles_in_section = [c for col in bubble_columns for c in col]
    for c in all_bubbles_in_section:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(debug_section_image, (x, y + int(h_sec * 0.05)), (x + w, y + h + int(h_sec * 0.05)), (0, 255, 0), 2)
    cv2.imshow(f"Debug Contours: Section {question_offset//15 + 1}", debug_section_image)
    section_answers = []; section_correct = 0
    all_bubble_intensities = [cv2.mean(gray, mask=cv2.drawContours(np.zeros(gray.shape, dtype="uint8"), [c], -1, 255, -1))[0] for c in all_bubbles_in_section]
    grading_threshold = get_intensity_threshold(all_bubble_intensities)
    logging.info(f"MCQ Section {question_offset//15 + 1} grading threshold: {grading_threshold:.2f}")
    question_rows = group_bubbles_into_rows(all_bubbles_in_section, tolerance=20)
    for row_idx, row in enumerate(question_rows):
        question_num = question_offset + row_idx
        if len(row) != 4:
            logging.warning(f"Q:{question_num+1} - Expected 4 bubbles, found {len(row)}. Skipping.")
            continue
        question_bubbles = contours.sort_contours(row, method="left-to-right")[0]
        marked_choice_idx = -1; marked_count = 0
        for choice_idx, bubble_contour in enumerate(question_bubbles):
            mask = cv2.drawContours(np.zeros(gray.shape, dtype="uint8"), [bubble_contour], -1, 255, -1)
            if cv2.mean(gray, mask=mask)[0] < grading_threshold:
                marked_choice_idx = choice_idx; marked_count += 1
        is_correct = False
        correct_answers = ANSWER_KEY.get(question_num)
        if marked_count == 1:
            if isinstance(correct_answers, list) and marked_choice_idx in correct_answers: is_correct = True
            elif marked_choice_idx == correct_answers: is_correct = True
        if is_correct: section_correct += 1
        student_answer_char = CHOICE_MAP.get(marked_choice_idx, 'Blank') if marked_count == 1 else ('Error' if marked_count > 1 else 'Blank')
        if isinstance(correct_answers, list): correct_answer_char = ", ".join([CHOICE_MAP.get(ans) for ans in correct_answers])
        else: correct_answer_char = CHOICE_MAP.get(correct_answers, 'N/A')
        section_answers.append({'question': question_num + 1, 'student_answer': student_answer_char, 'correct_answer': correct_answer_char, 'result': 'Correct' if is_correct else 'Incorrect'})
        if correct_answers is not None:
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            answers_to_draw = correct_answers if isinstance(correct_answers, list) else [correct_answers]
            for answer_idx in answers_to_draw:
                if answer_idx < len(question_bubbles):
                    (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[answer_idx])
                    y_b_draw = y_b + int(h_sec * 0.05)
                    cv2.rectangle(output_image, (x_b + section_x_offset, y_b_draw + section_y_offset), (x_b + w_b + section_x_offset, y_b_draw + h_b + section_y_offset), color, 3)
    return section_answers, section_correct


# --- Function to Write Results to CSV (Unchanged) ---
def write_results_to_csv(image_path, roll_number, student_answers, score):
    if roll_number and roll_number.isdigit() and len(roll_number) == 5: csv_filename = f"{roll_number}_results.csv"
    else:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        csv_filename = f"{base_name}_error_results.csv"
        logging.warning(f"Invalid roll number '{roll_number}'. Saving results using image filename: {csv_filename}")
    logging.info(f"Writing results to {csv_filename}")
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile); writer.writerow(['Roll Number', roll_number]); writer.writerow(['Score (%)', f"{score:.2f}"])
            writer.writerow([]); writer.writerow(['Question Number', 'Student Answer', 'Correct Answer', 'Result'])
            for answer in student_answers:
                writer.writerow([answer['question'], answer['student_answer'], answer['correct_answer'], answer['result']])
    except IOError as e:
        logging.error(f"Could not write to CSV file {csv_filename}. Reason: {e}")

# --- Main Processing Function (Unchanged) ---
def process_omr_sheet(image_path):
    setup_logging()
    original_image = cv2.imread(image_path)
    if original_image is None: logging.error(f"Could not load image from {image_path}"); return

    corners = find_fiducial_markers(original_image)
    if corners is None: return

    paper = four_point_transform(original_image, corners)

    h_paper, w_paper = paper.shape[:2]
    crop_margin_top = int(h_paper * 0.02); crop_margin_bottom = int(h_paper * 0.02)
    crop_margin_x_left = int(w_paper * 0.04); crop_margin_x_right = int(w_paper * 0.02)
    cropped_paper = paper[crop_margin_top:h_paper - crop_margin_bottom, crop_margin_x_left:w_paper - crop_margin_x_right]
    cv2.imshow("Debugging: Cropped Paper", cv2.resize(cropped_paper, (int(w_paper*0.5), int(h_paper*0.5))))

    h_crop, w_crop, _ = cropped_paper.shape
    split_y = find_horizontal_separator(cropped_paper)
    if split_y == -1:
        logging.warning("Could not find horizontal separator. Falling back to 22% split.")
        split_y = int(h_crop * 0.22)
    logging.info(f"Final section split at y={split_y}")

    roll_section = cropped_paper[0:split_y, :]
    mcq_section_full = cropped_paper[split_y:, :]
    output_image = cropped_paper.copy()

    roll_number = decode_roll_number(roll_section, output_image, 0)
    logging.info(f"--- Decoded Roll Number: {roll_number} ---")

    total_correct, all_student_answers = 0, []
    split_x1, split_x2, split_x3 = find_vertical_separators(mcq_section_full, debug=False)

    if split_x1 and split_x2 and split_x3:
        section1 = mcq_section_full[:, :split_x1]
        section2 = mcq_section_full[:, split_x1:split_x2]
        section3 = mcq_section_full[:, split_x2:split_x3]
        section4 = mcq_section_full[:, split_x3:]

        cv2.imshow("Section 1", section1)
        cv2.imshow("Section 2", section2)
        cv2.imshow("Section 3", section3)
        cv2.imshow("Section 4", section4)

        answers1, correct1 = grade_mcq_section(section1, 0, output_image, 0, split_y)
        answers2, correct2 = grade_mcq_section(section2, 15, output_image, split_x1, split_y)
        answers3, correct3 = grade_mcq_section(section3, 30, output_image, split_x2, split_y)
        answers4, correct4 = grade_mcq_section(section4, 45, output_image, split_x3, split_y)

        total_correct = correct1 + correct2 + correct3 + correct4
        all_student_answers = answers1 + answers2 + answers3 + answers4
    else:
        logging.error("Failed to split MCQ section by vertical lines. Grading aborted.")

    score = (total_correct / 60.0) * 100 if len(ANSWER_KEY) > 0 else 0
    logging.info(f"--- OMR Grading Results ---\nCorrect Answers: {total_correct} / 60\nScore: {score:.2f}%")

    all_student_answers.sort(key=lambda x: x['question'])
    write_results_to_csv(image_path, roll_number, all_student_answers, score)

    cv2.putText(output_image, f"Roll: {roll_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
    cv2.putText(output_image, f"Score: {score:.2f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    h_final, w_final = output_image.shape[:2]
    max_height = 900
    if h_final > max_height:
        ratio = max_height / h_final; result_image = cv2.resize(output_image, (int(w_final * ratio), max_height))
    else: result_image = output_image

    cv2.imshow("Graded Sheet", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # --- MODIFIED: Switched to the new image file for testing ---
    image_file = 'final_3.jpg' 
    process_omr_sheet(image_file)


