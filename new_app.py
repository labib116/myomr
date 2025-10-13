import cv2
import numpy as np
from imutils import contours
import os
import csv
import logging
from PIL import Image # Pillow is still needed for compression logic

# --- 1. Setup & Configuration ---

def setup_logging():
    """Configures the logging system to output to both a file and the console."""
    # This check prevents adding handlers multiple times if the function is called again
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("omr_grader.log"),
                logging.StreamHandler()
            ]
        )

# Answer key and choice mapping remain the same
ANSWER_KEY = {
    0: 1, 1: 2, 2: 0, 3: 2, 4: 1, 5: 0, 6: 1, 7: 2, 8: 3, 9: 0, 10: 1, 11: 3, 12: 0, 13: 2, 14: 1,
    15: 0, 16: 3, 17: 2, 18: 1, 19: 0, 20: 1, 21: 2, 22: [0, 2], 23: 3, 24: 1, 25: 0, 26: 2, 27: 3, 28: 1, 29: 0,
    30: 3, 31: 2, 32: 1, 33: 0, 34: 1, 35: 2, 36: 0, 37: 3, 38: 1, 39: 2, 40: 0, 41: 1, 42: 3, 43: 2, 44: 0,
    45: 1, 46: 3, 47: 0, 48: 2, 49: 1, 50: 3, 51: 0, 52: 1, 53: 2, 54: 3, 55: 0, 56: 1, 57: 2, 58: 3, 59: 0
}
CHOICE_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}


# --- 2. Core Image Processing Utilities ---

def compress_image_on_the_fly(cv2_image, max_size=1920, quality=75):
    """
    [NEW IN-MEMORY COMPRESSION]
    Converts a cv2 (NumPy) image to a Pillow image, applies resizing and JPEG 
    compression in memory, and converts it back to a cv2 image.

    :param cv2_image: Input image as a NumPy array (BGR format).
    :param max_size: The maximum dimension (longest side) in pixels.
    :param quality: JPEG quality (1-95).
    :return: Compressed image as a NumPy array (BGR format).
    """
    try:
        # 1. Convert cv2 (BGR) to Pillow (RGB)
        # OpenCV reads as BGR, Pillow expects RGB
        img_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_rgb)
        
        # 2. Resizing logic
        width, height = img.size
        
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 3. Apply JPEG compression in memory
        # We need a BytesIO object to simulate saving to disk
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, "JPEG", optimize=True, quality=quality)
        buffer.seek(0)
        
        # 4. Read the compressed bytes back into a cv2 image
        file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        compressed_cv2_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        logging.info(f"Image compressed in memory. Original: {width}x{height}, New: {img.size}. Quality: {quality}")
        
        return compressed_cv2_image

    except Exception as e:
        logging.error(f"In-memory compression failed. Reason: {e}. Returning original image.")
        return cv2_image # Fallback: return the original image if compression fails


def reorder(myPoints):
    """Reorders four corner points: top-left, top-right, bottom-left, bottom-right."""
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 2), np.float32)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def four_point_transform(image, pts):
    """Applies a perspective transform to an image based on four points."""
    (tl, tr, bl, br) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [0, maxHeight - 1], [maxWidth - 1, maxHeight - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (maxWidth, maxHeight))

# ... (preprocess_image, find_fiducial_markers, resize_for_display, 
# find_and_denoise_bubbles, decode_roll_number, grade_mcq_section functions remain the same) ...

def preprocess_image(image):
    """Applies CLAHE and a median filter for robust noise and lighting correction."""
    # 1. Convert to a single channel (Luminance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(gray)
    
    # 3. Apply a Median Filter to remove salt-and-pepper noise/compression artifacts
    denoised_image = cv2.medianBlur(clahe_image, 5) # Kernel size 5 is a good starting point
    
    return denoised_image


def find_fiducial_markers(image):
    """Finds the four corner markers on the OMR sheet for alignment."""
    # This step uses the initial, possibly compressed, BGR image
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
            
    logging.info(f"Found {len(marker_contours)} potential fiducial markers.")
    if len(marker_contours) < 4:
        logging.error("Could not find 4 fiducial markers. Alignment failed.")
        return None
        
    marker_contours = sorted(marker_contours, key=cv2.contourArea, reverse=True)[:4]
    points = []
    for c in marker_contours:
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else int(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2)
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else int(cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2)
        points.append([cX, cY])
        
    return reorder(np.array(points, dtype="float32"))

# --- 3. UNIFIED Bubble Detection Function ---

def resize_for_display(image, max_height=800):
    """Resizes an image to a maximum height for display, maintaining aspect ratio."""
    h, w = image.shape[:2]
    if h > max_height:
        ratio = max_height / float(h)
        return cv2.resize(image, (int(w * ratio), max_height))
    return image


def find_and_denoise_bubbles(image_section, filter_params, debug_window_name="Debug Bubbles"):
    """
    A unified function to find bubbles in an image section.
    It uses bilateral filtering for robust, edge-preserving denoising.
    """
    gray = cv2.cvtColor(image_section, cv2.COLOR_BGR2GRAY)
    
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 19, 5)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubble_contours = []
    debug_image = image_section.copy()
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        ar = w / float(h)
        if (filter_params['min_w'] < w < filter_params['max_w'] and
            filter_params['min_h'] < h < filter_params['max_h'] and
            filter_params['min_ar'] <= ar <= filter_params['max_ar']):
            bubble_contours.append(c)
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
    display_debug_image = resize_for_display(debug_image)
    cv2.imshow(debug_window_name, display_debug_image)
    
    return bubble_contours, gray

# --- 4. Section-Specific Grading Logic ---

def decode_roll_number(roll_section_image):
    """Decodes the roll number from its specific section."""
    if roll_section_image is None or roll_section_image.size == 0:
        return "N/A"

    h, w = roll_section_image.shape[:2]
    crop_y_start = int(h * 0.35)
    bubble_area = roll_section_image[crop_y_start:, :]
    
    roll_filter_params = {'min_w': 15, 'max_w': 55, 'min_h': 15, 'max_h': 55, 'min_ar': 0.7, 'max_ar': 1.3}
    bubble_contours, gray = find_and_denoise_bubbles(bubble_area, roll_filter_params, "Debug Roll Number")
    
    logging.info(f"Roll Number: Found {len(bubble_contours)} bubbles after filtering.")
    if len(bubble_contours) < 45:
        return f"Not Enough Bubbles ({len(bubble_contours)})"

    bubble_contours, _ = contours.sort_contours(bubble_contours, method="left-to-right")
    columns = []
    current_column = [bubble_contours[0]]
    avg_bubble_width = np.mean([cv2.boundingRect(c)[2] for c in bubble_contours])
    
    for i in range(1, len(bubble_contours)):
        prev_x, _, prev_w, _ = cv2.boundingRect(bubble_contours[i-1])
        curr_x, _, _, _ = cv2.boundingRect(bubble_contours[i])
        if (curr_x - (prev_x + prev_w)) > (avg_bubble_width * 0.4):
            columns.append(current_column)
            current_column = [bubble_contours[i]]
        else:
            current_column.append(bubble_contours[i])
    columns.append(current_column)

    logging.info(f"Clustered roll number bubbles into {len(columns)} columns.")
    
    if 5 <= len(columns) <= 6:
        bubble_columns = columns[-5:]
    else:
        logging.error(f"Expected 5 or 6 columns for roll number, found {len(columns)}.")
        return "Column Count Error"

    decoded_roll = []
    all_intensities = [cv2.mean(gray, mask=cv2.drawContours(np.zeros_like(gray), [c], -1, 255, -1))[0] for c in bubble_contours]
    grading_threshold = np.mean(all_intensities) - (np.std(all_intensities))
    logging.info(f"Roll number grading threshold: {grading_threshold:.2f}")

    for column in bubble_columns:
        column_sorted, _ = contours.sort_contours(column, method="top-to-bottom")
        
        if len(column_sorted) > 10:
            logging.info(f"Found {len(column_sorted)} contours in a column, taking the bottom 10.")
            column_sorted = column_sorted[-10:]

        if len(column_sorted) != 10:
            logging.warning(f"Column has {len(column_sorted)} bubbles after cleanup (expected 10). Skipping.")
            decoded_roll.append("X")
            continue
        
        marked_count = 0
        marked_digit = -1
        for i, bubble_c in enumerate(column_sorted):
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [bubble_c], -1, 255, -1)
            if cv2.mean(gray, mask=mask)[0] < grading_threshold:
                marked_digit = i
                marked_count += 1
        
        decoded_roll.append(str(marked_digit) if marked_count == 1 else "E")
        
    return "".join(decoded_roll)

def grade_mcq_section(section_image, question_offset, output_image, section_x_offset, section_y_offset):
    """Grades a single MCQ section using a robust, per-question grading method."""
    if section_image is None or section_image.size == 0:
        return [], 0

    h_sec, w_sec = section_image.shape[:2]
    y_crop_offset = int(h_sec * 0.05)
    bubble_area = section_image[y_crop_offset:, :]
    
    mcq_filter_params = {'min_w': 8, 'max_w': 60, 'min_h': 8, 'max_h': 60, 'min_ar': 0.2, 'max_ar': 5.0}
    bubble_contours, gray = find_and_denoise_bubbles(bubble_area, mcq_filter_params, f"Debug MCQ Section {question_offset//15+1}")
    
    logging.info(f"MCQ Section {question_offset//15+1}: Found {len(bubble_contours)} bubbles.")
    if not bubble_contours:
        return [], 0

    question_rows = []
    if bubble_contours:
        bubble_contours = contours.sort_contours(bubble_contours, method="top-to-bottom")[0]
        row = [bubble_contours[0]]
        for i in range(1, len(bubble_contours)):
            if abs(cv2.boundingRect(bubble_contours[i])[1] - cv2.boundingRect(row[0])[1]) < 20:
                row.append(bubble_contours[i])
            else:
                question_rows.append(row)
                row = [bubble_contours[i]]
        question_rows.append(row)

    section_answers = []
    section_correct = 0

    for row_idx, row in enumerate(question_rows):
        question_num = question_offset + row_idx
        
        if len(row) < 4:
            logging.warning(f"Q:{question_num+1} - Found only {len(row)} bubbles. Skipping.")
            continue
        if len(row) > 5:
            logging.warning(f"Q:{question_num+1} - Found {len(row)} bubbles, likely noise. Taking rightmost 4 or 5.")
            row = contours.sort_contours(row, method="left-to-right")[0][-5:]
        if len(row) == 5:
            row = contours.sort_contours(row, method="left-to-right")[0][1:]
        
        question_bubbles = contours.sort_contours(row, method="left-to-right")[0]
        
        intensities = []
        for bubble_contour in question_bubbles:
            mask = np.zeros_like(gray)
            (x, y, w, h) = cv2.boundingRect(bubble_contour)
            center_x, center_y = x + w // 2, y + h // 2
            # --- UPDATED: Radius scaled for 50% of the area ---
            radius = int(min(w, h) / 2 * 0.707) 
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)
            intensities.append(cv2.mean(gray, mask=mask)[0])

        # --- NEW ROBUST GRADING LOGIC ---
        marked_choice_idx = -1
        marked_count = 0
        
        if len(intensities) == 4:
            sorted_intensities = sorted(intensities)
            min_intensity = sorted_intensities[0]
            second_min_intensity = sorted_intensities[1]
            max_intensity = sorted_intensities[3]

            # --- TUNABLE PARAMETERS ---
            CONTRAST_THRESHOLD = 60 
            BLANK_THRESHOLD = 170
            SEPARATION_GAP = 20

            # 1. Check if the row has enough contrast and isn't completely blank.
            if (max_intensity - min_intensity) > CONTRAST_THRESHOLD and min_intensity < BLANK_THRESHOLD:
                
                # 2. Check if the darkest bubble is clearly separated from the next darkest.
                if (second_min_intensity - min_intensity) > SEPARATION_GAP:
                    marked_count = 1
                    marked_choice_idx = intensities.index(min_intensity)
                else:
                    # If not clearly separated, it's likely a multi-mark error.
                    dark_threshold = second_min_intensity + (SEPARATION_GAP / 2.0)
                    dark_bubbles = [i for i in intensities if i < dark_threshold]
                    marked_count = len(dark_bubbles)
            else:
                # Not enough contrast or too bright, so it's a blank row.
                marked_count = 0
        
        print(f"Q:{question_num+1} Intensities: {[f'{i:.2f}' for i in intensities]}, Marked Count: {marked_count}, Marked Choice: {marked_choice_idx}")
        # --- END OF NEW LOGIC ---
        
        is_correct = False
        correct_answers = ANSWER_KEY.get(question_num)

        if marked_count == 1:
            if correct_answers is None:
                is_correct = False
            elif isinstance(correct_answers, list):
                if marked_choice_idx in correct_answers:
                    is_correct = True
            else:
                if marked_choice_idx == correct_answers:
                    is_correct = True
        
        if is_correct:
            section_correct += 1
            
        student_ans_char = CHOICE_MAP.get(marked_choice_idx, 'Blank') if marked_count == 1 else ('Error' if marked_count > 1 else 'Blank')
        correct_ans_char = ", ".join([CHOICE_MAP.get(ans) for ans in correct_answers]) if isinstance(correct_answers, list) else CHOICE_MAP.get(correct_answers, 'N/A')
        
        section_answers.append({'question': question_num + 1, 'student_answer': student_ans_char, 'correct_answer': correct_ans_char, 'result': 'Correct' if is_correct else 'Incorrect'})
        
        if correct_answers is not None:
            color = (0, 255, 0) if is_correct else (0, 0, 255)
            answers_to_draw = correct_answers if isinstance(correct_answers, list) else [correct_answers]
            
            for answer_idx in answers_to_draw:
                if answer_idx < len(question_bubbles):
                    (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[answer_idx])
                    final_x = x_b + section_x_offset
                    final_y = y_b + section_y_offset + y_crop_offset
                    cv2.rectangle(output_image, (final_x, final_y), (final_x + w_b, final_y + h_b), color, 3)
                    
            if not is_correct and marked_choice_idx != -1 and marked_count == 1:
                (x_b, y_b, w_b, h_b) = cv2.boundingRect(question_bubbles[marked_choice_idx])
                final_x = x_b + section_x_offset
                final_y = y_b + section_y_offset + y_crop_offset
                cv2.circle(output_image, (final_x + w_b//2, final_y + h_b//2), int(w_b/2), (0, 0, 255), 2)
                
    return section_answers, section_correct



# --- 5. Main Processing Pipeline ---

def process_omr_sheet(image_path):
    """
    Main function to load an image, apply in-memory compression pre-processing, 
    and run the entire grading process.
    """
    setup_logging()
    
    # 1. Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        logging.error(f"Could not load image from {image_path}. Check the path.")
        return

    # 2. Apply in-memory compression (NEW STEP)
    logging.info(f"Applying in-memory compression pre-processing...")
    compressed_image = compress_image_on_the_fly(original_image, max_size=1920, quality=75)
    
    # 3. Use the compressed image for the rest of the pipeline
    corners = find_fiducial_markers(compressed_image)
    if corners is None:
        return
    
    # The rest of the pipeline operates on the perspective-transformed image
    paper = four_point_transform(compressed_image, corners)
    
    h_paper, w_paper = paper.shape[:2]
    crop_margin_top = int(h_paper * 0.02)
    crop_margin_bottom = int(h_paper * 0.02)
    crop_margin_x_left = int(w_paper * 0.04)
    crop_margin_x_right = int(w_paper * 0.02)
    cropped_paper = paper[crop_margin_top:h_paper - crop_margin_bottom, crop_margin_x_left:w_paper - crop_margin_x_right]
    output_image = cropped_paper.copy()
    
    h_crop, w_crop, _ = cropped_paper.shape
    split_y = int(h_crop * 0.22)
    gray_split = cv2.cvtColor(cropped_paper, cv2.COLOR_BGR2GRAY)
    thresh_split = cv2.adaptiveThreshold(cv2.GaussianBlur(gray_split, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    cnts_split, _ = cv2.findContours(thresh_split, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_split:
        x, y, w, h = cv2.boundingRect(c)
        if w > w_crop * 0.8 and h < 20:
            split_y = y + h // 2
            logging.info(f"Found horizontal separator at y={split_y}")
            break
            
    roll_section = cropped_paper[0:split_y, :]
    mcq_section_full = cropped_paper[split_y:, :]

    roll_number = decode_roll_number(roll_section)
    logging.info(f"--- Decoded Roll Number: {roll_number} ---")

    total_correct = 0
    all_student_answers = []
    
    mcq_gray = cv2.cvtColor(mcq_section_full, cv2.COLOR_BGR2GRAY)
    mcq_thresh = cv2.adaptiveThreshold(cv2.GaussianBlur(mcq_gray, (5,5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detected_lines = cv2.morphologyEx(mcq_thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
    cnts_lines, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    separator_xs = []
    for c in cnts_lines:
        if cv2.boundingRect(c)[3] > mcq_section_full.shape[0] * 0.5:
            separator_xs.append(cv2.boundingRect(c)[0])
    
    separator_xs.sort()
    
    if len(separator_xs) >= 3:
        logging.info(f"Found vertical separators at x={separator_xs}")
        split_x1, split_x2, split_x3 = separator_xs[0], separator_xs[1], separator_xs[2]

        sections = [
            (mcq_section_full[:, :split_x1], 0, 0),
            (mcq_section_full[:, split_x1:split_x2], 15, split_x1),
            (mcq_section_full[:, split_x2:split_x3], 30, split_x2),
            (mcq_section_full[:, split_x3:], 45, split_x3)
        ]

        for sec_img, q_offset, x_offset in sections:
            answers, correct = grade_mcq_section(sec_img, q_offset, output_image, x_offset, split_y)
            all_student_answers.extend(answers)
            total_correct += correct
    else:
        logging.error(f"Failed to find 3 clear vertical separators. Found {len(separator_xs)}. Grading aborted.")

    score = total_correct
    logging.info(f"--- OMR Grading Results ---\nCorrect Answers: {total_correct} / {len(ANSWER_KEY)}\nScore: {score:.2f}%")
    
    all_student_answers.sort(key=lambda x: x['question'])
    
    csv_filename = f"{roll_number}_results.csv" if roll_number.isalnum() else f"{os.path.splitext(os.path.basename(image_path))[0]}_results.csv"
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Roll Number', roll_number])
            writer.writerow(['Score (%)', f"{score:.2f}"])
            writer.writerow([])
            writer.writerow(['Question Number', 'Student Answer', 'Correct Answer', 'Result'])
            for answer in all_student_answers:
                writer.writerow([answer['question'], answer['student_answer'], answer['correct_answer'], answer['result']])
        logging.info(f"Results written to {csv_filename}")
    except IOError as e:
        logging.error(f"Could not write to CSV file {csv_filename}. Reason: {e}")
        
    cv2.putText(output_image, f"Roll: {roll_number}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 100, 0), 3)
    cv2.putText(output_image, f"Score: {score:.2f}%", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    result_image = resize_for_display(output_image, max_height=700)
        
    cv2.imshow("Graded Sheet", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# --- Main execution block ---
if __name__ == "__main__":
    # IMPORTANT: Replace this with the correct path to your image file
    image_file = 'final_54.jpg' 
    
    # Check for PIL/Pillow installation
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow (PIL Fork) is not installed.")
        print("Please install it using: pip install Pillow")
        exit()
        
    if os.path.exists(image_file):
        process_omr_sheet(image_file)
    else:
        print(f"Error: Image file not found at '{image_file}'. Please update the path.")
