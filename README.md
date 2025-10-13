# Advanced OMR Sheet Grader using OpenCV

This project provides a robust solution for automatically grading Optical Mark Recognition (OMR) answer sheets using Python and OpenCV. It is designed to be resilient to common issues like varying lighting conditions, shadows, and imperfect scans.

---

## Features

* **Automatic Sheet Alignment**: Detects four fiducial markers on the corners of the sheet to correct for perspective skew and rotation.
* **Roll Number Decoding**: Accurately reads the multi-digit roll number filled in by the student.
* **Robust MCQ Grading**: Employs a multi-stage decision logic to accurately grade multiple-choice questions, correctly identifying blank, single, and multiple marks.
* **Shadow & Lighting Resilience**: The grading logic is specifically designed to handle uneven lighting and shadows, preventing blank bubbles in shaded areas from being misidentified as marks.
* **Precise Intensity Measurement**: Calculates bubble intensity based on the central 50% area, ignoring the printed outlines for a more accurate reading.
* **Visual Feedback**: Generates a new image of the OMR sheet with the score, roll number, and color-coded feedback for each question (green for correct, red for incorrect).
* **Detailed Reporting**: Exports the final results, including the score and a question-by-question breakdown, to a CSV file for easy analysis.

---

## How It Works

The script follows a multi-step image processing pipeline to grade the OMR sheets:

1.  **Load & Align**: The image is loaded, and the four corner markers are located. A perspective transform is then applied to get a flat, top-down view of the sheet.
2.  **Section Splitting**: The aligned sheet is programmatically split into the roll number section and the four main MCQ answer columns.
3.  **Bubble Detection**: For each section, a bilateral filter is used to reduce noise while preserving edges. Adaptive thresholding is then applied to binarize the image, making it easy to find contours. These contours are filtered by size and aspect ratio to isolate the answer bubbles.
4.  **Grading Logic**: For each question row:
    * The intensity of the central 50% area of each of the four bubbles is measured.
    * A multi-stage check is performed:
        * **Contrast Check**: Is there a significant difference between the darkest and brightest bubble? If not, the row is blank.
        * **Absolute Darkness Check**: Is the darkest bubble actually dark enough to be a pencil mark? This prevents shadows from being counted.
        * **Separation Check**: Is the darkest bubble significantly darker than the second-darkest? This confirms a single, clear choice.
5.  **Output Generation**: The final score is calculated, and the results are drawn onto the output image and saved to a detailed CSV report named after the student's roll number.

---

## Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/omr-grader.git](https://github.com/your-username/omr-grader.git)
    cd omr-grader
    ```

2.  Install the required libraries:

    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
    Then, install the dependencies from the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

1.  Place your scanned OMR sheet images (e.g., in `.jpg` or `.png` format) into a directory (e.g., `scanned_sheets/`).

2.  Open the `omr_grader_fixed.py` script and update the `ANSWER_KEY` dictionary with the correct answers for your test.

3.  Modify the `image_file` variable in the main execution block at the bottom of the script to point to the OMR sheet you want to grade:
    ```python
    if __name__ == "__main__":
        # IMPORTANT: Replace this with the correct path to your image file
        image_file = 'scanned_sheets/student_01.jpg' 
        if os.path.exists(image_file):
            process_omr_sheet(image_file)
        else:
            print(f"Error: Image file not found at '{image_file}'.")
    ```

4.  Run the script from your terminal:
    ```bash
    python omr_grader_fixed.py
    ```
    The script will display the graded image on the screen. Press any key to close the window. A log file (`omr_grader.log`) and a results CSV (e.g., `12345_results.csv`) will be created in the same directory.

---

## Dependencies

* OpenCV
* NumPy
* imutils

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
