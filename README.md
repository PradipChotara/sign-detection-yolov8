# Sign Detection with YOLOv8 👋

This project is a real-time AI app that detects hand signs using your webcam. We trained a **YOLOv8** model to recognize different signs, and we built a simple web interface using **Streamlit** so anyone can use it easily.

## 🌟 What Makes This Project Special?

### 1. No Manual Labeling (Automated with MediaPipe)
Usually, training an AI requires drawing boxes around objects in thousands of photos by hand. **We didn't do that.**
Instead, we wrote a smart script using **MediaPipe Hands** to do the hard work for us:
* **It finds the hand:** MediaPipe detects the hand landmarks in the image.
* **It draws the box:** The script automatically calculates the box around the hand.
* **It saves the data:** It converts everything into the format YOLOv8 needs.
This makes our process fast, consistent, and fully automated.

### 2. We Made the AI "Tougher" (Augmentation)
To make sure the model works well in real life, we didn't just give it perfect photos. We "augmented" the training data by:
* **Rotating** images (±30°).
* **Zooming** in and out (Scaling).
* **Changing brightness** and contrast.
* **Blurring** some images slightly.
This helps the AI recognize signs even if the lighting is bad or the hand is tilted.

---

## 📊 How Well Does It Work?

We trained the model on **1,080 images** for 20 epochs. The results were excellent!
* **Overall Precision:** 97.5%
* **Overall Accuracy (mAP):** 92.7%

Here is the breakdown by class:

| Class | How Precise? (Precision) | Accuracy Score (mAP50-95) |
| :--- | :--- | :--- |
| **1** | 93.5% | 86.1% |
| **2** | 100% | 88.2% |
| **3** | 93.7% | 96.1% |
| **4** | 98.6% | 92.2% |
| **5** | 97.6% | 99.5% |
| **6** | 98.3% | 91.6% |
| **7** | 98.4% | 86.2% |
| **8** | 99.3% | 97.6% |
| **9** | 98.1% | 96.7% |
| **All** | **97.5%** | **92.7%** |

---

## 🛠️ Installation & Local Run

Follow these steps to run the project on your computer.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/PradipChotara/sign-detection-yolov8.git](https://github.com/PradipChotara/sign-detection-yolov8.git)
    cd sign-detection-yolov8
    ```

2.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure
* `app.py`: The main file that runs the Streamlit web app.
* `requirements.txt`: List of Python libraries needed.
* `assets/`: Folder containing images or other static files.