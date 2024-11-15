# 📸 QuickDocApp

**An automated document photo editor**  
QuickDocApp is a Python application designed to make document photo editing easy and efficient. With features like background removal, brightness adjustment, and auto-cropping, this app is perfect for preparing photos for official documents.

---

## 📂 Directory Structure

Here's an overview of the project structure and where to place the required model file (`u2net.pth`):

```plaintext
QuickDocApp/
├── U2Net/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── u2net_refactor.py
│   │   └── u2net.py
│   └── __pycache__/
├── processing/
│   ├── background_removal.py
│   └── u2net.py
├── saved_models/
│   └── u2net/
│       └── u2net.pth    <-- Download and place the model file here
├── __pycache__/
├── config.json
├── input.jpg
├── main.py
└── setup.py
```

### ⚠️ Important: Download the Model File

To run this application, you need to download the `u2net.pth` model file and place it in the `saved_models/u2net/` directory. You can download the model from the following link:

[Download u2net.pth](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing)

---

## 🌟 Features

- **🔄 Background Removal:** Automatically removes the background from images.
- **🌞 Adjustable Brightness, Contrast, and Sharpness:** Allows fine-tuning for optimal photo quality.
- **📏 Auto-Cropping for Document Sizes:** Supports standard sizes like 3x4, 5x7, and 2x2.
- **🖼️ Collage Creation:** Creates a 15x10 cm layout with multiple photo arrangements (4, 8, or 2 photos) for easy printing.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Tkinter, PIL (Pillow), OpenCV
- **Model File:** Requires `u2net.pth` for background removal (see download link above)

---

## 🚀 Getting Started

1. **Clone the Repository**  
   git clone https://github.com/byasinn/QuickDocApp.git

2. **Install Requirements**  
   Ensure Python is installed, then install the dependencies:

   pip install -r requirements.txt

3. **Place the Model File**  
   Download `u2net.pth` from the link above and place it in `saved_models/u2net/`.

4. **Run the Application**  
   Start the application with:

   python main.py

---

## 📐 Usage Example

1. **Load an Image**  
   Select an image to edit; the app will open it in the editor.

2. **Adjust Brightness, Contrast, and Sharpness**  
   Use sliders for fine-tuning the photo.

3. **Background Removal & Cropping**  
   - Click on "Remove Background" for a clean image.
   - Choose from preset crop sizes (3x4, 5x7, 2x2).

4. **Save the Collage**  
   Export your document-ready photo or collage for easy printing.

---

## 🎯 Goals

- **User-Friendly Interface:** Provides a streamlined process for quick document photo editing.
- **High Efficiency for Document Photos:** Focuses on essential features to save time on editing.

---

**For more details and updates, visit the [GitHub repository](https://github.com/byasinn/QuickDocApp).**
