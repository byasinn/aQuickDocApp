import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance, ExifTags
import numpy as np
import cv2
from processing.background_removal import remove_background
import os

def ensure_correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def optimize_image(image_path, max_size_mb=1):
    img = Image.open(image_path)
    img = ensure_correct_orientation(img)

    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
    if file_size_mb > 2:

        scale_factor = (max_size_mb / file_size_mb) ** 0.5
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img = img.resize(new_size, Image.ANTIALIAS)

    optimized_path = "optimized_image.jpg"
    img.save(optimized_path, format="JPEG", quality=85)

    return optimized_path

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Photo Editor")
        self.root.geometry("1200x700")
        self.root.configure(bg="#302B27")

        self.image_path = None
        self.processed_image = None
        self.cropped_image = None
        self.crop_proportion = (3, 4)

        self.create_layout()

        self.back_button = tk.Button(self.tools_frame, text="Voltar", command=self.go_back, state="disabled", width=20)
        self.back_button.pack(side="bottom", pady=10)
        
        self.canvas.bind("<ButtonPress-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan) 
        self.canvas.bind("<ButtonRelease-1>", self.end_pan)

    def create_layout(self):
        self.tools_frame = tk.Frame(self.root, width=350, bg="#302B27")
        self.tools_frame.pack(side="left", fill="y", padx=20, pady=20)

        self.image_frame = tk.Frame(self.root, bg="#F5F3F5", relief="groove", bd=2)
        self.image_frame.pack(side="right", fill="both", expand=True, padx=10, pady=20)

        self.canvas = tk.Canvas(self.image_frame, bg="#F5F3F5", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.load_button = self.create_button(self.tools_frame, "Load Image", self.load_image, "#B2675E")
        self.load_button.pack(pady=10)

    def go_back(self):
        self.load_image()
        self.back_button.config(state="disabled")

    def create_button(self, parent, text, command, color):
        button = tk.Button(parent, text=text, command=command, bg=color, fg="white", font=("Segoe UI", 12, "bold"),
                           relief="flat", width=20, height=2, activebackground="#e0e0e0", activeforeground="black",
                           bd=0, highlightthickness=0)
        button.bind("<Enter>", lambda e: button.config(bg="#dcdcdc"))
        button.bind("<Leave>", lambda e: button.config(bg=color))
        return button

    def load_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.image_path:
            img = Image.open(self.image_path)
            img = ensure_correct_orientation(img)
            self.original_image = img
            self.processed_image = img
            self.display_image(img)
            self.show_adjustments_button()

    def display_image(self, img):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img.thumbnail((canvas_width, canvas_height))
        self.img_tk = ImageTk.PhotoImage(img)
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, anchor="center", image=self.img_tk)

    def show_remove_bg_button(self):
        self.remove_bg_button = self.create_button(self.tools_frame, "Remove Background", self.process_image, "#FF6347")
        self.remove_bg_button.pack(pady=10)

    def process_image(self):
        output_path = "output_image.png"
        self.processed_image = remove_background("temp_adjusted_image.png", output_path)
        self.display_image(self.processed_image)
        self.show_crop_options()


    def show_adjustments_button(self):
        self.adjustments_button = self.create_button(self.tools_frame, "Next", self.show_adjustments, "#4CAF50")
        self.adjustments_button.pack(pady=10)

    def show_adjustments(self):
        self.clear_tools_frame()
        self.create_adjustment_slider("Brightness", 0.5, 2.0, 1.0, self.update_image, "brightness_slider")
        self.create_adjustment_slider("Contrast", 0.5, 2.0, 1.0, self.update_image, "contrast_slider")
        self.create_adjustment_slider("Sharpness", 0.5, 2.0, 1.0, self.update_image, "sharpness_slider")
        self.create_adjustment_slider("Hue", -0.1, 0.1, 0.0, self.update_image, "hue_slider")
        self.create_adjustment_slider("Saturation", 0.0, 2.0, 1.0, self.update_image, "saturation_slider")

        self.next_button = self.create_button(self.tools_frame, "Remove Background", self.process_image, "#FFA500")
        self.next_button.pack(pady=10)

    def rotate_image(self):
        if self.processed_image:
            self.processed_image = self.processed_image.rotate(180, expand=True)
            self.display_image(self.processed_image)

    def create_adjustment_slider(self, label, min_val, max_val, default_val, command, attr_name):
        slider = tk.Scale(self.tools_frame, from_=min_val, to=max_val, resolution=0.01, label=label, orient="horizontal",
                          command=command, bg="#2f4f4f", fg="white", font=("Segoe UI", 10))
        slider.set(default_val)
        slider.pack(fill="x", padx=5, pady=5)
        setattr(self, attr_name, slider)

    def update_image(self, event=None):
        img = self.original_image.copy()
        img = ImageEnhance.Brightness(img).enhance(self.brightness_slider.get())
        img = ImageEnhance.Contrast(img).enhance(self.contrast_slider.get())
        img = ImageEnhance.Sharpness(img).enhance(self.sharpness_slider.get())

        img_np = np.array(img)
        hsv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        hue_adjust = int(self.hue_slider.get() * 180)
        hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_adjust) % 180
        hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * self.saturation_slider.get(), 0, 255)

        img = Image.fromarray(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))

        self.display_image(img)
        img.save("temp_adjusted_image.png")
        self.processed_image = img
    
    def start_pan(self, event):
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def end_pan(self, event):
        self.is_panning = False

    def do_pan(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y

            self.pan_offset_x += dx
            self.pan_offset_y += dy

            self.pan_start_x = event.x
            self.pan_start_y = event.y

            self.display_image_with_pan()

    def display_image_with_pan(self):
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img = self.processed_image.copy()
        img.thumbnail((canvas_width, canvas_height))
        self.img_tk = ImageTk.PhotoImage(img)

        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2 + self.pan_offset_x,
            canvas_height // 2 + self.pan_offset_y,
            anchor="center",
            image=self.img_tk
        )
    def show_crop_options(self):
        self.clear_tools_frame()
        self.create_button(self.tools_frame, "3:4", lambda: self.apply_auto_crop("3:4"), "#4CAF50").pack(pady=5)
        self.create_button(self.tools_frame, "1:1", lambda: self.apply_auto_crop("1:1"), "#4CAF50").pack(pady=5)
        self.create_button(self.tools_frame, "2:2", lambda: self.apply_auto_crop("2:2"), "#4CAF50").pack(pady=5)
        self.create_button(self.tools_frame, "5:7", lambda: self.apply_auto_crop("5:7"), "#4CAF50").pack(pady=5)
        for button in self.crop_ratio_buttons.values():
            button.pack(pady=5)

    def preview_crop(self, ratio):
        for widget in self.tools_frame.winfo_children():
            if isinstance(widget, tk.Button) and widget.cget("text") == "Comfirm Crop":
                widget.destroy()

        if ratio == "3:4":
            target_width, target_height = 900, 1200
        elif ratio == "1:1":
            target_width, target_height = 300, 300
        elif ratio == "2:2":
            target_width, target_height = 600, 600
        elif ratio == "5:7":
            target_width, target_height = 1500, 2100
        else:
            messagebox.showinfo("Erro", "Invalid Proportion.")
            return

        cropped_image = self.auto_crop_face_opencv(self.processed_image, target_width, target_height)
        if cropped_image:
            self.cropped_image = cropped_image
            self.display_image(cropped_image)

            self.confirm_crop_button = self.create_button(self.tools_frame, "Confirm Crop", lambda: self.apply_auto_crop(ratio), "#4CAF50")
            self.confirm_crop_button.pack(pady=10)
            messagebox.showinfo("Preview", "Crop preview generated. Click 'Confirm Crop' to apply.")
        else:
            messagebox.showinfo("Error", "No face detected for cropping.")

    def auto_crop_face_opencv(self, image, target_width, target_height):
        import cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            (x, y, w, h) = faces[0]

            if target_width == 300 and target_height == 300:  # Para 1:1
                distance_factor = 1.0
            elif target_width == 600 and target_height == 600:  # Para 2:2
                distance_factor = 1.0
            else:
                distance_factor = 1.8

            new_height = int(h * distance_factor)
            aspect_ratio = target_width / target_height
            new_width = int(new_height * aspect_ratio)

            center_x, center_y = x + w // 2, y + h // 2
            x1, y1 = max(0, center_x - new_width // 2), max(0, center_y - new_height // 2)
            x2, y2 = x1 + new_width, y1 + new_height

            cropped_image = image.crop((x1, y1, x2, y2)).resize((target_width, target_height), Image.LANCZOS)
            return cropped_image
        else:
            print("No face detected.")
            return None

    def apply_auto_crop(self, ratio):
        self.crop_ratio = ratio
        
        size_map = {
            "3:4": (900, 1200),
            "1:1": (300, 300),
            "2:2": (600, 600),
            "5:7": (500, 700)
        }
        target_width, target_height = size_map.get(ratio, (900, 1200))
        cropped_image = self.auto_crop_face_opencv(self.processed_image, target_width, target_height)

        if cropped_image:
            self.cropped_image = cropped_image
            self.cropped_image.save("cropped_image.png")
            self.display_image(self.cropped_image)
            self.show_quantity_options()
        else:
            messagebox.showinfo("Error", "No face detected for cropping.")

    def show_quantity_options(self):
        self.clear_tools_frame()
        self.create_button(self.tools_frame, "4 Fotos", lambda: self.create_collage(4), "#4CAF50").pack(pady=5)
        self.create_button(self.tools_frame, "8 Fotos", lambda: self.create_collage(8), "#4CAF50").pack(pady=5)
        self.create_button(self.tools_frame, "2 Fotos (5:7)", lambda: self.create_collage(2), "#4CAF50").pack(pady=5)

        if self.crop_ratio == "5:7":
            self.create_button(self.tools_frame, "2 Fotos", lambda: self.create_collage(2), "#4CAF50").pack(pady=5)
        
        self.back_button.config(state="normal")
        self.back_button.pack(side="bottom", pady=10)


    def create_collage(self, quantity):
        collage = Image.new("RGB", (4500, 3000), "gray" if self.crop_ratio != "3:4" else "white")

        size_map = {
            "1:1": (300, 300),
            "2:2": (600, 600),
            "3:4": (900, 1200),
            "5:7": (1500, 2100)
        }
        size = size_map.get(self.crop_ratio, (900, 1200))

        resized_photo = self.cropped_image.resize(size, Image.LANCZOS)

        positions = self.get_collage_positions(quantity, size[0], size[1])
        if not positions:
            print("Unsupported number of photos or aspect ratio.")
            return

        for i in range(quantity):
            x, y = positions[i]
            collage.paste(resized_photo, (x, y))

        collage.save("final_collage.png")
        collage.show()
        messagebox.showinfo("Sucess", "Collage created and image saved successfully.")
            
    def get_collage_positions(self, quantity, photo_width, photo_height):
        margin = 150
        spacing = 2 

        if self.crop_ratio == "3:4":
            if quantity == 4:
                return [
                    (margin, margin), 
                    (margin + photo_width + spacing, margin), 
                    (margin + 2 * (photo_width + spacing), margin), 
                    (margin + 3 * (photo_width + spacing), margin)
                ]
            elif quantity == 8:
                return [
                    (margin, margin), 
                    (margin + photo_width + spacing, margin), 
                    (margin + 2 * (photo_width + spacing), margin), 
                    (margin + 3 * (photo_width + spacing), margin),
                    (margin, margin + photo_height + spacing),
                    (margin + photo_width + spacing, margin + photo_height + spacing),
                    (margin + 2 * (photo_width + spacing), margin + photo_height + spacing),
                    (margin + 3 * (photo_width + spacing), margin + photo_height + spacing)
                ]

        elif self.crop_ratio == "2:2" or self.crop_ratio == "1:1":
            if quantity == 4:
                return [
                    (margin, margin), 
                    (margin + photo_width + spacing, margin), 
                    (margin, margin + photo_height + spacing), 
                    (margin + photo_width + spacing, margin + photo_height + spacing)
                ]

        elif self.crop_ratio == "5:7":
            if quantity == 2:
                return [
                    (margin, margin), 
                    (margin + photo_width + spacing, margin)
                ]

        return []

    def clear_tools_frame(self):
        for widget in self.tools_frame.winfo_children():
            widget.pack_forget()

root = tk.Tk()
app = App(root)
root.mainloop()
