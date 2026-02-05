# This brings in the basic tools needed to build a window on your computer screen — like the frame, buttons, and menus.
import tkinter as tk

# These are extra tools from Tkinter: 	filedialog →open awindow so you can choose a file : messagebox →shows pop‑up messages 
from tkinter import filedialog, messagebox

# This gives you nicer‑looking buttons and widgets..
from tkinter import ttk

# PIL (Pillow) is a library that helps your program open and display pictures : 	Image → opens the picture file : ImageTk → converts the picture so Tkinter can show it on the screen
from PIL import Image, ImageTk

# This is OpenCV, a powerful image‑editing library
import cv2

# NumPy helps the computer do fast math on images
import numpy as np

# This helps your program work with files and folders on your computer.
import os


class ImageModel:
    """
    Handles image data and processing (Encapsulation of image state and operations).
    """

    def __init__(self):
        self.original_image = None       # Stored as OpenCV BGR image
        self.current_image = None        # Working image
        self.filename = None
        self.history = []                # For Undo
        self.redo_stack = []             # For Redo

    # --------------- Basic state helpers ---------------

    def has_image(self):
        return self.current_image is not None

    def _push_history(self):
        """Save current state for Undo."""
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            # Clear redo whenever a new operation is applied
            self.redo_stack.clear()

    def undo(self):
        if not self.history:
            return False
        self.redo_stack.append(self.current_image.copy())
        self.current_image = self.history.pop()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.history.append(self.current_image.copy())
        self.current_image = self.redo_stack.pop()
        return True

    # --------------- File operations ---------------

    def load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Could not read image.")
        self.original_image = img
        self.current_image = img.copy()
        self.filename = path
        self.history.clear()
        self.redo_stack.clear()

    def save_image(self, path=None):
        if not self.has_image():
            raise ValueError("No image to save.")
        if path is None:
            path = self.filename
        if path is None:
            raise ValueError("No filename specified.")
        cv2.imwrite(path, self.current_image)
        self.filename = path

    # --------------- Image operations using OpenCV ---------------

    def to_grayscale(self):
        if not self.has_image():
            return
        self._push_history()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        self.current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def blur(self, ksize=5):
        if not self.has_image():
            return
        # ksize must be odd and >= 1
        if ksize < 1:
            ksize = 1
        if ksize % 2 == 0:
            ksize += 1
        self._push_history()
        self.current_image = cv2.GaussianBlur(self.current_image, (ksize, ksize), 0)

    def canny_edges(self, threshold1=100, threshold2=200):
        if not self.has_image():
            return
        self._push_history()
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        self.current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def adjust_brightness_contrast(self, brightness=0, contrast=0):
        """
        brightness: -100 to 100
        contrast:   -100 to 100
        Implementation uses alpha (contrast) and beta (brightness).
        """
        if not self.has_image():
            return
        self._push_history()
        # Simple mapping
        alpha = 1.0 + (contrast / 100.0)    # contrast factor
        beta = brightness                  # brightness added directly
        new = cv2.convertScaleAbs(self.current_image, alpha=alpha, beta=beta)
        self.current_image = new

    def rotate(self, angle):
        if not self.has_image():
            return
        self._push_history()
        if angle == 90:
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_180)
        elif angle == 270:
            self.current_image = cv2.rotate(self.current_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def flip(self, mode="horizontal"):
        if not self.has_image():
            return
        self._push_history()
        if mode == "horizontal":
            self.current_image = cv2.flip(self.current_image, 1)
        elif mode == "vertical":
            self.current_image = cv2.flip(self.current_image, 0)

    def resize(self, scale_percent):
        """
        scale_percent: e.g., 50 for 50%, 200 for 200%, etc.
        """
        if not self.has_image():
            return
        if scale_percent <= 0:
            return
        self._push_history()
        width = int(self.current_image.shape[1] * scale_percent / 100)
        height = int(self.current_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        self.current_image = cv2.resize(self.current_image, dim, interpolation=cv2.INTER_AREA)


class ImageController:
    """
    Connects GUI actions to the ImageModel and notifies the GUI to update.
    """

    def __init__(self, model, view):
        self.model = model
        self.view = view

    # --------------- File methods ---------------

    def open_image(self):
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Open Image", filetypes=filetypes)
        if not path:
            return
        try:
            self.model.load_image(path)
            self.view.update_image_display()
            self.view.update_status_bar()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open image:\n{e}")

    def save_image(self):
        if not self.model.has_image():
            messagebox.showerror("Error", "No image to save.")
            return
        if self.model.filename is None:
            self.save_image_as()
            return
        try:
            self.model.save_image()
            messagebox.showinfo("Saved", f"Image saved to:\n{self.model.filename}")
            self.view.update_status_bar()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def save_image_as(self):
        if not self.model.has_image():
            messagebox.showerror("Error", "No image to save.")
            return
        filetypes = [
            ("JPEG", "*.jpg"),
            ("PNG", "*.png"),
            ("Bitmap", "*.bmp"),
            ("All files", "*.*")
        ]
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=filetypes, title="Save Image As"
        )
        if not path:
            return
        try:
            self.model.save_image(path)
            messagebox.showinfo("Saved", f"Image saved to:\n{path}")
            self.view.update_status_bar()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{e}")

    def exit_app(self, root):
        if messagebox.askokcancel("Exit", "Do you really want to exit?"):
            root.destroy()

    # --------------- Edit (Undo/Redo) ---------------

    def undo(self):
        if self.model.undo():
            self.view.update_image_display()
            self.view.update_status_bar()
        else:
            messagebox.showinfo("Undo", "Nothing to undo.")

    def redo(self):
        if self.model.redo():
            self.view.update_image_display()
            self.view.update_status_bar()
        else:
            messagebox.showinfo("Redo", "Nothing to redo.")

    # --------------- Filter/Effect methods ---------------

    def apply_grayscale(self):
        if not self.model.has_image():
            messagebox.showerror("Error", "Load an image first.")
            return
        self.model.to_grayscale()
        self.view.update_image_display()
        self.view.update_status_bar()

    def apply_blur_from_slider(self, value):
        # Slider passes a string value
        if not self.model.has_image():
            return
        ksize = int(float(value))
        if ksize < 1:
            ksize = 1
        self.model.blur(ksize)
        self.view.update_image_display()
        self.view.update_status_bar()

    def apply_canny(self):
        if not self.model.has_image():
            messagebox.showerror("Error", "Load an image first.")
            return
        self.model.canny_edges()
        self.view.update_image_display()
        self.view.update_status_bar()

    def apply_brightness_contrast(self, brightness, contrast):
        if not self.model.has_image():
            return
        self.model.adjust_brightness_contrast(brightness, contrast)
        self.view.update_image_display()
        self.view.update_status_bar()

    def rotate(self, angle):
        if not self.model.has_image():
            messagebox.showerror("Error", "Load an image first.")
            return
        self.model.rotate(angle)
        self.view.update_image_display()
        self.view.update_status_bar()

    def flip_horizontal(self):
        if not self.model.has_image():
            messagebox.showerror("Error", "Load an image first.")
            return
        self.model.flip("horizontal")
        self.view.update_image_display()
        self.view.update_status_bar()

    def flip_vertical(self):
        if not self.model.has_image():
            messagebox.showerror("Error", "Load an image first.")
            return
        self.model.flip("vertical")
        self.view.update_image_display()
        self.view.update_status_bar()

    def resize_from_entry(self, scale_str):
        if not self.model.has_image():
            messagebox.showerror("Error", "Load an image first.")
            return
        try:
            scale_percent = float(scale_str)
        except ValueError:
            messagebox.showerror("Error", "Scale must be a number.")
            return
        if scale_percent <= 0:
            messagebox.showerror("Error", "Scale must be > 0.")
            return
        self.model.resize(scale_percent)
        self.view.update_image_display()
        self.view.update_status_bar()


class MainApp:
    """
    Tkinter GUI: main window, menus, image display, controls, status bar.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("HIT137 Image Editor")
        self.root.geometry("1000x700")

        # Model and Controller
        self.model = ImageModel()
        self.controller = ImageController(self.model, self)

        # Tkinter image reference to avoid garbage collection
        self.tk_image = None

        # GUI components
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()

    # --------------- GUI Layout ---------------

    def _create_menu(self):
        menubar = tk.Menu(self.root)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.controller.open_image)
        file_menu.add_command(label="Save", command=self.controller.save_image)
        file_menu.add_command(label="Save As", command=self.controller.save_image_as)
        file_menu.add_separator()
        file_menu.add_command(
            label="Exit", command=lambda: self.controller.exit_app(self.root)
        )
        menubar.add_cascade(label="File", menu=file_menu)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.controller.undo)
        edit_menu.add_command(label="Redo", command=self.controller.redo)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        self.root.config(menu=menubar)

    def _create_main_layout(self):
        # Main horizontal split: left = image, right = controls
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: image display area
        self.image_label = ttk.Label(main_frame, background="gray")
        self.image_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right: control panel
        control_frame = ttk.Frame(main_frame, width=250)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)

        # ---- Filters/Buttons ----
        ttk.Label(control_frame, text="Filters / Effects").pack(pady=(0, 5))

        btn_gray = ttk.Button(
            control_frame, text="Grayscale", command=self.controller.apply_grayscale
        )
        btn_gray.pack(fill=tk.X, pady=2)

        btn_canny = ttk.Button(
            control_frame, text="Edge Detection (Canny)", command=self.controller.apply_canny
        )
        btn_canny.pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="Blur (Gaussian)").pack(pady=(10, 0))
        # Slider for adjustable blur intensity (Required: at least one slider)
        self.blur_slider = ttk.Scale(
            control_frame, from_=1, to=25, orient=tk.HORIZONTAL,
            command=self.controller.apply_blur_from_slider
        )
        self.blur_slider.set(5)
        self.blur_slider.pack(fill=tk.X, pady=2)

        # Brightness / Contrast sliders
        ttk.Label(control_frame, text="Brightness").pack(pady=(10, 0))
        self.brightness_slider = ttk.Scale(
            control_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
            command=lambda v: self._update_brightness_contrast()
        )
        self.brightness_slider.set(0)
        self.brightness_slider.pack(fill=tk.X, pady=2)

        ttk.Label(control_frame, text="Contrast").pack(pady=(10, 0))
        self.contrast_slider = ttk.Scale(
            control_frame, from_=-100, to=100, orient=tk.HORIZONTAL,
            command=lambda v: self._update_brightness_contrast()
        )
        self.contrast_slider.set(0)
        self.contrast_slider.pack(fill=tk.X, pady=2)

        # Rotation
        ttk.Label(control_frame, text="Rotate").pack(pady=(10, 0))
        rotate_frame = ttk.Frame(control_frame)
        rotate_frame.pack(fill=tk.X, pady=2)
        ttk.Button(
            rotate_frame, text="90°", command=lambda: self.controller.rotate(90)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(
            rotate_frame, text="180°", command=lambda: self.controller.rotate(180)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(
            rotate_frame, text="270°", command=lambda: self.controller.rotate(270)
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Flip
        ttk.Label(control_frame, text="Flip").pack(pady=(10, 0))
        flip_frame = ttk.Frame(control_frame)
        flip_frame.pack(fill=tk.X, pady=2)
        ttk.Button(
            flip_frame, text="Horizontal", command=self.controller.flip_horizontal
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(
            flip_frame, text="Vertical", command=self.controller.flip_vertical
        ).pack(side=tk.LEFT, expand=True, fill=tk.X)

        # Resize
        ttk.Label(control_frame, text="Resize / Scale (%)").pack(pady=(10, 0))
        self.resize_entry = ttk.Entry(control_frame)
        self.resize_entry.insert(0, "100")
        self.resize_entry.pack(fill=tk.X, pady=2)
        ttk.Button(
            control_frame, text="Apply Resize",
            command=lambda: self.controller.resize_from_entry(self.resize_entry.get())
        ).pack(fill=tk.X, pady=2)

    def _create_status_bar(self):
        self.status_var = tk.StringVar()
        self.status_var.set("No image loaded")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # --------------- GUI helpers ---------------

    def update_image_display(self):
        """
        Convert current OpenCV image (BGR) to Tkinter PhotoImage and show in label.
        """
        if not self.model.has_image():
            self.image_label.configure(image="", text="No image")
            return

        img = self.model.current_image
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Resize image to fit label if necessary
        label_width = self.image_label.winfo_width()
        label_height = self.image_label.winfo_height()

        if label_width > 10 and label_height > 10:
            pil_img.thumbnail((label_width, label_height))

        self.tk_image = ImageTk.PhotoImage(pil_img)
        self.image_label.configure(image=self.tk_image, text="")

    def update_status_bar(self):
        if not self.model.has_image():
            self.status_var.set("No image loaded")
            return
        h, w, c = self.model.current_image.shape
        name = (
            os.path.basename(self.model.filename)
            if self.model.filename is not None
            else "Unsaved Image"
        )
        self.status_var.set(f"File: {name} | Dimensions: {w}x{h} | Channels: {c}")

    def _update_brightness_contrast(self):
        """
        Called whenever brightness or contrast slider changes.
        Uses brightness/contrast relative to original_image to avoid compounding.
        """
        if not self.model.has_image() or self.model.original_image is None:
            return

        # Reset to original then apply brightness/contrast
        # This is one simple approach to keep changes consistent.
        self.model.current_image = self.model.original_image.copy()
        brightness = int(float(self.brightness_slider.get()))
        contrast = int(float(self.contrast_slider.get()))
        self.model.adjust_brightness_contrast(brightness, contrast)
        # Note: adjust_brightness_contrast already pushes history;
        # for slider behavior you might prefer not to push history each time.
        # For assignment clarity, this is kept simple.
        self.update_image_display()
        self.update_status_bar()


def main():
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
