# ============================================================
# Image Processing Application
# ============================================================
# Group Name: DAN/EXT 01
# Group Members: 4
# Member 1: Md Hasibul Raihan - S397592
# Member 2: Tanisa Sanam Vabna - S397593
# Member 3: JESHIKA SAPKOTA - S399269
# Member 4: LADDA DAWSON - S382273
# ============================================================


# Import Required Libraries


# Tkinter is used for building the graphical user interface (GUI)
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Menu

# PIL (Python Imaging Library) is used to convert OpenCV images for Tkinter display
from PIL import Image, ImageTk

# OpenCV is used for image processing operations
import cv2

# NumPy is used for numerical operations on image arrays
import numpy as np

# Typing improves readability and maintainability
from typing import Optional, List



# ImageProcessor Class
# Handles all image loading, editing, and transformation logic
# Demonstrates encapsulation and abstraction (OOP principles)


class ImageProcessor:
    """
    Class responsible for all image processing operations.
    Demonstrates encapsulation and methods.
    """
    
    def __init__(self):
        """Constructor initializes the processor with no image loaded."""
        self._original_image: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None
        self._filename: str = ""

   
    # Image File Handling Methods
   
    
    def load_image(self, filepath: str) -> bool:
        """Load an image from file."""
        try:
            self._original_image = cv2.imread(filepath)
            if self._original_image is None:
                return False
            self._current_image = self._original_image.copy()
            self._filename = filepath
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def save_image(self, filepath: str) -> bool:
        """Save the current image to file."""
        if self._current_image is None:
            return False
        try:
            cv2.imwrite(filepath, self._current_image)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    

    
    # Getter and Utility Methods
   

    def get_current_image(self) -> Optional[np.ndarray]:
        """Get the current processed image."""
        return self._current_image
    
    def get_original_image(self) -> Optional[np.ndarray]:
        """Get the original unprocessed image."""
        return self._original_image
    
    def reset_to_original(self):
        """Reset current image to original."""
        if self._original_image is not None:
            self._current_image = self._original_image.copy()
    
    def get_dimensions(self) -> tuple:
        """Get image dimensions (height, width)."""
        if self._current_image is not None:
            return self._current_image.shape[:2]
        return (0, 0)
    
    def get_filename(self) -> str:
        """Get the current filename."""
        return self._filename
    
    
   
    # Image Processing Operations
    
    
    def convert_to_grayscale(self):
        """Convert image to grayscale."""
        if self._current_image is not None:
            gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
            self._current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def apply_blur(self, intensity: int = 5):
        """Apply Gaussian blur with adjustable intensity."""
        if self._current_image is not None:
            kernel_size = max(1, intensity * 2 + 1)
            self._current_image = cv2.GaussianBlur(
                self._current_image, (kernel_size, kernel_size), 0
            )
    
    def detect_edges(self, threshold1: int = 100, threshold2: int = 200):
        """Apply Canny edge detection."""
        if self._current_image is not None:
            gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, threshold1, threshold2)
            self._current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def adjust_brightness(self, value: int):
        """Adjust image brightness (-100 to 100)."""
        if self._current_image is not None:
            hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, value)
            v = np.clip(v, 0, 255)
            final_hsv = cv2.merge((h, s, v))
            self._current_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, value: float):
        """Adjust image contrast (0.5 to 3.0)."""
        if self._current_image is not None:
            self._current_image = cv2.convertScaleAbs(
                self._current_image, alpha=value, beta=0
            )
    
    def adjust_color_balance(self, red: int, green: int, blue: int):
        """Adjust RGB color channels (-100 to 100 for each)."""
        if self._current_image is not None:
            b, g, r = cv2.split(self._current_image)
            
            # Adjust each channel
            r = cv2.add(r, red)
            g = cv2.add(g, green)
            b = cv2.add(b, blue)
            
            # Clip values to valid range
            r = np.clip(r, 0, 255)
            g = np.clip(g, 0, 255)
            b = np.clip(b, 0, 255)
            
            # Merge channels back
            self._current_image = cv2.merge([b, g, r])
    
    def adjust_saturation(self, value: float):
        """Adjust image saturation (0.0 to 2.0)."""
        if self._current_image is not None:
            hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            
            # Adjust saturation
            s = s * value
            s = np.clip(s, 0, 255)
            
            # Merge and convert back
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            self._current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_hue(self, value: int):
        """Adjust image hue (-180 to 180)."""
        if self._current_image is not None:
            hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            
            # Adjust hue
            h = h + value
            h = np.where(h > 180, h - 180, h)
            h = np.where(h < 0, h + 180, h)
            
            # Merge and convert back
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            self._current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def rotate_image(self, angle: int):
        """Rotate image by specified angle (90, 180, 270)."""
        if self._current_image is not None:
            if angle == 90:
                self._current_image = cv2.rotate(
                    self._current_image, cv2.ROTATE_90_CLOCKWISE
                )
            elif angle == 180:
                self._current_image = cv2.rotate(
                    self._current_image, cv2.ROTATE_180
                )
            elif angle == 270:
                self._current_image = cv2.rotate(
                    self._current_image, cv2.ROTATE_90_COUNTERCLOCKWISE
                )
    
    def flip_image(self, direction: str):
        """Flip image horizontally or vertically."""
        if self._current_image is not None:
            if direction == "horizontal":
                self._current_image = cv2.flip(self._current_image, 1)
            elif direction == "vertical":
                self._current_image = cv2.flip(self._current_image, 0)
    
    def resize_image(self, scale: float):
        """Resize image by scale factor."""
        if self._current_image is not None:
            height, width = self._current_image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            self._current_image = cv2.resize(
                self._current_image, (new_width, new_height), 
                interpolation=cv2.INTER_LINEAR
            )


class HistoryManager:
    """
    Class to manage undo/redo functionality.
    Demonstrates encapsulation and state management.
    """
    
    def __init__(self, max_history: int = 20):
        """Constructor initializes history stacks."""
        self._history: List[np.ndarray] = []
        self._redo_stack: List[np.ndarray] = []
        self._max_history = max_history
    
    def save_state(self, image: np.ndarray):
        """Save current image state to history."""
        if image is not None:
            self._history.append(image.copy())
            if len(self._history) > self._max_history:
                self._history.pop(0)
            self._redo_stack.clear()
    
    def undo(self) -> Optional[np.ndarray]:
        """Undo to previous state."""
        if len(self._history) > 1:
            current = self._history.pop()
            self._redo_stack.append(current)
            return self._history[-1].copy()
        return None
    
    def redo(self) -> Optional[np.ndarray]:
        """Redo to next state."""
        if self._redo_stack:
            state = self._redo_stack.pop()
            self._history.append(state)
            return state.copy()
        return None
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._history) > 1
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
    
    def clear(self):
        """Clear all history."""
        self._history.clear()
        self._redo_stack.clear()


class ImageProcessorGUI:
    """
    Main GUI class that integrates ImageProcessor and HistoryManager.
    Demonstrates class interaction and composition.
    """
    
    def __init__(self, root: tk.Tk):
        """Constructor initializes the GUI and creates all components."""
        self.root = root
        self.root.title("Image Processor Pro")
        self.root.geometry("1200x800")
        
        # Initialize processor and history manager (class interaction)
        self.processor = ImageProcessor()
        self.history = HistoryManager()
        
        # Setup GUI components
        self._create_menu_bar()
        self._create_main_layout()
        self._create_control_panel()
        self._create_status_bar()
        
        # Display placeholder
        self._update_display()
    
    def _create_menu_bar(self):
        """Create the menu bar with File and Edit menus."""
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self._open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Capture from Camera", command=self._capture_from_camera, accelerator="Ctrl+C")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self._save_image, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As", command=self._save_as_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_app)
        
        # Edit menu
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self._undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Reset to Original", command=self._reset_image)
        
        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._open_image())
        self.root.bind('<Control-c>', lambda e: self._capture_from_camera())
        self.root.bind('<Control-s>', lambda e: self._save_image())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
    
    def _create_main_layout(self):
        """Create the main layout with image display area."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel will be created first (left side)
        # Image display area (right side)
        display_frame = ttk.LabelFrame(main_frame, text="Image Display", padding=10)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas for image
        self.canvas = tk.Canvas(display_frame, bg='gray20', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scroll = ttk.Scrollbar(display_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scroll = ttk.Scrollbar(display_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        
        # Store main_frame for control panel
        self.main_frame = main_frame
    
    def _create_control_panel(self):
        """Create the control panel with all image processing options."""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Scrollable frame
        canvas = tk.Canvas(control_frame, width=280)
        scrollbar = ttk.Scrollbar(control_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Basic Filters
        basic_frame = ttk.LabelFrame(scrollable_frame, text="Basic Filters", padding=10)
        basic_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(basic_frame, text="Grayscale", command=self._apply_grayscale).pack(fill=tk.X, pady=2)
        ttk.Button(basic_frame, text="Edge Detection", command=self._apply_edge_detection).pack(fill=tk.X, pady=2)
        
        # Blur Effect with Slider
        blur_frame = ttk.LabelFrame(scrollable_frame, text="Blur Effect", padding=10)
        blur_frame.pack(fill=tk.X, pady=5)
        
        self.blur_var = tk.IntVar(value=3)
        ttk.Label(blur_frame, text="Intensity:").pack()
        blur_slider = ttk.Scale(blur_frame, from_=1, to=10, variable=self.blur_var, orient=tk.HORIZONTAL)
        blur_slider.pack(fill=tk.X)
        ttk.Button(blur_frame, text="Apply Blur", command=self._apply_blur).pack(fill=tk.X, pady=2)
        
        # Brightness Adjustment
        brightness_frame = ttk.LabelFrame(scrollable_frame, text="Brightness", padding=10)
        brightness_frame.pack(fill=tk.X, pady=5)
        
        self.brightness_var = tk.IntVar(value=0)
        ttk.Label(brightness_frame, text="Level (-100 to 100):").pack()
        brightness_slider = ttk.Scale(brightness_frame, from_=-100, to=100, variable=self.brightness_var, orient=tk.HORIZONTAL)
        brightness_slider.pack(fill=tk.X)
        ttk.Button(brightness_frame, text="Apply Brightness", command=self._apply_brightness).pack(fill=tk.X, pady=2)
        
        # Contrast Adjustment
        contrast_frame = ttk.LabelFrame(scrollable_frame, text="Contrast", padding=10)
        contrast_frame.pack(fill=tk.X, pady=5)
        
        self.contrast_var = tk.DoubleVar(value=1.0)
        ttk.Label(contrast_frame, text="Level (0.5 to 3.0):").pack()
        contrast_slider = ttk.Scale(contrast_frame, from_=0.5, to=3.0, variable=self.contrast_var, orient=tk.HORIZONTAL)
        contrast_slider.pack(fill=tk.X)
        ttk.Button(contrast_frame, text="Apply Contrast", command=self._apply_contrast).pack(fill=tk.X, pady=2)
        
        # Color Adjustment - RGB Channels
        color_frame = ttk.LabelFrame(scrollable_frame, text="Color Balance (RGB)", padding=10)
        color_frame.pack(fill=tk.X, pady=5)
        
        # Red channel
        ttk.Label(color_frame, text="Red (-100 to 100):", foreground="red").pack(anchor=tk.W)
        self.red_var = tk.IntVar(value=0)
        red_slider = ttk.Scale(color_frame, from_=-100, to=100, variable=self.red_var, orient=tk.HORIZONTAL)
        red_slider.pack(fill=tk.X)
        
        # Green channel
        ttk.Label(color_frame, text="Green (-100 to 100):", foreground="green").pack(anchor=tk.W, pady=(5,0))
        self.green_var = tk.IntVar(value=0)
        green_slider = ttk.Scale(color_frame, from_=-100, to=100, variable=self.green_var, orient=tk.HORIZONTAL)
        green_slider.pack(fill=tk.X)
        
        # Blue channel
        ttk.Label(color_frame, text="Blue (-100 to 100):", foreground="blue").pack(anchor=tk.W, pady=(5,0))
        self.blue_var = tk.IntVar(value=0)
        blue_slider = ttk.Scale(color_frame, from_=-100, to=100, variable=self.blue_var, orient=tk.HORIZONTAL)
        blue_slider.pack(fill=tk.X)
        
        ttk.Button(color_frame, text="Apply Color Balance", command=self._apply_color_balance).pack(fill=tk.X, pady=(5,2))
        ttk.Button(color_frame, text="Reset Colors", command=self._reset_color_sliders).pack(fill=tk.X, pady=2)
        
        # Saturation Adjustment
        saturation_frame = ttk.LabelFrame(scrollable_frame, text="Saturation", padding=10)
        saturation_frame.pack(fill=tk.X, pady=5)
        
        self.saturation_var = tk.DoubleVar(value=1.0)
        ttk.Label(saturation_frame, text="Level (0.0 to 2.0):").pack()
        saturation_slider = ttk.Scale(saturation_frame, from_=0.0, to=2.0, variable=self.saturation_var, orient=tk.HORIZONTAL)
        saturation_slider.pack(fill=tk.X)
        ttk.Button(saturation_frame, text="Apply Saturation", command=self._apply_saturation).pack(fill=tk.X, pady=2)
        
        # Hue Adjustment
        hue_frame = ttk.LabelFrame(scrollable_frame, text="Hue Shift", padding=10)
        hue_frame.pack(fill=tk.X, pady=5)
        
        self.hue_var = tk.IntVar(value=0)
        ttk.Label(hue_frame, text="Shift (-180 to 180):").pack()
        hue_slider = ttk.Scale(hue_frame, from_=-180, to=180, variable=self.hue_var, orient=tk.HORIZONTAL)
        hue_slider.pack(fill=tk.X)
        ttk.Button(hue_frame, text="Apply Hue Shift", command=self._apply_hue).pack(fill=tk.X, pady=2)
        
        # Rotation
        rotation_frame = ttk.LabelFrame(scrollable_frame, text="Rotation", padding=10)
        rotation_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(rotation_frame, text="Rotate 90°", command=lambda: self._apply_rotation(90)).pack(fill=tk.X, pady=2)
        ttk.Button(rotation_frame, text="Rotate 180°", command=lambda: self._apply_rotation(180)).pack(fill=tk.X, pady=2)
        ttk.Button(rotation_frame, text="Rotate 270°", command=lambda: self._apply_rotation(270)).pack(fill=tk.X, pady=2)
        
        # Flip
        flip_frame = ttk.LabelFrame(scrollable_frame, text="Flip", padding=10)
        flip_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(flip_frame, text="Flip Horizontal", command=lambda: self._apply_flip("horizontal")).pack(fill=tk.X, pady=2)
        ttk.Button(flip_frame, text="Flip Vertical", command=lambda: self._apply_flip("vertical")).pack(fill=tk.X, pady=2)
        
        # Resize
        resize_frame = ttk.LabelFrame(scrollable_frame, text="Resize", padding=10)
        resize_frame.pack(fill=tk.X, pady=5)
        
        self.scale_var = tk.DoubleVar(value=1.0)
        ttk.Label(resize_frame, text="Scale (0.1 to 3.0):").pack()
        scale_slider = ttk.Scale(resize_frame, from_=0.1, to=3.0, variable=self.scale_var, orient=tk.HORIZONTAL)
        scale_slider.pack(fill=tk.X)
        ttk.Button(resize_frame, text="Apply Resize", command=self._apply_resize).pack(fill=tk.X, pady=2)
    
    def _create_status_bar(self):
        """Create the status bar to display image information."""
        self.status_bar = ttk.Label(self.root, text="No image loaded", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _update_status_bar(self):
        """Update status bar with current image information."""
        if self.processor.get_current_image() is not None:
            height, width = self.processor.get_dimensions()
            filename = self.processor.get_filename().split('/')[-1]
            if not filename:
                filename = self.processor.get_filename().split('\\')[-1]
            self.status_bar.config(
                text=f"File: {filename} | Dimensions: {width}x{height}px"
            )
        else:
            self.status_bar.config(text="No image loaded")
    
    def _update_display(self):
        """Update the canvas with the current image centered."""
        img = self.processor.get_current_image()
        if img is not None:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(pil_img)
            
            # Clear canvas
            self.canvas.delete("all")
            
            # Get canvas dimensions
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Calculate center position
            x_center = canvas_width // 2
            y_center = canvas_height // 2
            
            # Create image centered on canvas
            self.canvas.create_image(x_center, y_center, anchor=tk.CENTER, image=self.photo)
            
            # Update scroll region
            img_width = pil_img.width
            img_height = pil_img.height
            x1 = x_center - img_width // 2
            y1 = y_center - img_height // 2
            x2 = x_center + img_width // 2
            y2 = y_center + img_height // 2
            self.canvas.config(scrollregion=(x1, y1, x2, y2))
        else:
            self.canvas.delete("all")
            self.canvas.update_idletasks()
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.canvas.create_text(
                canvas_width // 2 if canvas_width > 1 else 400,
                canvas_height // 2 if canvas_height > 1 else 300,
                text="No image loaded\nUse File > Open to load an image",
                font=("Arial", 16), fill="white"
            )
        
        self._update_status_bar()
    
    def _save_to_history(self):
        """Save current state to history before making changes."""
        img = self.processor.get_current_image()
        if img is not None:
            self.history.save_state(img)
    
    # File operations
    
    def _capture_from_camera(self):
        """Capture image from camera."""
        # Create camera capture window
        camera_window = tk.Toplevel(self.root)
        camera_window.title("Camera Capture")
        camera_window.geometry("800x650")
        
        # Camera preview label
        preview_label = tk.Label(camera_window)
        preview_label.pack(pady=10)
        
        # Button frame
        button_frame = ttk.Frame(camera_window)
        button_frame.pack(pady=10)
        
        # Open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            camera_window.destroy()
            return
        
        captured_image = None
        is_running = True
        
        def update_camera():
            """Update camera preview."""
            if is_running:
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Resize for preview
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    # Convert to PhotoImage
                    img = Image.fromarray(frame_resized)
                    photo = ImageTk.PhotoImage(img)
                    preview_label.config(image=photo)
                    preview_label.image = photo
                
                # Continue updating
                preview_label.after(10, update_camera)
        
        def capture():
            """Capture the current frame."""
            nonlocal captured_image, is_running
            ret, frame = cap.read()
            if ret:
                captured_image = frame.copy()
                is_running = False
                cap.release()
                
                # Load captured image into processor
                self.processor._original_image = captured_image
                self.processor._current_image = captured_image.copy()
                self.processor._filename = "captured_image.jpg"
                
                self.history.clear()
                self._save_to_history()
                self._update_display()
                
                camera_window.destroy()
                messagebox.showinfo("Success", "Image captured successfully!")
        
        def cancel():
            """Cancel camera capture."""
            nonlocal is_running
            is_running = False
            cap.release()
            camera_window.destroy()
        
        # Buttons
        ttk.Button(button_frame, text="📷 Capture", command=capture, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=cancel, width=15).pack(side=tk.LEFT, padx=5)
        
        # Info label
        info_label = ttk.Label(camera_window, text="Position yourself and click 'Capture' to take a photo", 
                               font=("Arial", 10))
        info_label.pack(pady=5)
        
        # Start camera preview
        update_camera()
        
        # Handle window close
        camera_window.protocol("WM_DELETE_WINDOW", cancel)
    
    def _open_image(self):
        """Open an image file."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(title="Open Image", filetypes=filetypes)
        
        if filepath:
            if self.processor.load_image(filepath):
                self.history.clear()
                self._save_to_history()
                self._update_display()
                messagebox.showinfo("Success", "Image loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load image.")
    
    def _save_image(self):
        """Save the current image."""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        
        filepath = self.processor.get_filename()
        if not filepath:
            self._save_as_image()
            return
        
        if self.processor.save_image(filepath):
            messagebox.showinfo("Success", "Image saved successfully!")
        else:
            messagebox.showerror("Error", "Failed to save image.")
    
    def _save_as_image(self):
        """Save the current image with a new filename."""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        
        filetypes = [
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg"),
            ("BMP files", "*.bmp"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(
            title="Save Image As",
            filetypes=filetypes,
            defaultextension=".png"
        )
        
        if filepath:
            if self.processor.save_image(filepath):
                messagebox.showinfo("Success", "Image saved successfully!")
            else:
                messagebox.showerror("Error", "Failed to save image.")
    
    def _exit_app(self):
        """Exit the application."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
    
    # Edit operations
    
    def _undo(self):
        """Undo last operation."""
        if self.history.can_undo():
            restored = self.history.undo()
            if restored is not None:
                self.processor._current_image = restored
                self._update_display()
        else:
            messagebox.showinfo("Info", "Nothing to undo.")
    
    def _redo(self):
        """Redo last undone operation."""
        if self.history.can_redo():
            restored = self.history.redo()
            if restored is not None:
                self.processor._current_image = restored
                self._update_display()
        else:
            messagebox.showinfo("Info", "Nothing to redo.")
    
    def _reset_image(self):
        """Reset image to original."""
        if self.processor.get_original_image() is not None:
            self._save_to_history()
            self.processor.reset_to_original()
            self._update_display()
    
    # Image processing operations
    
    def _apply_grayscale(self):
        """Apply grayscale filter."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.convert_to_grayscale()
            self._update_display()
    
    def _apply_blur(self):
        """Apply blur effect."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.apply_blur(self.blur_var.get())
            self._update_display()
    
    def _apply_edge_detection(self):
        """Apply edge detection."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.detect_edges()
            self._update_display()
    
    def _apply_brightness(self):
        """Apply brightness adjustment."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.adjust_brightness(self.brightness_var.get())
            self._update_display()
    
    def _apply_contrast(self):
        """Apply contrast adjustment."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.adjust_contrast(self.contrast_var.get())
            self._update_display()
    
    def _apply_color_balance(self):
        """Apply RGB color balance adjustment."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.adjust_color_balance(
                self.red_var.get(),
                self.green_var.get(),
                self.blue_var.get()
            )
            self._update_display()
    
    def _reset_color_sliders(self):
        """Reset color sliders to default."""
        self.red_var.set(0)
        self.green_var.set(0)
        self.blue_var.set(0)
    
    def _apply_saturation(self):
        """Apply saturation adjustment."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.adjust_saturation(self.saturation_var.get())
            self._update_display()
    
    def _apply_hue(self):
        """Apply hue shift."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.adjust_hue(self.hue_var.get())
            self._update_display()
    
    def _apply_rotation(self, angle: int):
        """Apply rotation."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.rotate_image(angle)
            self._update_display()
    
    def _apply_flip(self, direction: str):
        """Apply flip."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.flip_image(direction)
            self._update_display()
    
    def _apply_resize(self):
        """Apply resize."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.resize_image(self.scale_var.get())
            self._update_display()


def main():
    """Main function to run the application."""
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
