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
# 
# This is a professional image and video editing application built with:
# - Tkinter for GUI
# - OpenCV (cv2) for image processing
# - PIL for image display
# - NumPy for array operations
#
# Key Features:
# - Hover-based category navigation (no clicking needed!)
# - Video frame extraction
# - Full color palette with custom picker
# - Before/After comparison view
# - Professional logo and branding
# ============================================================


# ============================================================
# IMPORT SECTION - All required libraries
# ============================================================

import tkinter as tk  # Main GUI framework
from tkinter import ttk, filedialog, messagebox, Menu, colorchooser  # GUI components
from PIL import Image, ImageTk, ImageDraw, ImageFont  # Image handling for display
import cv2  # OpenCV for image processing
import numpy as np  # Numerical operations
from typing import Optional, List  # Type hints for better code


# ============================================================
# ImageProcessor Class
# ============================================================
# This class handles ALL image processing operations
# It uses encapsulation (private variables with _) to protect data
# Demonstrates OOP principles: Encapsulation, Abstraction, Methods
# ============================================================

class ImageProcessor:
    """
    Main image processing engine.
    Handles loading, saving, and all image transformations.
    """
    
    def __init__(self):
        """
        Constructor - initializes the processor with no image loaded.
        All variables start as None until an image is loaded.
        """
        self._original_image: Optional[np.ndarray] = None  # Stores the original unmodified image
        self._current_image: Optional[np.ndarray] = None   # Stores the currently edited image
        self._filename: str = ""  # Stores the filename

    # ============================================================
    # FILE HANDLING METHODS
    # These methods handle loading and saving images
    # ============================================================
    
    def load_image(self, filepath: str) -> bool:
        """
        Load an image from a file path.
        
        Args:
            filepath: Full path to the image file
            
        Returns:
            True if successful, False if failed
        """
        try:
            # cv2.imread reads the image as a numpy array in BGR format
            self._original_image = cv2.imread(filepath)
            
            if self._original_image is None:
                return False  # Image failed to load
            
            # Create a copy for editing (preserves original)
            self._current_image = self._original_image.copy()
            self._filename = filepath
            return True
            
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def load_image_from_array(self, img_array: np.ndarray, filename: str = "extracted_frame.jpg"):
        """
        Load an image from a numpy array (used for video frames and camera).
        
        Args:
            img_array: The image as a numpy array (from video or camera)
            filename: Name to give this image
        """
        self._original_image = img_array.copy()
        self._current_image = img_array.copy()
        self._filename = filename
    
    def save_image(self, filepath: str) -> bool:
        """
        Save the current edited image to a file.
        
        Args:
            filepath: Where to save the image
            
        Returns:
            True if successful, False if failed
        """
        if self._current_image is None:
            return False  # No image to save
            
        try:
            cv2.imwrite(filepath, self._current_image)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False

    # ============================================================
    # GETTER METHODS
    # These allow controlled access to private variables
    # ============================================================
    
    def get_current_image(self) -> Optional[np.ndarray]:
        """Get the currently edited image."""
        return self._current_image
    
    def get_original_image(self) -> Optional[np.ndarray]:
        """Get the original unedited image."""
        return self._original_image
    
    def reset_to_original(self):
        """Reset the current image back to the original."""
        if self._original_image is not None:
            self._current_image = self._original_image.copy()
    
    def get_dimensions(self) -> tuple:
        """
        Get image dimensions.
        
        Returns:
            Tuple of (height, width)
        """
        if self._current_image is not None:
            return self._current_image.shape[:2]
        return (0, 0)
    
    def get_filename(self) -> str:
        """Get the current filename."""
        return self._filename

    # ============================================================
    # IMAGE PROCESSING METHODS
    # These methods apply various effects and transformations
    # ============================================================
    
    def convert_to_grayscale(self):
        """Convert the image to grayscale (black and white)."""
        if self._current_image is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR (3 channels) for consistency
            self._current_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def apply_blur(self, intensity: int = 5):
        """
        Apply Gaussian blur to smooth the image.
        
        Args:
            intensity: Blur strength (1-10)
        """
        if self._current_image is not None:
            # Kernel size must be odd, so we use (intensity * 2 + 1)
            kernel_size = max(1, intensity * 2 + 1)
            self._current_image = cv2.GaussianBlur(
                self._current_image, 
                (kernel_size, kernel_size), 
                0
            )
    
    def detect_edges(self, threshold1: int = 100, threshold2: int = 200):
        """
        Apply Canny edge detection to find edges in the image.
        
        Args:
            threshold1: Lower threshold for edge detection
            threshold2: Upper threshold for edge detection
        """
        if self._current_image is not None:
            # Convert to grayscale first
            gray = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2GRAY)
            # Apply Canny edge detection
            edges = cv2.Canny(gray, threshold1, threshold2)
            # Convert back to BGR
            self._current_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    def adjust_brightness(self, value: int):
        """
        Adjust image brightness.
        
        Args:
            value: Brightness change (-100 to +100)
        """
        if self._current_image is not None:
            # Convert to HSV color space (Hue, Saturation, Value)
            hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Add brightness value to the V channel
            v = cv2.add(v, value)
            v = np.clip(v, 0, 255)  # Keep values in valid range
            
            # Merge back and convert to BGR
            final_hsv = cv2.merge((h, s, v))
            self._current_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_contrast(self, value: float):
        """
        Adjust image contrast.
        
        Args:
            value: Contrast multiplier (0.5 to 3.0)
        """
        if self._current_image is not None:
            # Alpha controls contrast, beta controls brightness (we use 0 for beta)
            self._current_image = cv2.convertScaleAbs(
                self._current_image, 
                alpha=value, 
                beta=0
            )
    
    def adjust_color_balance(self, red: int, green: int, blue: int):
        """
        Adjust RGB color channels individually.
        
        Args:
            red: Red channel adjustment (-100 to +100)
            green: Green channel adjustment (-100 to +100)
            blue: Blue channel adjustment (-100 to +100)
        """
        if self._current_image is not None:
            # Split into color channels (BGR order in OpenCV)
            b, g, r = cv2.split(self._current_image)
            
            # Adjust each channel
            r = cv2.add(r, red)
            g = cv2.add(g, green)
            b = cv2.add(b, blue)
            
            # Clip values to valid range (0-255)
            r = np.clip(r, 0, 255)
            g = np.clip(g, 0, 255)
            b = np.clip(b, 0, 255)
            
            # Merge channels back together
            self._current_image = cv2.merge([b, g, r])
    
    def apply_color_tint(self, color_rgb: tuple):
        """
        Apply a color tint overlay to the image.
        
        Args:
            color_rgb: RGB tuple (r, g, b) with values 0-255
        """
        if self._current_image is not None:
            # Create a colored overlay (convert RGB to BGR)
            overlay = np.full_like(self._current_image, color_rgb[::-1])
            # Blend original with overlay (70% original, 30% color)
            self._current_image = cv2.addWeighted(
                self._current_image, 0.7, 
                overlay, 0.3, 
                0
            )
    
    def adjust_saturation(self, value: float):
        """
        Adjust color saturation (intensity of colors).
        
        Args:
            value: Saturation multiplier (0.0 to 2.0)
        """
        if self._current_image is not None:
            # Convert to HSV
            hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            
            # Multiply saturation
            s = s * value
            s = np.clip(s, 0, 255)
            
            # Merge and convert back
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            self._current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def adjust_hue(self, value: int):
        """
        Shift the hue (color tone).
        
        Args:
            value: Hue shift (-180 to +180 degrees)
        """
        if self._current_image is not None:
            hsv = cv2.cvtColor(self._current_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            
            # Adjust hue with wrapping
            h = h + value
            h = np.where(h > 180, h - 180, h)
            h = np.where(h < 0, h + 180, h)
            
            # Merge and convert back
            hsv = cv2.merge([h, s, v]).astype(np.uint8)
            self._current_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def rotate_image(self, angle: int):
        """
        Rotate the image by 90, 180, or 270 degrees.
        
        Args:
            angle: Rotation angle (90, 180, or 270)
        """
        if self._current_image is not None:
            if angle == 90:
                self._current_image = cv2.rotate(
                    self._current_image, 
                    cv2.ROTATE_90_CLOCKWISE
                )
            elif angle == 180:
                self._current_image = cv2.rotate(
                    self._current_image, 
                    cv2.ROTATE_180
                )
            elif angle == 270:
                self._current_image = cv2.rotate(
                    self._current_image, 
                    cv2.ROTATE_90_COUNTERCLOCKWISE
                )
    
    def flip_image(self, direction: str):
        """
        Flip the image horizontally or vertically.
        
        Args:
            direction: "horizontal" or "vertical"
        """
        if self._current_image is not None:
            if direction == "horizontal":
                self._current_image = cv2.flip(self._current_image, 1)
            elif direction == "vertical":
                self._current_image = cv2.flip(self._current_image, 0)
    
    def resize_image(self, scale: float):
        """
        Resize the image by a scale factor.
        
        Args:
            scale: Scale multiplier (0.1 to 3.0)
        """
        if self._current_image is not None:
            height, width = self._current_image.shape[:2]
            new_width = int(width * scale)
            new_height = int(height * scale)
            self._current_image = cv2.resize(
                self._current_image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_LINEAR
            )


# ============================================================
# HistoryManager Class
# ============================================================
# Manages undo/redo functionality using two stacks
# Demonstrates the Stack data structure concept
# ============================================================

class HistoryManager:
    """
    Manages undo/redo operations using history stacks.
    Uses a circular buffer to limit memory usage.
    """
    
    def __init__(self, max_history: int = 20):
        """
        Initialize history manager.
        
        Args:
            max_history: Maximum number of states to remember
        """
        self._history: List[np.ndarray] = []  # Stack of previous states
        self._redo_stack: List[np.ndarray] = []  # Stack of undone states
        self._max_history = max_history  # Memory limit
    
    def save_state(self, image: np.ndarray):
        """
        Save current state to history before making changes.
        
        Args:
            image: Current image to save
        """
        if image is not None:
            self._history.append(image.copy())
            
            # Limit history size (remove oldest if too many)
            if len(self._history) > self._max_history:
                self._history.pop(0)
            
            # Clear redo stack (can't redo after new action)
            self._redo_stack.clear()
    
    def undo(self) -> Optional[np.ndarray]:
        """
        Undo to previous state.
        
        Returns:
            Previous image state, or None if can't undo
        """
        if len(self._history) > 1:
            # Move current to redo stack
            current = self._history.pop()
            self._redo_stack.append(current)
            # Return previous state
            return self._history[-1].copy()
        return None
    
    def redo(self) -> Optional[np.ndarray]:
        """
        Redo previously undone action.
        
        Returns:
            Next image state, or None if can't redo
        """
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
        """Clear all history (used when loading new image)."""
        self._history.clear()
        self._redo_stack.clear()


# ============================================================
# ImageProcessorGUI Class - MAIN APPLICATION
# ============================================================
# This is the main GUI class that brings everything together
# Features HOVER-BASED navigation (no clicking needed!)
# ============================================================

class ImageProcessorGUI:
    """
    Main GUI application with hover-based navigation.
    
    Key Features:
    - Hover over categories to see tools (no clicking!)
    - Professional logo and branding
    - Before/After comparison view
    - Video frame extraction
    - Full color palette
    """
    
    def __init__(self, root: tk.Tk):
        """
        Initialize the main application window.
        
        Args:
            root: The main Tkinter window
        """
        self.root = root
        self.root.title("Image Processor Pro - DAN/EXT 01")
        self.root.geometry("1450x850")  # Window size
        
        # Initialize the image processor and history manager
        self.processor = ImageProcessor()
        self.history = HistoryManager()
        
        # Track currently active category
        self.active_category = "file"
        
        # Define color scheme for the entire application
        self.colors = {
            'bg_dark': '#2c3e50',        # Main dark background
            'bg_darker': '#1a252f',      # Darker sections
            'accent': '#3498db',         # Blue accent color
            'accent_hover': '#5dade2',   # Lighter blue for hover
            'text_light': '#ecf0f1',     # Light text
            'border': '#34495e',         # Border color
            'canvas_bg': '#1e1e1e',      # Canvas background
            'logo_bg': '#16213e'         # Logo section background
        }
        
        # Build the GUI
        self._create_menu_bar()
        self._create_logo_section()
        self._create_main_layout()
        self._create_status_bar()
        
        # Show initial placeholder
        self._update_display()
    
    # ============================================================
    # GUI CREATION METHODS
    # These methods build the user interface
    # ============================================================
    
    def _create_logo_section(self):
        """
        Create the top logo and branding section.
        Shows professional logo, app name, and group info.
        """
        # Create frame for logo section
        logo_frame = tk.Frame(self.root, bg=self.colors['logo_bg'], height=80)
        logo_frame.pack(fill=tk.X, side=tk.TOP)
        logo_frame.pack_propagate(False)  # Prevent shrinking
        
        # Logo canvas (for drawing custom logo)
        logo_canvas = tk.Canvas(
            logo_frame, 
            bg=self.colors['logo_bg'], 
            width=70, 
            height=70, 
            highlightthickness=0
        )
        logo_canvas.pack(side=tk.LEFT, padx=20, pady=5)
        
        # Draw the logo
        self._draw_logo(logo_canvas)
        
        # Text frame for branding
        text_frame = tk.Frame(logo_frame, bg=self.colors['logo_bg'])
        text_frame.pack(side=tk.LEFT, fill=tk.Y, pady=10)
        
        # Application title
        title = tk.Label(
            text_frame,
            text="IMAGE PROCESSOR PRO",
            bg=self.colors['logo_bg'],
            fg='#ffffff',
            font=('Arial', 20, 'bold')
        )
        title.pack(anchor=tk.W)
        
        # Subtitle/tagline
        subtitle = tk.Label(
            text_frame,
            text="Professional Image & Video Editing Suite",
            bg=self.colors['logo_bg'],
            fg=self.colors['accent'],
            font=('Arial', 10)
        )
        subtitle.pack(anchor=tk.W)
        
        # Group name badge
        group_label = tk.Label(
            logo_frame,
            text="DAN/EXT 01",
            bg=self.colors['logo_bg'],
            fg=self.colors['text_light'],
            font=('Arial', 10, 'bold')
        )
        group_label.pack(side=tk.RIGHT, padx=20)
    
    def _draw_logo(self, canvas):
        """
        Draw a custom camera-style logo on the canvas.
        
        Args:
            canvas: The canvas widget to draw on
        """
        # Color gradient for professional look
        colors_gradient = ['#3498db', '#2980b9', '#21618c', '#1b4f72']
        
        # Outer circle
        canvas.create_oval(
            10, 10, 60, 60, 
            fill=colors_gradient[0], 
            outline=colors_gradient[3], 
            width=2
        )
        
        # Inner hexagon (camera aperture style)
        center = 35
        points = []
        for i in range(6):
            angle = i * 60 - 30  # 6 sides at 60° intervals
            x = center + 15 * np.cos(np.radians(angle))
            y = center + 15 * np.sin(np.radians(angle))
            points.extend([x, y])
        
        canvas.create_polygon(
            points, 
            fill=colors_gradient[3], 
            outline='white', 
            width=1
        )
        
        # Center dot
        canvas.create_oval(30, 30, 40, 40, fill='white', outline='')
        
        # Decorative arcs
        canvas.create_arc(
            5, 5, 65, 65, 
            start=45, extent=90, 
            outline='#ecf0f1', 
            width=2, 
            style=tk.ARC
        )
        canvas.create_arc(
            5, 5, 65, 65, 
            start=225, extent=90, 
            outline='#ecf0f1', 
            width=2, 
            style=tk.ARC
        )
    
    def _create_menu_bar(self):
        """
        Create the top menu bar with File and Edit menus.
        Includes keyboard shortcuts for power users.
        """
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        
        # ===== FILE MENU =====
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Open Image", 
            command=self._open_image, 
            accelerator="Ctrl+O"
        )
        file_menu.add_command(
            label="Capture from Camera", 
            command=self._capture_from_camera, 
            accelerator="Ctrl+C"
        )
        file_menu.add_command(
            label="Extract from Video", 
            command=self._extract_from_video, 
            accelerator="Ctrl+V"
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Save", 
            command=self._save_image, 
            accelerator="Ctrl+S"
        )
        file_menu.add_command(
            label="Save As", 
            command=self._save_as_image
        )
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_app)
        
        # ===== EDIT MENU =====
        edit_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(
            label="Undo", 
            command=self._undo, 
            accelerator="Ctrl+Z"
        )
        edit_menu.add_command(
            label="Redo", 
            command=self._redo, 
            accelerator="Ctrl+Y"
        )
        edit_menu.add_separator()
        edit_menu.add_command(
            label="Reset to Original", 
            command=self._reset_image
        )
        
        # ===== KEYBOARD SHORTCUTS =====
        # Bind keyboard shortcuts to methods
        self.root.bind('<Control-o>', lambda e: self._open_image())
        self.root.bind('<Control-c>', lambda e: self._capture_from_camera())
        self.root.bind('<Control-v>', lambda e: self._extract_from_video())
        self.root.bind('<Control-s>', lambda e: self._save_image())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Control-y>', lambda e: self._redo())
    
    def _create_main_layout(self):
        """
        Create the main 3-column layout:
        LEFT: Category buttons
        MIDDLE: Tool panel (changes on hover)
        RIGHT: Workspace (before/after comparison)
        """
        # Main container frame
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # LEFT: Category sidebar
        self._create_category_sidebar(main_frame)
        
        # MIDDLE: Tools panel (dynamic, changes on hover)
        self.tools_container = tk.Frame(
            main_frame, 
            bg=self.colors['bg_darker'], 
            width=280
        )
        self.tools_container.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        self.tools_container.pack_propagate(False)  # Fixed width
        
        # Thin separator line (makes boundary clear, no gap)
        separator = tk.Frame(main_frame, bg=self.colors['border'], width=1)
        separator.pack(side=tk.LEFT, fill=tk.Y)
        
        # RIGHT: Workspace area (touches separator directly)
        self._create_comparison_area(main_frame)
        
        # Load default category (File)
        self._load_category_tools("file")
    
    def _create_category_sidebar(self, parent):
        """
        Create the left sidebar with category buttons.
        HOVER FEATURE: Categories show tools on hover, no clicking needed!
        
        Args:
            parent: The parent frame to place this sidebar in
        """
        # Create sidebar frame
        sidebar = tk.Frame(parent, bg=self.colors['bg_darker'], width=180)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)  # Fixed width
        
        # Header label
        header = tk.Label(
            sidebar,
            text="TOOLS",
            bg=self.colors['bg_darker'],
            fg=self.colors['text_light'],
            font=('Arial', 12, 'bold'),
            pady=15
        )
        header.pack(fill=tk.X)
        
        # Define categories (File is first!)
        categories = [
            ("📁 File", "file"),
            ("🎨 Filters", "filters"),
            ("🎛️ Adjustments", "adjustments"),
            ("🌈 Colors", "colors"),
            ("🔧 Transform", "transform")
        ]
        
        self.category_buttons = {}  # Store buttons for later access
        
        # Create each category button
        for label, cat_id in categories:
            btn = tk.Button(
                sidebar,
                text=label,
                bg=self.colors['border'],
                fg=self.colors['text_light'],
                font=('Arial', 11),
                relief=tk.FLAT,
                cursor='hand2',
                anchor=tk.W,
                padx=20,
                pady=12
            )
            btn.pack(fill=tk.X, padx=10, pady=2)
            self.category_buttons[cat_id] = btn
            
            # ===== HOVER BEHAVIOR =====
            # When mouse enters button, load its tools and show arrow
            def on_enter(e, category=cat_id, button=btn, lbl=label):
                """Called when mouse hovers over category button."""
                # Load this category's tools
                self._load_category_tools(category)
                # Show arrow indicator
                button.config(text=lbl + " ›", bg=self.colors['accent_hover'])
            
            def on_leave(e, category=cat_id, button=btn, lbl=label):
                """Called when mouse leaves category button."""
                # If this is the active category, keep it highlighted
                if self.active_category == category:
                    button.config(text=lbl + " ›", bg=self.colors['accent'])
                else:
                    button.config(text=lbl, bg=self.colors['border'])
            
            # Bind hover events
            btn.bind("<Enter>", on_enter)
            btn.bind("<Leave>", on_leave)
    
    def _load_category_tools(self, category):
        """
        Load tools for the selected category.
        This is called automatically when hovering over a category!
        
        Args:
            category: Category ID (file, filters, adjustments, colors, transform)
        """
        # Update which category is active
        self.active_category = category
        
        # Update all button states
        label_map = {
            'file': "📁 File",
            'filters': "🎨 Filters",
            'adjustments': "🎛️ Adjustments",
            'colors': "🌈 Colors",
            'transform': "🔧 Transform"
        }
        
        for cat_id, btn in self.category_buttons.items():
            original_label = label_map.get(cat_id, "")
            if cat_id == category:
                # Active category: blue background with arrow
                btn.config(bg=self.colors['accent'], text=original_label + " ›")
            else:
                # Inactive: grey background, no arrow
                btn.config(bg=self.colors['border'], text=original_label)
        
        # Clear existing tools
        for widget in self.tools_container.winfo_children():
            widget.destroy()
        
        # Create scrollable frame for tools
        canvas = tk.Canvas(
            self.tools_container, 
            bg=self.colors['bg_darker'], 
            highlightthickness=0
        )
        scrollbar = tk.Scrollbar(
            self.tools_container, 
            orient=tk.VERTICAL, 
            command=canvas.yview
        )
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_darker'])
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind('<Enter>', lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all("<MouseWheel>"))
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load appropriate tools based on category
        if category == "file":
            self._add_file_tools(scrollable_frame)
        elif category == "filters":
            self._add_filters_tools(scrollable_frame)
        elif category == "adjustments":
            self._add_adjustment_tools(scrollable_frame)
        elif category == "colors":
            self._add_color_tools(scrollable_frame)
        elif category == "transform":
            self._add_transform_tools(scrollable_frame)
    
    # ============================================================
    # HELPER METHODS FOR CREATING UI ELEMENTS
    # ============================================================
    
    def _create_section_header(self, parent, title):
        """
        Create a styled section header with blue accent.
        
        Args:
            parent: Parent widget
            title: Header text
        """
        header = tk.Label(
            parent,
            text=title,
            bg=self.colors['bg_darker'],
            fg=self.colors['accent'],
            font=('Arial', 10, 'bold'),
            anchor=tk.W,
            pady=8
        )
        header.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        # Blue separator line
        sep = tk.Frame(parent, bg=self.colors['accent'], height=2)
        sep.pack(fill=tk.X, padx=15, pady=(0, 10))
    
    def _create_tool_button(self, parent, text, command):
        """
        Create a styled button for tools.
        
        Args:
            parent: Parent widget
            text: Button text
            command: Function to call when clicked
            
        Returns:
            The created button widget
        """
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=self.colors['border'],
            fg=self.colors['text_light'],
            font=('Arial', 9),
            relief=tk.FLAT,
            cursor='hand2',
            padx=10,
            pady=8
        )
        btn.pack(fill=tk.X, padx=15, pady=3)
        
        # Hover effect
        btn.bind("<Enter>", lambda e: btn.config(bg=self.colors['accent']))
        btn.bind("<Leave>", lambda e: btn.config(bg=self.colors['border']))
        
        return btn
    
    def _create_slider(self, parent, label, from_, to, variable, resolution=1):
        """
        Create a slider control with label.
        
        Args:
            parent: Parent widget
            label: Label text
            from_: Minimum value
            to: Maximum value
            variable: Tkinter variable to bind to
            resolution: Step size
            
        Returns:
            The created slider widget
        """
        # Label
        label_widget = tk.Label(
            parent,
            text=label,
            bg=self.colors['bg_darker'],
            fg=self.colors['text_light'],
            font=('Arial', 8)
        )
        label_widget.pack(anchor=tk.W, padx=15, pady=(5, 2))
        
        # Slider
        slider = tk.Scale(
            parent,
            from_=from_,
            to=to,
            variable=variable,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            bg=self.colors['bg_darker'],
            fg=self.colors['text_light'],
            troughcolor=self.colors['bg_dark'],
            highlightthickness=0,
            font=('Arial', 8)
        )
        slider.pack(fill=tk.X, padx=15, pady=(0, 5))
        
        return slider
    
    # ============================================================
    # TOOL PANEL SECTIONS
    # These methods create the tools for each category
    # ============================================================
    
    def _add_file_tools(self, parent):
        """
        Create tools for the FILE category.
        Includes load, save, and history options.
        
        Args:
            parent: Parent frame to add tools to
        """
        # LOAD section
        self._create_section_header(parent, "LOAD")
        self._create_tool_button(parent, "📁 Open Image File", self._open_image)
        self._create_tool_button(parent, "📷 Capture from Camera", self._capture_from_camera)
        self._create_tool_button(parent, "🎬 Extract from Video", self._extract_from_video)
        
        # SAVE section
        self._create_section_header(parent, "SAVE")
        self._create_tool_button(parent, "💾 Save", self._save_image)
        self._create_tool_button(parent, "💾 Save As", self._save_as_image)
        
        # HISTORY section
        self._create_section_header(parent, "HISTORY")
        self._create_tool_button(parent, "↶ Undo", self._undo)
        self._create_tool_button(parent, "↷ Redo", self._redo)
        self._create_tool_button(parent, "🔄 Reset to Original", self._reset_image)
    
    def _add_filters_tools(self, parent):
        """
        Create tools for the FILTERS category.
        
        Args:
            parent: Parent frame
        """
        self._create_section_header(parent, "BASIC FILTERS")
        self._create_tool_button(parent, "Grayscale", self._apply_grayscale)
        self._create_tool_button(parent, "Edge Detection", self._apply_edge_detection)
        
        self._create_section_header(parent, "BLUR EFFECT")
        self.blur_var = tk.IntVar(value=3)
        self._create_slider(parent, "Intensity:", 1, 10, self.blur_var)
        self._create_tool_button(parent, "Apply Blur", self._apply_blur)
    
    def _add_adjustment_tools(self, parent):
        """
        Create tools for the ADJUSTMENTS category.
        
        Args:
            parent: Parent frame
        """
        # Brightness
        self._create_section_header(parent, "BRIGHTNESS")
        self.brightness_var = tk.IntVar(value=0)
        self._create_slider(parent, "Level (-100 to 100):", -100, 100, self.brightness_var)
        self._create_tool_button(parent, "Apply Brightness", self._apply_brightness)
        
        # Contrast
        self._create_section_header(parent, "CONTRAST")
        self.contrast_var = tk.DoubleVar(value=1.0)
        self._create_slider(parent, "Level (0.5 to 3.0):", 0.5, 3.0, self.contrast_var, 0.1)
        self._create_tool_button(parent, "Apply Contrast", self._apply_contrast)
        
        # Saturation
        self._create_section_header(parent, "SATURATION")
        self.saturation_var = tk.DoubleVar(value=1.0)
        self._create_slider(parent, "Level (0.0 to 2.0):", 0.0, 2.0, self.saturation_var, 0.1)
        self._create_tool_button(parent, "Apply Saturation", self._apply_saturation)
        
        # Hue
        self._create_section_header(parent, "HUE SHIFT")
        self.hue_var = tk.IntVar(value=0)
        self._create_slider(parent, "Shift (-180 to 180):", -180, 180, self.hue_var)
        self._create_tool_button(parent, "Apply Hue Shift", self._apply_hue)
    
    def _add_color_tools(self, parent):
        """
        Create tools for the COLORS category.
        Includes color palette and RGB sliders.
        
        Args:
            parent: Parent frame
        """
        # Color palette section
        self._create_section_header(parent, "COLOR PALETTE")
        
        # Grid of color boxes
        palette_frame = tk.Frame(parent, bg=self.colors['bg_darker'])
        palette_frame.pack(fill=tk.X, padx=15, pady=10)
        
        # 16 preset colors
        colors = [
            '#FF6B6B', '#FF8E53', '#FFC75F', '#F9F871',
            '#51CF66', '#4ECDC4', '#4DABF7', '#748FFC',
            '#9775FA', '#DA77F2', '#F783AC', '#FF6B9D',
            '#000000', '#495057', '#ADB5BD', '#FFFFFF'
        ]
        
        # Create 4x4 grid
        row = 0
        col = 0
        for color in colors:
            btn = tk.Button(
                palette_frame,
                bg=color,
                width=3,
                height=1,
                relief=tk.RAISED,
                cursor='hand2',
                command=lambda c=color: self._apply_palette_color(c)
            )
            btn.grid(row=row, column=col, padx=2, pady=2)
            
            col += 1
            if col >= 4:
                col = 0
                row += 1
        
        # Custom color picker
        self._create_tool_button(parent, "🎨 Pick Custom Color", self._pick_custom_color)
        
        # RGB Balance section
        self._create_section_header(parent, "RGB BALANCE")
        
        self.red_var = tk.IntVar(value=0)
        self.green_var = tk.IntVar(value=0)
        self.blue_var = tk.IntVar(value=0)
        
        # Red channel
        red_label = tk.Label(
            parent, 
            text="Red (-100 to 100):", 
            bg=self.colors['bg_darker'],
            fg='#ff6b6b',  # Red color
            font=('Arial', 8, 'bold')
        )
        red_label.pack(anchor=tk.W, padx=15, pady=(5, 2))
        self._create_slider(parent, "", -100, 100, self.red_var)
        
        # Green channel
        green_label = tk.Label(
            parent, 
            text="Green (-100 to 100):", 
            bg=self.colors['bg_darker'],
            fg='#51cf66',  # Green color
            font=('Arial', 8, 'bold')
        )
        green_label.pack(anchor=tk.W, padx=15, pady=(5, 2))
        self._create_slider(parent, "", -100, 100, self.green_var)
        
        # Blue channel
        blue_label = tk.Label(
            parent, 
            text="Blue (-100 to 100):", 
            bg=self.colors['bg_darker'],
            fg='#4dabf7',  # Blue color
            font=('Arial', 8, 'bold')
        )
        blue_label.pack(anchor=tk.W, padx=15, pady=(5, 2))
        self._create_slider(parent, "", -100, 100, self.blue_var)
        
        self._create_tool_button(parent, "Apply RGB Balance", self._apply_color_balance)
        self._create_tool_button(parent, "Reset", self._reset_color_sliders)
    
    def _add_transform_tools(self, parent):
        """
        Create tools for the TRANSFORM category.
        
        Args:
            parent: Parent frame
        """
        # Rotation
        self._create_section_header(parent, "ROTATION")
        self._create_tool_button(parent, "Rotate 90°", lambda: self._apply_rotation(90))
        self._create_tool_button(parent, "Rotate 180°", lambda: self._apply_rotation(180))
        self._create_tool_button(parent, "Rotate 270°", lambda: self._apply_rotation(270))
        
        # Flip
        self._create_section_header(parent, "FLIP")
        self._create_tool_button(parent, "Flip Horizontal", lambda: self._apply_flip("horizontal"))
        self._create_tool_button(parent, "Flip Vertical", lambda: self._apply_flip("vertical"))
        
        # Resize
        self._create_section_header(parent, "RESIZE")
        self.scale_var = tk.DoubleVar(value=1.0)
        self._create_slider(parent, "Scale (0.1 to 3.0):", 0.1, 3.0, self.scale_var, 0.1)
        self._create_tool_button(parent, "Apply Resize", self._apply_resize)
    
    def _create_comparison_area(self, parent):
        """
        Create the workspace area with before/after comparison.
        ZERO PADDING - workspace touches the tools panel directly!
        
        Args:
            parent: Parent frame
        """
        # Main comparison frame - ABSOLUTELY NO PADDING!
        comparison_frame = tk.Frame(parent, bg=self.colors['canvas_bg'])
        comparison_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Top toolbar
        toolbar = tk.Frame(comparison_frame, bg=self.colors['bg_darker'], height=50)
        toolbar.pack(fill=tk.X, padx=0, pady=0)
        toolbar.pack_propagate(False)
        
        # Toolbar title
        title = tk.Label(
            toolbar,
            text="WORKSPACE",
            bg=self.colors['bg_darker'],
            fg=self.colors['text_light'],
            font=('Arial', 11, 'bold')
        )
        title.pack(side=tk.LEFT, padx=15)
        
        # Before/After panels with minimal padding
        panels_frame = tk.Frame(comparison_frame, bg=self.colors['canvas_bg'])
        panels_frame.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)
        
        # BEFORE panel
        before_frame = tk.Frame(panels_frame, bg=self.colors['bg_darker'])
        before_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        
        before_label = tk.Label(
            before_frame,
            text="BEFORE",
            bg=self.colors['bg_darker'],
            fg=self.colors['text_light'],
            font=('Arial', 12, 'bold')
        )
        before_label.pack(pady=8)
        
        self.canvas_before = tk.Canvas(
            before_frame,
            bg=self.colors['canvas_bg'],
            highlightthickness=1,
            highlightbackground=self.colors['border']
        )
        self.canvas_before.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        # AFTER panel
        after_frame = tk.Frame(panels_frame, bg=self.colors['bg_darker'])
        after_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 0))
        
        after_label = tk.Label(
            after_frame,
            text="AFTER",
            bg=self.colors['bg_darker'],
            fg=self.colors['accent'],
            font=('Arial', 12, 'bold')
        )
        after_label.pack(pady=8)
        
        self.canvas_after = tk.Canvas(
            after_frame,
            bg=self.colors['canvas_bg'],
            highlightthickness=1,
            highlightbackground=self.colors['accent']
        )
        self.canvas_after.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
    
    def _create_status_bar(self):
        """Create the bottom status bar."""
        self.status_bar = tk.Label(
            self.root,
            text="No image loaded",
            relief=tk.FLAT,
            anchor=tk.W,
            bg=self.colors['bg_darker'],
            fg=self.colors['text_light'],
            font=('Arial', 9),
            padx=10,
            pady=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    # ============================================================
    # DISPLAY UPDATE METHODS
    # ============================================================
    
    def _update_status_bar(self):
        """Update status bar with current image info."""
        if self.processor.get_current_image() is not None:
            height, width = self.processor.get_dimensions()
            filename = self.processor.get_filename().split('/')[-1]
            if not filename:
                filename = self.processor.get_filename().split('\\')[-1]
            if not filename:
                filename = "Untitled"
            self.status_bar.config(
                text=f"📄 {filename} | 📐 {width}x{height}px | ✅ Ready"
            )
        else:
            self.status_bar.config(
                text="No image loaded | Click File > Open to load image"
            )
    
    def _update_display(self):
        """Update both before and after canvases."""
        original = self.processor.get_original_image()
        current = self.processor.get_current_image()
        
        if original is not None:
            self._display_image_on_canvas(self.canvas_before, original)
        else:
            self._show_placeholder(
                self.canvas_before, 
                "No image loaded\nUse File > Open"
            )
        
        if current is not None:
            self._display_image_on_canvas(self.canvas_after, current)
        else:
            self._show_placeholder(
                self.canvas_after, 
                "No image loaded\nUse File > Open"
            )
        
        self._update_status_bar()
    
    def _display_image_on_canvas(self, canvas, img):
        """
        Display an image on canvas, scaled to fit.
        
        Args:
            canvas: Canvas widget to draw on
            img: OpenCV image (numpy array)
        """
        # Convert BGR to RGB for display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get canvas size
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Calculate scale to fit
        img_height, img_width = img.shape[:2]
        scale_w = (canvas_width - 20) / img_width
        scale_h = (canvas_height - 20) / img_height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        # Resize if needed
        if scale < 1.0:
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            img_rgb = cv2.resize(
                img_rgb, 
                (new_width, new_height), 
                interpolation=cv2.INTER_AREA
            )
        
        # Convert to PhotoImage
        pil_img = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_img)
        
        # Display centered
        canvas.delete("all")
        x_center = canvas_width // 2
        y_center = canvas_height // 2
        canvas.create_image(x_center, y_center, anchor=tk.CENTER, image=photo)
        
        # Keep reference to prevent garbage collection
        canvas.image = photo
    
    def _show_placeholder(self, canvas, text):
        """
        Show placeholder text on canvas.
        
        Args:
            canvas: Canvas widget
            text: Text to display
        """
        canvas.delete("all")
        canvas.update_idletasks()
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        canvas.create_text(
            canvas_width // 2 if canvas_width > 1 else 300,
            canvas_height // 2 if canvas_height > 1 else 250,
            text=text,
            font=("Arial", 14),
            fill=self.colors['text_light'],
            justify=tk.CENTER
        )
    
    def _save_to_history(self):
        """Save current state before making changes."""
        img = self.processor.get_current_image()
        if img is not None:
            self.history.save_state(img)
    
    # ============================================================
    # FILE OPERATIONS
    # ============================================================
    
    def _open_image(self):
        """Open image file dialog and load selected image."""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Open Image", 
            filetypes=filetypes
        )
        
        if filepath:
            if self.processor.load_image(filepath):
                self.history.clear()
                self._save_to_history()
                self._update_display()
                messagebox.showinfo("Success", "Image loaded successfully!")
            else:
                messagebox.showerror("Error", "Failed to load image.")
    
    def _capture_from_camera(self):
        """Open camera and capture a photo."""
        # Create camera window
        camera_window = tk.Toplevel(self.root)
        camera_window.title("Camera Capture")
        camera_window.geometry("800x650")
        camera_window.configure(bg=self.colors['bg_dark'])
        
        # Preview area
        preview_label = tk.Label(camera_window, bg=self.colors['canvas_bg'])
        preview_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Button frame
        button_frame = tk.Frame(camera_window, bg=self.colors['bg_dark'])
        button_frame.pack(pady=15)
        
        # Open camera (0 is default camera)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera!")
            camera_window.destroy()
            return
        
        captured_image = None
        is_running = True
        
        def update_camera():
            """Update camera preview continuously."""
            if is_running:
                ret, frame = cap.read()
                if ret:
                    # Convert and resize for preview
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
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
                
                # Load into processor
                self.processor.load_image_from_array(
                    captured_image, 
                    "captured_image.jpg"
                )
                self.history.clear()
                self._save_to_history()
                self._update_display()
                
                camera_window.destroy()
                messagebox.showinfo("Success", "Image captured!")
        
        def cancel():
            """Cancel and close camera."""
            nonlocal is_running
            is_running = False
            cap.release()
            camera_window.destroy()
        
        # Capture button
        capture_btn = tk.Button(
            button_frame,
            text="📷 Capture",
            command=capture,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=25,
            pady=10
        )
        capture_btn.pack(side=tk.LEFT, padx=10)
        
        # Cancel button
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=cancel,
            bg=self.colors['border'],
            fg=self.colors['text_light'],
            font=('Arial', 11),
            relief=tk.FLAT,
            cursor='hand2',
            padx=25,
            pady=10
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Info label
        info_label = tk.Label(
            camera_window,
            text="Position yourself and click 'Capture'",
            font=("Arial", 10),
            bg=self.colors['bg_dark'],
            fg=self.colors['text_light']
        )
        info_label.pack(pady=5)
        
        # Start camera preview
        update_camera()
        
        # Handle window close
        camera_window.protocol("WM_DELETE_WINDOW", cancel)
    
    def _extract_from_video(self):
        """
        Extract a frame from a video file.
        Opens a video, lets user scrub through, and extract any frame.
        """
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(
            title="Select Video File", 
            filetypes=filetypes
        )
        
        if not filepath:
            return
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file!")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video window
        video_window = tk.Toplevel(self.root)
        video_window.title("Extract Frame from Video")
        video_window.geometry("900x700")
        video_window.configure(bg=self.colors['bg_dark'])
        
        # Video preview
        preview_label = tk.Label(video_window, bg=self.colors['canvas_bg'])
        preview_label.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Controls
        controls_frame = tk.Frame(video_window, bg=self.colors['bg_dark'])
        controls_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Frame slider
        current_frame = tk.IntVar(value=0)
        
        info_label = tk.Label(
            controls_frame,
            text=f"Frame: 0 / {total_frames} | Time: 0.00s",
            bg=self.colors['bg_dark'],
            fg=self.colors['text_light'],
            font=('Arial', 10)
        )
        info_label.pack(pady=5)
        
        def update_frame(val=None):
            """Update preview when slider moves."""
            frame_num = current_frame.get()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                # Display frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (640, 480))
                img = Image.fromarray(frame_resized)
                photo = ImageTk.PhotoImage(img)
                preview_label.config(image=photo)
                preview_label.image = photo
                
                # Update info
                time_sec = frame_num / fps if fps > 0 else 0
                info_label.config(
                    text=f"Frame: {frame_num} / {total_frames} | Time: {time_sec:.2f}s"
                )
        
        # Slider
        slider = tk.Scale(
            controls_frame,
            from_=0,
            to=total_frames-1,
            variable=current_frame,
            orient=tk.HORIZONTAL,
            command=update_frame,
            bg=self.colors['bg_dark'],
            fg=self.colors['text_light'],
            troughcolor=self.colors['border'],
            highlightthickness=0,
            length=700
        )
        slider.pack(pady=5)
        
        # Buttons
        button_frame = tk.Frame(video_window, bg=self.colors['bg_dark'])
        button_frame.pack(pady=10)
        
        def extract_frame():
            """Extract current frame."""
            frame_num = current_frame.get()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            
            if ret:
                cap.release()
                self.processor.load_image_from_array(
                    frame, 
                    f"video_frame_{frame_num}.jpg"
                )
                self.history.clear()
                self._save_to_history()
                self._update_display()
                video_window.destroy()
                messagebox.showinfo(
                    "Success", 
                    f"Frame {frame_num} extracted!"
                )
        
        extract_btn = tk.Button(
            button_frame,
            text="📸 Extract This Frame",
            command=extract_frame,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 11, 'bold'),
            relief=tk.FLAT,
            cursor='hand2',
            padx=25,
            pady=10
        )
        extract_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(
            button_frame,
            text="Cancel",
            command=lambda: [cap.release(), video_window.destroy()],
            bg=self.colors['border'],
            fg=self.colors['text_light'],
            font=('Arial', 11),
            relief=tk.FLAT,
            cursor='hand2',
            padx=25,
            pady=10
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Initial frame
        update_frame()
        
        # Handle close
        video_window.protocol(
            "WM_DELETE_WINDOW", 
            lambda: [cap.release(), video_window.destroy()]
        )
    
    def _save_image(self):
        """Save current image."""
        if self.processor.get_current_image() is None:
            messagebox.showwarning("Warning", "No image to save.")
            return
        
        filepath = self.processor.get_filename()
        
        # If temporary file, use Save As
        if not filepath or filepath in ["captured_image.jpg", "extracted_frame.jpg"] or filepath.startswith("video_frame_"):
            self._save_as_image()
            return
        
        if self.processor.save_image(filepath):
            messagebox.showinfo("Success", "Image saved!")
        else:
            messagebox.showerror("Error", "Failed to save.")
    
    def _save_as_image(self):
        """Save image with new filename."""
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
                messagebox.showinfo("Success", "Image saved!")
            else:
                messagebox.showerror("Error", "Failed to save.")
    
    def _exit_app(self):
        """Exit application."""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.quit()
    
    # ============================================================
    # EDIT OPERATIONS
    # ============================================================
    
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
        """Reset to original image."""
        if self.processor.get_original_image() is not None:
            self._save_to_history()
            self.processor.reset_to_original()
            self._update_display()
    
    # ============================================================
    # IMAGE PROCESSING OPERATIONS
    # ============================================================
    
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
        """Apply RGB color balance."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            self.processor.adjust_color_balance(
                self.red_var.get(),
                self.green_var.get(),
                self.blue_var.get()
            )
            self._update_display()
    
    def _apply_palette_color(self, hex_color):
        """Apply color from palette."""
        if self.processor.get_current_image() is not None:
            self._save_to_history()
            # Convert hex to RGB
            rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
            self.processor.apply_color_tint(rgb)
            self._update_display()
    
    def _pick_custom_color(self):
        """Pick custom color with color chooser."""
        if self.processor.get_current_image() is not None:
            color = colorchooser.askcolor(title="Choose Color")
            if color[0]:
                self._save_to_history()
                self.processor.apply_color_tint(color[0])
                self._update_display()
    
    def _reset_color_sliders(self):
        """Reset RGB sliders to 0."""
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


# ============================================================
# MAIN FUNCTION - Entry point of the application
# ============================================================

def main():
    """
    Main function to run the application.
    Creates the root window and starts the GUI.
    """
    root = tk.Tk()
    app = ImageProcessorGUI(root)
    root.mainloop()


# ============================================================
# Run the application if this file is executed directly
# ============================================================

if __name__ == "__main__":
    main()



