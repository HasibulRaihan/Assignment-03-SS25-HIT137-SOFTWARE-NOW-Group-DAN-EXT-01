# 🎨 Image Processor Pro

**Professional Image & Video Editing Suite**

A powerful, feature-rich desktop application for image processing and video frame extraction built with Python, OpenCV, and Tkinter. Designed for both beginners and professionals who need quick, efficient image editing without the complexity of professional software like Photoshop.

## 🌟 Key Features

### 🖼️ Image Processing
- **Basic Filters**: Grayscale, Edge Detection, Blur
- **Advanced Adjustments**: Brightness, Contrast, Saturation, Hue Shift
- **Color Manipulation**: 16 preset color palette + custom color picker + RGB channel balance
- **Transformations**: Rotate (90°, 180°, 270°), Flip (Horizontal/Vertical), Resize

### 📹 Video Capabilities
- **Frame Extraction**: Load video files and extract any frame as an image
- **Video Scrubbing**: Navigate through videos frame-by-frame with an intuitive slider
- **Format Support**: MP4, AVI, MOV, MKV, FLV

### 📷 Camera Integration
- **Live Capture**: Capture images directly from your webcam
- **Real-time Preview**: See yourself before capturing

### 🎯 User Experience
- **Hover-Based Navigation**: No clicking needed! Hover over categories to see tools instantly
- **Before/After Comparison**: See original and edited images side-by-side
- **Undo/Redo**: Full history management (up to 20 states)
- **Keyboard Shortcuts**: Power user features (Ctrl+O, Ctrl+S, Ctrl+Z, etc.)

---

## Screenshots

### Main Interface
```
┌─────────────────────────────────────────────────────────────────┐
│  IMAGE PROCESSOR PRO                           DAN/EXT 01        │
│  Professional Image & Video Editing Suite                        │
├─────────┬──────────────┬──────────────────────────────────────┤
│ TOOLS   │    LOAD      │           WORKSPACE                  │
│         │              │  ┌────────────┬────────────┐         │
│ File ›  │ Open Image   │  │   BEFORE   │   AFTER    │         │
│ Filters │ Camera       │  │            │            │         │
│ Adjust  │ Video        │  │  Original  │   Edited   │         │
│ Colors  │              │  │   Image    │   Image    │         │
│ Trans.  │ SAVE         │  │            │            │         │
│         │ Save/Save As │  └────────────┴────────────┘         │
│         │              │                                        │
│         │ HISTORY      │                                        │
│         │ Undo/Redo    │                                        │
└─────────┴──────────────┴──────────────────────────────────────┘
```

---
## Usage Guide

### Loading Images
- **From File**: Hover over "File" category → Click "Open Image File"
- **From Camera**: Hover over "File" → Click "Capture from Camera"
- **From Video**: Hover over "File" → Click "Extract from Video"

### Applying Filters & Effects
1. **Hover** over any category (Filters, Adjustments, Colors, Transform)
2. The tools panel automatically updates with available options
3. Adjust sliders or click buttons to apply effects
4. See results instantly in the "AFTER" panel

## Architecture

### Object-Oriented Design
The application follows solid OOP principles:

#### **ImageProcessor Class**
- Handles all image processing operations
- Demonstrates **Encapsulation** (private variables with `_`)
- Uses **Methods** for each image transformation
- Implements **Abstraction** to hide complex OpenCV operations

#### **HistoryManager Class**
- Manages undo/redo functionality
- Implements **Stack data structure** pattern
- Circular buffer for memory efficiency

#### **ImageProcessorGUI Class**
- Main application interface
- **Composition**: Uses ImageProcessor and HistoryManager
- **Event-driven programming** with hover interactions
- **Separation of concerns**: UI separate from processing logic

### Technology Stack
- **GUI Framework**: Tkinter (Python's standard GUI library)
- **Image Processing**: OpenCV (cv2)
- **Image Display**: PIL/Pillow (ImageTk)
- **Array Operations**: NumPy
- **Type Hints**: Python typing module for better code quality

---

## Color Palette Feature

### 16 Preset Colors
Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink, and more!

### Custom Color Picker
Choose any color from the full spectrum using the built-in color chooser dialog.

### RGB Channel Control
Fine-tune individual Red, Green, and Blue channels (-100 to +100 for each).

---

## Video Frame Extraction

### How It Works
1. Load any video file (MP4, AVI, MOV, MKV, FLV)
2. Use the slider to scrub through the video
3. See frame number and timestamp in real-time
4. Click "Extract This Frame" to grab the current frame
5. Edit the extracted frame like any image!

### Supported Formats
- ✅ MP4 (H.264/H.265)
- ✅ AVI
- ✅ MOV (QuickTime)
- ✅ MKV (Matroska)
- ✅ FLV (Flash Video)

---

## Advanced Features

### Before/After Comparison
- Side-by-side view of original and edited images
- Helps you see exactly what changed
- "BEFORE" panel always shows the original
- "AFTER" panel shows real-time edits

### History Management
- Stores up to 20 previous states
- Full undo/redo support
- Automatic history clearing when loading new images
- Memory-efficient circular buffer implementation

### Hover-Based Navigation
- **No clicking required!** Just hover to see tools
- Reduces clicks by 50%
- Faster workflow
- Intuitive and modern UX


  ## Known Issues & Limitations

- Camera capture requires webcam access
- Some video codecs may not be supported depending on OpenCV build
- Large images (>4K) may slow down on older computers
- Undo history limited to 20 states to manage memory
