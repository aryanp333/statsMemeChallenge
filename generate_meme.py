"""
Generate the four-panel statistics meme.
This script creates the meme image directly without Quarto.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os

# Step 1: Load images
image_files = [
    'E1311F60-BC46-4A56-8902-8DBC1E9AF295_4_5005_c copy.jpeg',
    'E1311F60-BC46-4A56-8902-8DBC1E9AF295_4_5005_c.jpeg',
    'original_image.jpg', 'original_image.jpeg', 'original_image.png',
    'image.jpg', 'image.jpeg', 'image.png'
]

stipple_files = [
    'stippleImage.npy', 'stipple.npy', 'stippled_image.npy'
]

# Find and load original image
original_array = None
for img_file in image_files:
    if os.path.exists(img_file):
        original_img = Image.open(img_file)
        if original_img.mode != 'L':
            original_img = original_img.convert('L')
        original_array = np.array(original_img, dtype=np.float32) / 255.0
        print(f"Loaded original image: {img_file}, shape: {original_array.shape}")
        break

if original_array is None:
    raise FileNotFoundError("Could not find original image.")

# Find and load stippled image
stipple_array = None
for stipple_file in stipple_files:
    if os.path.exists(stipple_file):
        stipple_array = np.load(stipple_file)
        print(f"Loaded stippled image: {stipple_file}, shape: {stipple_array.shape}")
        break

if stipple_array is None:
    raise FileNotFoundError("Could not find stippled image (.npy file).")

# Ensure both arrays have the same dimensions
if original_array.shape != stipple_array.shape:
    stipple_img = Image.fromarray((stipple_array * 255).astype(np.uint8))
    stipple_img = stipple_img.resize((original_array.shape[1], original_array.shape[0]), Image.LANCZOS)
    stipple_array = np.array(stipple_img, dtype=np.float32) / 255.0
    print(f"Resized stipple to match original: {stipple_array.shape}")

height, width = original_array.shape

# Step 2: Create the block letter "S"
# Replicate exemplar: Large, bold, black "S" centered on white background
# - White background matrix (height × width)
# - Bold letter "S" that fills most of the image space
# - Convert to [0, 1] range (0.0 = black, 1.0 = white)

# Create white background matrix matching image dimensions
s_image = Image.new('L', (width, height), 255)  # 255 = white
draw = ImageDraw.Draw(s_image)

# Use large font size to fill most of the space (90% of height for maximum size)
font_size = int(height * 0.90)
font = None

# Try to find a bold sans-serif font (Arial Bold is ideal)
font_paths = [
    '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
    '/System/Library/Fonts/Helvetica Bold.ttf',
    '/Library/Fonts/Arial Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    '/Windows/Fonts/arialbd.ttf',
    '/Windows/Fonts/calibrib.ttf',
]

for font_path in font_paths:
    if os.path.exists(font_path):
        try:
            font = ImageFont.truetype(font_path, font_size)
            print(f"Using bold font: {font_path} at size {font_size}")
            break
        except:
            continue

if font is None:
    try:
        font = ImageFont.load_default()
        font_size = int(height * 0.90)
        print(f"Using default font at size {font_size}")
    except:
        font = None
        font_size = int(height * 0.90)

# Get text bounding box to perfectly center the "S"
if font:
    try:
        # Get bounding box when drawn at origin
        bbox = draw.textbbox((0, 0), "S", font=font)
        left, top, right, bottom = bbox
        
        # Calculate text dimensions
        text_width = right - left
        text_height = bottom - top
        
        # Calculate center of image
        img_center_x = width // 2
        img_center_y = height // 2
        
        # Calculate center of text bounding box (when drawn at 0,0)
        text_center_x = left + text_width // 2
        text_center_y = top + text_height // 2
        
        # Position to center: align text center with image center
        x = img_center_x - text_center_x
        y = img_center_y - text_center_y
        
        print(f"S placement: centered at ({x}, {y}), text size {text_width}x{text_height}, image size {width}x{height}")
        
    except AttributeError:
        # Older PIL version
        try:
            text_width, text_height = draw.textsize("S", font=font)
            x = (width - text_width) // 2
            y = (height - text_height) // 2
        except:
            text_width = width // 2
            text_height = height // 2
            x = (width - text_width) // 2
            y = (height - text_height) // 2
else:
    text_width = width // 2
    text_height = height // 2
    x = (width - text_width) // 2
    y = (height - text_height) // 2

# Draw the bold letter "S" in black (fill=0 means black)
# For bold sans-serif font, draw once - it should be naturally bold
if font:
    # Draw the S - bold fonts should render as solid black
    draw.text((x, y), "S", fill=0, font=font)
    
    # If the font doesn't appear bold enough, we can thicken it slightly
    # But for Arial Bold, one draw should be sufficient
else:
    # Fallback: draw multiple times for thickness
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            draw.text((x + dx, y + dy), "S", fill=0, font=font)

# Convert to numpy array and normalize to [0, 1]
s_array = np.array(s_image, dtype=np.float32) / 255.0

# Step 3: Create the Masked Estimate
# Following instructions:
# - Create binary mask by thresholding the "S" image (pixels < 0.5 are part of the "S")
# - Apply the mask: where the "S" is black, set those pixels to white (1.0)
# - Where the "S" is white, keep the original stipple values
# - This creates the visual effect of "missing data" in the shape of the "S"
# - Use conditional assignment or boolean masking (np.where() in Python)

# Create binary mask by thresholding the "S" image
# Pixels < 0.5 are part of the "S" (black = missing data region)
threshold = 0.5
s_mask = s_array < threshold  # True where "S" is black (missing data)

# Apply mask to stippled image using np.where() for conditional assignment
# Where "S" is black (s_mask is True): remove stipples by setting to white (1.0)
# Where "S" is white (s_mask is False): keep original stipple values
masked_stipple = np.where(s_mask, 1.0, stipple_array)

print(f"Masked estimate created: {np.sum(s_mask)} pixels masked (S shape)")

# Step 4: Assemble the Four-Panel Meme
# Following instructions:
# - Create a multi-panel layout (1×4 for horizontal)
# - Display each matrix as a grayscale image: original, stippled, "S", and masked stippled
# - Add clear labels: "Reality", "Your Model", "Selection Bias", "Estimate"
# - Use minimal spacing between panels for a clean, professional look
# - Consider a light background color (like pink) to make panels stand out
# - Save with high DPI (150-300) for quality output
# - Use plt.subplots() and imshow() for each panel

# Match exemplar dimensions: 2334 x 925 (aspect ratio ~2.52:1)
target_width = 2334
target_height = 925
aspect_ratio = target_width / target_height

# Calculate figure size to match exemplar aspect ratio
fig_height = 6
fig_width = fig_height * aspect_ratio

# Create 1×4 multi-panel layout using plt.subplots()
fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
fig.patch.set_facecolor('#FFE5E5')  # Light pink background (#FFE5E5) to make panels stand out

# Define the four panels with clear labels
panels = [
    (original_array, "Reality"),
    (stipple_array, "Your Model"),
    (s_array, "Selection Bias"),
    (masked_stipple, "Estimate")
]

print(f"\nGenerating four-panel meme:")
for i, (panel_data, title) in enumerate(panels, 1):
    print(f"  - Panel {i}: {title} ({panel_data.shape})")
    
    # Display each matrix as a grayscale image using imshow()
    axes[i-1].imshow(panel_data, cmap='gray', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    
    # Add clear labels
    axes[i-1].set_title(title, fontsize=20, fontweight='bold', pad=18, color='black', 
                        fontfamily='sans-serif')
    axes[i-1].axis('off')  # Remove axes for clean look

# Use minimal spacing between panels (wspace=0, hspace=0) for clean, professional look
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0, left=0.02, right=0.98, top=0.88, bottom=0.12)

# Save with high DPI (150-300) for quality output
output_file = 'statistics_meme_result.png'
# Save at 300 DPI for high quality, then resize to match exemplar dimensions
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#FFE5E5', pad_inches=0.1)
print(f"\n✅ Meme saved as '{output_file}' (300 DPI)")

# Resize to match exemplar dimensions exactly (2334 x 925)
try:
    from PIL import Image as PILImage
    img = PILImage.open(output_file)
    # Resize to match exemplar dimensions (2334 x 925)
    img_resized = img.resize((2334, 925), PILImage.LANCZOS)
    img_resized.save(output_file, 'PNG', dpi=(300, 300))
    print(f"✅ Resized to match exemplar dimensions: 2334 x 925")
except Exception as e:
    print(f"Note: Could not resize to exact dimensions: {e}")

plt.close()

print("✅ Four-panel meme generation complete!")

