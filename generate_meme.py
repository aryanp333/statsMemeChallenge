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
s_image = Image.new('L', (width, height), 255)
draw = ImageDraw.Draw(s_image)

font_size = int(height * 0.85)
font = None

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
            print(f"Using font: {font_path}")
            break
        except:
            continue

if font is None:
    try:
        font = ImageFont.load_default()
        font_size = min(width, height) // 2
        print("Using default font (will be made bold)")
    except:
        font = None
        font_size = min(width, height) // 2

# Get text bounding box to center it properly
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
        
        # Small visual adjustment: text often looks better slightly higher than 
        # mathematically centered (optical centering)
        y = y - int(height * 0.02)  # Move up by 2% of height for better visual balance
        
    except AttributeError:
        try:
            # Older PIL version
            text_width, text_height = draw.textsize("S", font=font)
            x = (width - text_width) // 2
            y = (height - text_height) // 2
        except:
            text_width = width // 3
            text_height = height // 3
            x = (width - text_width) // 2
            y = (height - text_height) // 2
else:
    text_width = width // 3
    text_height = height // 3
    x = (width - text_width) // 2
    y = (height - text_height) // 2

print(f"S placement: position ({x}, {y}), text bbox {bbox if font else 'N/A'}, image size {width}x{height}")

# Draw the letter "S" - replicate the exemplar style: wide horizontally, narrow vertically
if font:
    # Create a high-resolution temporary image for better quality
    scale_factor = 4  # High resolution for smooth curves
    temp_width = width * scale_factor
    temp_height = height * scale_factor
    temp_img = Image.new('L', (temp_width, temp_height), 255)
    temp_draw = ImageDraw.Draw(temp_img)
    
    # Draw S at a good size - make it large to fill the space
    large_font_size = int(font_size * scale_factor * 1.2)  # Larger base size
    try:
        large_font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf', large_font_size)
    except:
        try:
            large_font = ImageFont.truetype('/System/Library/Fonts/Helvetica Bold.ttf', large_font_size)
        except:
            large_font = font
    
    # Get bbox and center the S in temp image
    temp_bbox = temp_draw.textbbox((0, 0), "S", font=large_font)
    temp_left, temp_top, temp_right, temp_bottom = temp_bbox
    temp_text_width = temp_right - temp_left
    temp_text_height = temp_bottom - temp_top
    
    temp_center_x = temp_width // 2
    temp_center_y = temp_height // 2
    temp_text_center_x = temp_left + temp_text_width // 2
    temp_text_center_y = temp_top + temp_text_height // 2
    
    temp_x = temp_center_x - temp_text_center_x
    temp_y = temp_center_y - temp_text_center_y - int(temp_height * 0.02)  # Slight upward adjustment
    
    # Draw bold, thick S - use multiple passes for thickness
    # Create a very bold appearance with smooth curves
    bold_offsets = [
        (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2),
        (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
        (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3),
        (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3),
        (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3),
        (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
        (3, -2), (3, -1), (3, 0), (3, 1), (3, 2)
    ]
    for dx, dy in bold_offsets:
        temp_draw.text((temp_x + dx, temp_y + dy), "S", fill=0, font=large_font)
    
    # Transform: Make it WIDER (stretch horizontally) and NARROWER (compress vertically)
    # Match exemplar: notably wide horizontally, compressed vertically
    stretch_horizontal = 1.5  # Make it 50% wider
    compress_vertical = 0.65  # Make it 35% narrower (more compressed)
    
    new_width = int(temp_img.width * stretch_horizontal)
    new_height = int(temp_img.height * compress_vertical)
    stretched = temp_img.resize((new_width, new_height), Image.LANCZOS)
    
    # Find the bounding box of the S in the stretched image
    stretched_array = np.array(stretched)
    rows = np.any(stretched_array < 200, axis=1)  # Threshold for black pixels
    cols = np.any(stretched_array < 200, axis=0)
    
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Add padding to keep smooth edges
        padding = int(scale_factor * 10)
        rmin = max(0, rmin - padding)
        rmax = min(stretched.height, rmax + padding)
        cmin = max(0, cmin - padding)
        cmax = min(stretched.width, cmax + padding)
        
        cropped = stretched.crop((cmin, rmin, cmax, rmax))
        
        # Resize to fill most of the image (significant portion of canvas)
        # Scale to fill about 85% of the smaller dimension
        scale_to_fit = min(width / cropped.width, height / cropped.height) * 0.85
        final_width = int(cropped.width * scale_to_fit)
        final_height = int(cropped.height * scale_to_fit)
        
        resized_s = cropped.resize((final_width, final_height), Image.LANCZOS)
        
        # Center it perfectly in the main image
        paste_x = (width - final_width) // 2
        paste_y = (height - final_height) // 2
        s_image.paste(resized_s, (paste_x, paste_y))
        
        print(f"S created: {final_width}x{final_height} (stretched {stretch_horizontal}x wider, {compress_vertical}x narrower)")
    else:
        # Fallback: draw normally with stretching
        print("Warning: Could not detect S bounds, using fallback")
        for dx, dy in bold_offsets[:25]:  # Use fewer offsets for fallback
            draw.text((x + dx, y + dy), "S", fill=0, font=font)
else:
    # Fallback: draw multiple times
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            draw.text((x + dx, y + dy), "S", fill=0, font=font)

# Convert to numpy array and normalize to [0, 1]
s_array = np.array(s_image, dtype=np.float32) / 255.0

# Step 3: Create the masked estimate
threshold = 0.5
s_mask = s_array < threshold

# Apply mask: where S is black, remove stipples (set to white)
masked_stipple = stipple_array.copy()
masked_stipple[s_mask] = 1.0

# Step 4: Assemble the four-panel meme
# Match exemplar dimensions: 2334 x 925 (aspect ratio ~2.52:1)
# For 1x4 layout, calculate appropriate figure size
target_width = 2334
target_height = 925
aspect_ratio = target_width / target_height

# Calculate figure size to match exemplar aspect ratio
fig_height = 6
fig_width = fig_height * aspect_ratio

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
fig.patch.set_facecolor('#FFE5E5')  # Light pink background

panels = [
    (original_array, "Reality"),
    (stipple_array, "Your Model"),
    (s_array, "Selection Bias"),
    (masked_stipple, "Estimate")
]

print(f"\nGenerating four-panel meme:")
for i, (panel_data, title) in enumerate(panels, 1):
    print(f"  - Panel {i}: {title} ({panel_data.shape})")
    # Display image - use 'auto' aspect to fill panel properly
    axes[i-1].imshow(panel_data, cmap='gray', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    # Match exemplar title styling - larger, bold, centered
    axes[i-1].set_title(title, fontsize=20, fontweight='bold', pad=18, color='black', 
                        fontfamily='sans-serif')
    axes[i-1].axis('off')

# Match exemplar spacing - very minimal gaps between panels, more space for titles
plt.tight_layout()
plt.subplots_adjust(wspace=0.02, hspace=0, left=0.02, right=0.98, top=0.88, bottom=0.12)

# Save the meme with dimensions matching exemplar
output_file = 'statistics_meme_result.png'
# Save at higher DPI to match exemplar quality, then we can resize if needed
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='#FFE5E5', pad_inches=0.1)
print(f"\n✅ Meme saved as '{output_file}'")

# Optionally resize to match exemplar dimensions exactly
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

