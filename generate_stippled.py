"""
Generate a stippled image from the original image using blue noise stippling.
This creates the stippled image needed for Part 2 of the challenge.
"""

import numpy as np
from PIL import Image
import os

def compute_importance(gray_img, extreme_downweight=0.5, extreme_threshold_low=0.4, 
                       extreme_threshold_high=0.8, extreme_sigma=0.1, 
                       mid_tone_boost=0.4, mid_tone_sigma=0.2):
    """
    Compute importance map for stippling.
    Higher importance = more dots should be placed there.
    """
    I = np.clip(gray_img, 0.0, 1.0)
    
    # Invert brightness: dark areas should get more dots (higher importance)
    I_inverted = 1.0 - I
    
    # Create smooth downweighting mask for extreme tones
    dark_mask = np.exp(-((I - 0.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    dark_mask = np.where(I < extreme_threshold_low, dark_mask, 0.0)
    if dark_mask.max() > 0:
        dark_mask = dark_mask / dark_mask.max()
    
    light_mask = np.exp(-((I - 1.0) ** 2) / (2.0 * (extreme_sigma ** 2)))
    light_mask = np.where(I > extreme_threshold_high, light_mask, 0.0)
    if light_mask.max() > 0:
        light_mask = light_mask / light_mask.max()
    
    extreme_mask = np.maximum(dark_mask, light_mask)
    importance = I_inverted * (1.0 - extreme_downweight * extreme_mask)
    
    # Add smooth gradual mid-tone boost
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    
    importance = importance + mid_tone_boost * mid_tone_gaussian
    importance = np.clip(importance, 0.0, 1.0)
    
    return importance

def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel."""
    kernel = np.zeros((size, size))
    center = size // 2
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def blue_noise_stippling(importance_map, dot_percentage=0.12, kernel_size=12, sigma=2.5):
    """
    Create blue noise stippling using importance sampling and repulsion.
    Improved to match exemplar style: small dots with varying densities.
    """
    height, width = importance_map.shape
    
    # Use adaptive dot percentage based on importance
    # Higher importance areas get more dots
    base_dots = int(height * width * dot_percentage)
    
    # Scale number of dots by average importance to get better coverage
    avg_importance = np.mean(importance_map)
    num_dots = int(base_dots * (1.0 + avg_importance))
    
    # Initialize with white background (1.0 = white, 0.0 = black dot)
    stipple = np.ones((height, width), dtype=np.float32)
    
    # Create repulsion kernel (Gaussian) - smaller for tighter spacing
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel_center = kernel_size // 2
    
    # Energy map: higher energy = less desirable to place a dot
    energy = np.ones((height, width), dtype=np.float32) * 1e6
    
    # Sample points based on importance and energy
    samples = []
    
    for i in range(num_dots):
        # Combine importance (want dots in important areas) and energy (avoid existing dots)
        # Lower energy = better place for new dot
        # Higher importance = better place for new dot
        score = importance_map / (energy + 1e-6)
        
        # Add controlled randomness - less random for better distribution
        randomness = 0.85 + 0.3 * np.random.random(score.shape)
        score = score * randomness
        
        # Find best location
        flat_idx = np.argmax(score.flatten())
        y, x = np.unravel_index(flat_idx, (height, width))
        
        # Place dot (single pixel for small dots)
        stipple[y, x] = 0.0
        
        # Update energy map (add repulsion around this dot)
        # Use stronger repulsion to ensure even spacing
        y_min = max(0, y - kernel_center)
        y_max = min(height, y + kernel_center + 1)
        x_min = max(0, x - kernel_center)
        x_max = min(width, x + kernel_center + 1)
        
        # Calculate kernel slice to match energy slice exactly
        k_y_min = kernel_center - (y - y_min)
        k_x_min = kernel_center - (x - x_min)
        
        # Get the actual sizes
        energy_h = y_max - y_min
        energy_w = x_max - x_min
        
        # Extract matching kernel slice - ensure it matches energy slice size
        k_y_max = min(k_y_min + energy_h, kernel_size)
        k_x_max = min(k_x_min + energy_w, kernel_size)
        k_y_min = max(0, k_y_min)
        k_x_min = max(0, k_x_min)
        
        # Adjust energy slice if kernel was clipped
        actual_k_h = k_y_max - k_y_min
        actual_k_w = k_x_max - k_x_min
        
        if actual_k_h < energy_h or actual_k_w < energy_w:
            # Adjust energy slice to match kernel
            y_max = y_min + actual_k_h
            x_max = x_min + actual_k_w
        
        # Extract the matching kernel portion
        kernel_slice = kernel[k_y_min:k_y_max, k_x_min:k_x_max]
        
        # Stronger repulsion for better blue noise properties
        energy[y_min:y_max, x_min:x_max] += kernel_slice * 1.5
        
        samples.append((y, x))
    
    return stipple, samples

def main():
    # Load original image
    img_path = 'E1311F60-BC46-4A56-8902-8DBC1E9AF295_4_5005_c copy.jpeg'
    
    if not os.path.exists(img_path):
        print(f"Error: Image file '{img_path}' not found!")
        return
    
    print(f"Loading image: {img_path}")
    original_img = Image.open(img_path)
    
    # Convert to grayscale
    if original_img.mode != 'L':
        original_img = original_img.convert('L')
    
    # Convert to numpy array and normalize
    original_array = np.array(original_img, dtype=np.float32) / 255.0
    print(f"Image shape: {original_array.shape}")
    
    # Resize if too large (for faster processing)
    max_dimension = 800
    if max(original_array.shape) > max_dimension:
        scale = max_dimension / max(original_array.shape)
        new_height = int(original_array.shape[0] * scale)
        new_width = int(original_array.shape[1] * scale)
        original_img_resized = original_img.resize((new_width, new_height), Image.LANCZOS)
        original_array = np.array(original_img_resized, dtype=np.float32) / 255.0
        print(f"Resized to: {original_array.shape}")
    
    # Compute importance map
    print("Computing importance map...")
    importance_map = compute_importance(original_array)
    
    # Generate stippled image
    print("Generating blue noise stippling...")
    # Increased dot percentage for better density variation matching exemplar
    dot_percentage = 0.12  # 12% dot coverage for denser, more detailed stippling
    stipple_array, samples = blue_noise_stippling(importance_map, dot_percentage=dot_percentage, 
                                                   kernel_size=12, sigma=2.5)
    
    print(f"Generated {len(samples)} stipple points ({dot_percentage*100:.1f}% coverage)")
    
    # Save as .npy file
    output_file = 'stippleImage.npy'
    np.save(output_file, stipple_array)
    print(f"✅ Saved stippled image to: {output_file}")
    print(f"   Shape: {stipple_array.shape}")
    print(f"   Value range: [{stipple_array.min():.2f}, {stipple_array.max():.2f}]")
    print(f"   (0.0 = black dot, 1.0 = white background)")

if __name__ == '__main__':
    main()

