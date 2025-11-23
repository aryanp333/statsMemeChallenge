"""
Generate a stippled image from the original image using blue noise stippling.
This uses the exact algorithm from Part 1 with toroidal Gaussian kernel.
"""

import numpy as np
from PIL import Image
from typing import Optional
import os

def compute_importance(
    gray_img: np.ndarray,
    extreme_downweight: float = 0.5,
    extreme_threshold_low: float = 0.4,
    extreme_threshold_high: float = 0.8,
    extreme_sigma: float = 0.1,
    mid_tone_boost: float = 0.4,
    mid_tone_sigma: float = 0.2,
):
    """
    Importance map computation that downweights extreme tones (very dark and very light)
    using smooth functions, while boosting mid-tones.
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
    
    # Add smooth gradual mid-tone boost (Gaussian centered on 0.65)
    mid_tone_center = 0.65
    mid_tone_gaussian = np.exp(-((I - mid_tone_center) ** 2) / (2.0 * (mid_tone_sigma ** 2)))
    if mid_tone_gaussian.max() > 0:
        mid_tone_gaussian = mid_tone_gaussian / mid_tone_gaussian.max()
    
    importance = importance * (1.0 + mid_tone_boost * mid_tone_gaussian)
    
    # Normalize to [0,1]
    m, M = importance.min(), importance.max()
    if M > m: 
        importance = (importance - m) / (M - m)
    return importance

def toroidal_gaussian_kernel(h: int, w: int, sigma: float):
    """
    Create a periodic (toroidal) 2D Gaussian kernel centered at (0,0).
    The toroidal property means the kernel wraps around at the edges,
    ensuring consistent repulsion behavior regardless of point location.
    """
    y = np.arange(h)
    x = np.arange(w)
    # Compute toroidal distances (minimum distance considering wrapping)
    dy = np.minimum(y, h - y)[:, None]
    dx = np.minimum(x, w - x)[None, :]
    # Compute Gaussian
    kern = np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))
    s = kern.sum()
    if s > 0:
        kern /= s  # Normalize
    return kern

def void_and_cluster(
    input_img: np.ndarray,
    percentage: float = 0.08,
    sigma: float = 0.9,
    content_bias: float = 0.9,
    importance_img: Optional[np.ndarray] = None,
    noise_scale_factor: float = 0.1,
):
    """
    Generate blue noise stippling pattern from input image using a modified
    void-and-cluster algorithm with content-weighted importance.
    This is the exact algorithm from Part 1.
    """
    I = np.clip(input_img, 0.0, 1.0)
    h, w = I.shape

    # Compute or use provided importance map
    if importance_img is None:
        importance = compute_importance(I)
    else:
        importance = np.clip(importance_img, 0.0, 1.0)

    # Create toroidal Gaussian kernel for repulsion
    kernel = toroidal_gaussian_kernel(h, w, sigma)

    # Initialize energy field: lower energy → more likely to be picked
    energy_current = -importance * content_bias

    # Stipple buffer: start with white background; selected points become black dots
    final_stipple = np.ones_like(I)
    samples = []

    # Helper function to roll kernel to an arbitrary position
    def energy_splat(y, x):
        """Get energy contribution by rolling the kernel to position (y, x)."""
        return np.roll(np.roll(kernel, shift=y, axis=0), shift=x, axis=1)

    # Number of points to select
    num_points = int(I.size * percentage)

    # Choose first point near center with minimal energy
    cy, cx = h // 2, w // 2
    r = min(20, h // 10, w // 10)
    ys = slice(max(0, cy - r), min(h, cy + r))
    xs = slice(max(0, cx - r), min(w, cx + r))
    region = energy_current[ys, xs]
    flat = np.argmin(region)
    y0 = flat // region.shape[1] + (cy - r) if region.shape[1] > 0 else cy - r
    x0 = flat % region.shape[1] + (cx - r) if region.shape[1] > 0 else cx - r

    # Place first point
    energy_current = energy_current + energy_splat(y0, x0)
    energy_current[y0, x0] = np.inf  # Prevent reselection
    samples.append((y0, x0, I[y0, x0]))
    final_stipple[y0, x0] = 0.0  # Black dot

    # Iteratively place remaining points
    for i in range(1, num_points):
        # Add exploration noise that decreases over time
        exploration = 1.0 - (i / num_points) * 0.5  # Decrease from 1.0 to 0.5
        noise = np.random.normal(0.0, noise_scale_factor * content_bias * exploration, size=energy_current.shape)
        energy_with_noise = energy_current + noise

        # Find position with minimum energy (with noise for exploration)
        pos_flat = np.argmin(energy_with_noise)
        y = pos_flat // w
        x = pos_flat % w

        # Add Gaussian splat to prevent nearby points from being selected
        energy_current = energy_current + energy_splat(y, x)
        energy_current[y, x] = np.inf  # Prevent reselection

        # Record the sample
        samples.append((y, x, I[y, x]))
        final_stipple[y, x] = 0.0  # Black dot

    return final_stipple, np.array(samples)

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
    img_array = np.array(original_img, dtype=np.float32) / 255.0
    print(f"Image shape: {img_array.shape}")
    
    # Resize if too large (for faster processing) - matching Part 1
    max_size = 512
    if img_array.shape[0] > max_size or img_array.shape[1] > max_size:
        scale = max_size / max(img_array.shape[0], img_array.shape[1])
        new_size = (int(img_array.shape[1] * scale), int(img_array.shape[0] * scale))
        img_resized_pil = original_img.resize(new_size, Image.Resampling.LANCZOS)
        if img_resized_pil.mode != 'L':
            img_resized_pil = img_resized_pil.convert('L')
        img_resized = np.array(img_resized_pil, dtype=np.float32) / 255.0
        print(f"Resized image from {img_array.shape} to {img_resized.shape} for processing")
    else:
        img_resized = img_array.copy()
    
    # Ensure img_resized is 2D grayscale
    if len(img_resized.shape) > 2:
        img_resized = img_resized[:, :, 0]
    elif len(img_resized.shape) == 2:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {img_resized.shape}")
    
    print(f"Final image shape: {img_resized.shape} (should be 2D for grayscale)")
    
    # Compute importance map using Part 1 parameters
    importance_map = compute_importance(
        img_resized,
        extreme_downweight=0.5,
        extreme_threshold_low=0.2,
        extreme_threshold_high=0.8,
        extreme_sigma=0.1
    )
    print("Importance map computed")
    
    # Generate stippled image using Part 1 algorithm
    print("Generating blue noise stippling pattern...")
    stipple_pattern, samples = void_and_cluster(
        img_resized,
        percentage=0.08,  # 8% dot coverage as in Part 1
        sigma=0.9,  # Part 1 parameter
        content_bias=0.9,  # Part 1 parameter
        importance_img=importance_map,
        noise_scale_factor=0.1  # Part 1 parameter
    )
    
    print(f"Generated {len(samples)} stipple points ({0.08*100:.1f}% coverage)")
    
    # Save as .npy file
    output_file = 'stippleImage.npy'
    np.save(output_file, stipple_pattern)
    print(f"✅ Saved stippled image to: {output_file}")
    print(f"   Shape: {stipple_pattern.shape}")
    print(f"   Value range: [{stipple_pattern.min():.2f}, {stipple_pattern.max():.2f}]")
    print(f"   (0.0 = black dot, 1.0 = white background)")

if __name__ == '__main__':
    main()
