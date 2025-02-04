import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

num_files = 17

def process_image(file_path, angle=-35, h_center=675):
    """
    Reads an image in color, applies cropping and rotation (as in your original processing),
    and returns two versions:
      - original_image: The original image in RGB (for color display)
      - processed_image: The processed image converted to grayscale.
    """
    # Read the image in color (BGR) and convert to RGB for display.
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found: {file_path}")
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- Original Processing Steps ---
    # Crop the top 20% of the image.
    height, width, _ = original_image.shape
    crop_height = int(0.2 * height)
    cropped_image = original_image[crop_height:, :]
    
    # Rotate the image clockwise.
    center = (width // 2, (height - crop_height) // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Calculate the new bounding dimensions of the rotated image.
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int((height - crop_height) * sin + width * cos)
    new_height = int((height - crop_height) * cos + width * sin)
    
    # Adjust the rotation matrix to account for translation.
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform the rotation.
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (new_width, new_height))
    
    # Crop a central region of the rotated image.
    new_height_rot, new_width_rot = rotated_image.shape[:2]
    rotated_cropped = rotated_image[int(new_height_rot * 0.3):int(new_height_rot * 0.75),
                                    int(new_width_rot * 0.35):int(new_width_rot * 0.65)]
    
    # Further crop: take only the left 75% of the width.
    final_crop = rotated_cropped[:, :int(0.75 * rotated_cropped.shape[1])]
    # --- End Original Processing Steps ---
    
    # Convert the processed image to grayscale.
    processed_gray = cv2.cvtColor(final_crop, cv2.COLOR_RGB2GRAY)
    
    return original_image, processed_gray

def resize_to_height(img, target_height):
    """
    Resizes an image (color or grayscale) to have the target height while preserving aspect ratio.
    """
    h, w = img.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)
    return resized

# Set up the Matplotlib figure and axes.
fig, axs = plt.subplots(1, 2, figsize=(12, 8))
ax_orig, ax_proc = axs
plt.tight_layout()


# Create dummy images. Here we also specify an initial extent for the grayscale image.
dummy_color = np.zeros((10, 10, 3), dtype=np.uint8)
dummy_gray = np.zeros((10, 10), dtype=np.uint8)
extent = [0, dummy_gray.shape[1], dummy_gray.shape[0], 0]

im_orig = ax_orig.imshow(dummy_color)
ax_orig.set_title('Original Color Image')
ax_orig.axis('off')

im_proc = ax_proc.imshow(dummy_gray, cmap='gray', extent=extent)
ax_proc.set_title('Processed Grayscale Image')
ax_proc.axis('off')

# Draw an initial vertical line at the center of the dummy processed image.
line_artist = ax_proc.axvline(dummy_gray.shape[1] // 2, color='red', linestyle='--', label='Intensity Profile')
ax_proc.legend(loc='upper right')

def update(frame):
    """
    Update function for the animation:
      - Process the image corresponding to the current frame.
      - Resize both images to have the same height.
      - Update both subplots and adjust the image extent so that the vertical line is drawn correctly.
    """
    file_name = os.path.join('images', f'pattern_{frame}.jpg')
    try:
        orig, proc = process_image(file_name)
    except Exception as e:
        print(e)
        return im_orig, im_proc, line_artist

    # Determine a common target height.
    orig_h = orig.shape[0]
    proc_h = proc.shape[0]
    target_height = max(orig_h, proc_h)
    
    orig_resized = resize_to_height(orig, target_height)
    proc_resized = resize_to_height(proc, target_height)
    
    # Update the original image.
    im_orig.set_data(orig_resized)
    
    # Update the processed image and its extent.
    im_proc.set_data(proc_resized)
    new_extent = [0, proc_resized.shape[1], proc_resized.shape[0], 0]
    im_proc.set_extent(new_extent)
    
    # Update the axes limits for the processed image.
    ax_proc.set_xlim(0, proc_resized.shape[1])
    ax_proc.set_ylim(proc_resized.shape[0], 0)
    
    # Remove previous vertical lines and draw a new one at the center.
    for line in ax_proc.get_lines():
        line.remove()
    x_center = proc_resized.shape[1] // 2
    line_artist = ax_proc.axvline(x_center, color='red', linestyle='--', label='Intensity Profile')
    ax_proc.legend(loc='upper right')
    
    # Update the grayscale image's color limits.
    im_proc.set_clim(vmin=np.min(proc_resized), vmax=np.max(proc_resized))
    return im_orig, im_proc, line_artist

# Create the animation (blit is disabled so that both axes update correctly).
anim = FuncAnimation(fig, update, frames=range(num_files), interval=1000, blit=False)

# Save the animation as a GIF.
gif_filename = 'color_and_gray_with_line.gif'
writer = PillowWriter(fps=3)
anim.save(gif_filename, writer=writer)
print(f'Animation saved as {gif_filename}')
