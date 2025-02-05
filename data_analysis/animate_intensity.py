import cv2
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.signal import find_peaks

# Global parameters
num_files = 17
angle = -35
h_center = 675

# Other parameters (not all are used in this example)
peak_err = 1.57   # sigma R
B_err = 100; I_err = 0.025
lambda_0 = 585e-9  # m
h = 6.626e-34   # J/Hz
c = 3e8         # m/s
t = 0.01        # 1 cm
g = 1
mu_B = 9.274e-24  # J/T
m = 3   # peak of interest

def process_single_image(i, file_pattern='pattern_{}.jpg'):
    """
    Process image with index i, returning the intensity profile.
    """
    file_name = file_pattern.format(i)
    file_path = os.path.join('images', file_name)
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image file {file_path} not found.")

    # Crop the top 20% of the image
    height, width = image.shape[:2]
    crop_height = int(0.2 * height)
    cropped_image = image[crop_height:, :]

    # Rotate the image clockwise by the specified angle
    height_cropped, width_cropped = cropped_image.shape[:2]
    center = (width_cropped // 2, height_cropped // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    # Calculate the new bounding dimensions of the rotated image
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])
    new_width = int(height_cropped * sin + width_cropped * cos)
    new_height = int(height_cropped * cos + width_cropped * sin)

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform the rotation
    rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (new_width, new_height))

    # Crop a central region of the rotated image
    new_height, new_width = rotated_image.shape[:2]
    # Here we take a vertical slice between 30% and 75% of the height and
    # a horizontal slice between 35% and 65% of the width.
    rotated_cropped = rotated_image[int(new_height*0.3):int(new_height*0.75),
                                     int(new_width*0.35):int(new_width*0.65)]

    # Further crop the image horizontally (e.g., to the left 75% of the width)
    final_crop = rotated_cropped[:, :int(0.75 * rotated_cropped.shape[1])]

    # Extract the intensity profile from the middle column,
    # starting from pixel h_center (adjusted relative to the cropped image)
    col = final_crop.shape[1] // 2
    intensity = final_crop[h_center:, col]
    
    # Optionally, you could also compute the reversed intensity if needed:
    # reversed_intensity = intensity[::-1]

    return intensity

# Precompute the intensity profiles for all images.
# (Alternatively, you can compute them on the fly in the animation function.)
intensity_profiles = []
for i in range(num_files):
    try:
        intensity = process_single_image(i)
        intensity_profiles.append(intensity)
    except FileNotFoundError as e:
        print(e)
        intensity_profiles.append(np.array([]))  # Append empty array if file not found

# Set up the figure and initial plot.
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], 'k.', markersize=3)
ax.set_title('Intensity Profile (Image 0)')
ax.set_xlabel('Pixel Position')
ax.set_ylabel('Pixel Intensity')
ax.grid(True)

def init():
    """Initialize the background of the animation."""
    ax.set_xlim(0, 500)  # Adjust limits if necessary
    ax.set_ylim(0, 170)  # Assuming 8-bit grayscale images
    line.set_data([], [])
    return line,

def animate(frame):
    ax.cla()  # Clear the current axes
    # Redraw static elements
    ax.grid(True)
    ax.set_xlabel('Pixel Position')
    ax.set_ylabel('Pixel Intensity')
    intensity = intensity_profiles[frame]
    if intensity.size == 0:
        x = []
        y = []
    else:
        x = np.arange(len(intensity))
        y = intensity

    peaks, _ = find_peaks(y, width=10, distance=40, prominence=5)
    # Plot the intensity profile
    ax.plot(x, y, 'k.', markersize=3)
    # Plot the peaks
    for p in peaks:
        ax.axvline(x=p, color='red', linestyle='--', alpha=0.5)
    
    ax.set_title(f'm=3 Intensity Profile (Image {frame})')
    ax.set_xlim(1200, 1570)
    ax.set_ylim(0, 170)
    return ax.lines


# Create the animation.
anim = FuncAnimation(fig, animate, init_func=init,
                     frames=num_files, interval=1000, blit=True)

# Save the animation as a GIF.
gif_filename = 'intensity_animation.gif'
writer = PillowWriter(fps=3)
anim.save(gif_filename, writer=writer)

print(f'Animation saved as {gif_filename}')
