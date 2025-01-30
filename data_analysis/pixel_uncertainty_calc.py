import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os

num_files = 17

#indices where splitting starts
split_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16]

#indices corresponding to m values
m_peaks = [[0,1,3,4], [0,2,4,5], [0,3,6,9], [0,3,6,9], [0,3,6,9], [0,3,6,9], [1,4,7,10], [1,4,7,10], [1,4,7,10]]


def process_image(file_pattern='pattern_{}.jpg', num_files=num_files, angle = -35, h_center = 675):
    intensity_peaks = []
    mean_diffs = []
    
    for i in range(num_files):
        file_name = file_pattern.format(i)
        # change the file directory to images/file_name
        file_name = os.path.join('images', file_name)
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        
        # Crop the top 20% of the image
        height, width = image.shape[:2]
        crop_height = int(0.2 * height)  # Calculate the height to crop
        cropped_image = image[crop_height:, :]  # Crop from 20% to the bottom

        # Rotate the image clockwise
        center = (width // 2, (height - crop_height) // 2)  # Center of rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        
        # Calculate the new bounding dimensions of the rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height - crop_height) * sin + width * cos)
        new_height = int((height - crop_height) * cos + width * sin)

        # Adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        # Perform the rotation
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (new_width, new_height))
        new_height, new_width = rotated_image.shape[:2]
        rotated_image = rotated_image[int(new_height*0.3):int(new_height*0.75), int(new_width*0.35):int(new_width*0.65)]
        
        #crop RHS
        height, width = rotated_image.shape[:2]
        crop_width = int(0.75 * width)
        cropped_image = rotated_image[:, :crop_width]  
        
        """plt.figure()
        plt.imshow(cropped_image, cmap='gray')
        plt.axvline(cropped_image.shape[1] // 2)
        plt.axhline(y=h_center)
        plt.title(f'rotate image {i}')
        plt.show()"""
        
        #get the intensity of middle column (vertical line)
        intensity = cropped_image[h_center:, cropped_image.shape[1] // 2]
        reveresed_intensity = intensity[::-1]
        
        peaks, _ = find_peaks(intensity, width = 10, distance = 40, prominence=5)
        reversed_peaks, _ = find_peaks(reveresed_intensity, width = 10, distance = 40, prominence=5)
        peaks = list(peaks)
        reveresed_peaks = list(reversed_peaks)

        # Change the reversed_peaks back to the original orientation
        reveresed_peaks = [len(intensity) - p for p in reveresed_peaks]
        
        j = 0
        while j < len(peaks):
            if (peaks[j] < 475 or peaks[j] > 1900):
                peaks.pop(j)
            else: 
                j +=1

        j = 0
        while j < len(reveresed_peaks):
            if (reveresed_peaks[j] < 475 or reveresed_peaks[j] > 1900):
                reveresed_peaks.pop(j)
            else: 
                j +=1

        reveresed_peaks = reveresed_peaks[::-1]

        differences = [abs(peaks[i] - reveresed_peaks[i]) for i in range(len(peaks))]

        print('Differences:', differences)
        print('Mean Difference:', np.mean(np.array(differences)))
        mean_diffs.append(np.mean(np.array(differences)))
        
        # Plot the intensity values
        plt.figure()
        plt.plot(intensity, 'k.')
        plt.title(f'Forward and Reverse Peaks of Intensity Profile to Zeeman Lines')
        for k, p in enumerate(peaks):
            if k == 0:
                plt.axvline(x=p, color='red', linestyle='--', label='Forward Peaks')
            else:
                plt.axvline(x=p, color='red', linestyle='--')
        for k, p in enumerate(reveresed_peaks):
            if k == 0:
                plt.axvline(x=p, color='blue', linestyle='--', label='Reverse Peaks')
            else:
                plt.axvline(x=p, color='blue', linestyle='--')
        plt.xlabel('Pixel Position (Y)')
        plt.ylabel('Intensity Value')
        plt.grid()
        plt.legend()
        print(i)
        #if i > 13:
            #plt.savefig(f'forward_reverse_peaks_{i}.png', dpi=300)
        
        intensity_peaks.append(peaks)
        
    return mean_diffs, np.mean(np.array(mean_diffs))

differences, mean_difference = process_image()


# plot a histogram of differences
plt.figure()
plt.hist(differences, bins=10)
plt.axvline(x=mean_difference, color='red', linestyle='--', label='Mean Difference = {:.2f} pixels'.format(mean_difference))
plt.title('Histogram of Differences Between Forward and Reverse Peaks')
plt.xlabel('Pixel Difference')
plt.ylabel('Frequency')
plt.grid()
plt.legend()
plt.savefig(f'histogram_differences.png', dpi=300)