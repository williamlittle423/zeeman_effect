import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

num_files = 17

#indices where splitting starts
split_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16]

#indices corresponding to m values
m_peaks = [[0,1,3,4], [0,2,4,5], [0,3,6,9], [0,3,6,9], [0,3,6,9], [0,3,6,9], [1,4,7,10], [1,4,7,10], [1,4,7,10]]


def process_image(file_pattern='pattern_{}.jpg', num_files=num_files, angle = -35, h_center = 675):
    intensity_peaks = []
    
    for i in range(num_files):
        file_name = file_pattern.format(i)
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
        
        plt.figure()
        plt.imshow(cropped_image, cmap='gray')
        plt.axvline(cropped_image.shape[1] // 2)
        plt.axhline(y=h_center)
        plt.title(f'rotate image {i}')
        plt.show()
        
        #get the intensity of middle column (vertical line)
        intensity = cropped_image[h_center:, cropped_image.shape[1] // 2]
        
        peaks, _ = find_peaks(intensity, width = 10, distance = 40, prominence=5)
        
        peaks = list(peaks)
        
        j = 0
        while j < len(peaks):
            if (peaks[j] < 475 or peaks[j] > 1900):
                peaks.pop(j)
            else: 
                j +=1
        
        # Plot the intensity values
        plt.figure()
        plt.plot(intensity, 'k.')
        plt.title('Vertical Line Pixel Intensity in Centre of Image')
        for p in peaks:
            plt.axvline(x=p, color='red', linestyle='--')
        plt.xlabel('Pixel Position (Y)')
        plt.ylabel('Intensity Value')
        plt.grid()
        plt.show()
        
        intensity_peaks.append(peaks)
        
    return intensity_peaks

def find_slope(peak, split_indices=split_indices, m_peaks=m_peaks, num_files=num_files):
    
    fit_params = []
    plt.figure(figsize=(10,8))

    for i in range(num_files):
        X = []; Y = []
        
        if i in split_indices:
            for j in range(len(peak[i])):
                if j in m_peaks[split_indices.index(i)]:
                    X.append(m_peaks[split_indices.index(i)].index(j)+1)
                    Y.append(peak[i][j]**2)
        else:
            for j in range(len(peak[i])):
                X.append(j+1)
                Y.append(peak[i][j]**2)
            
        x = np.array(X); y = np.array(Y)
            
        a,b = np.polyfit(x,y,1)
        y_est = a * x + b
        
        for j in range(len(peak[i])):
            X.append(j+1)
            Y.append(peak[i][j]**2)
            
        fit_params.append([a, b])
        
        
        fig = plt.plot(x, y, '.', label=f'trial {i}')
        plt.plot(x, y_est, color = fig[-1].get_color())
    
    plt.xlabel('m')
    plt.ylabel('R $^2$')
    plt.title('Distance between m level relationship')
    plt.legend(); plt.show()
    
    return fit_params

intensity_peaks = process_image()

fit_params = find_slope(intensity_peaks)


#m = 3 peak 
m = 3
delta_R = []
B = np.linspace(0, 0.8, num_files)

left_split = []; m_line = []; right_split = []

plt.figure()

for i in range(num_files):
    if i in (10,11,12,13,14,15,16):
        for j in range(len(intensity_peaks[i])):
            if j in m_peaks[i-8]:
                if (m_peaks[i-8].index(j)+1 == m):
                    left_split = intensity_peaks[i][j-1]
                    m_line = intensity_peaks[i][j]
                    right_split = intensity_peaks[i][j+1]
                    dR = ((abs(left_split-m_line) + abs(right_split-m_line))/2)**2
                    
    elif i in (8, 9):
        for j in range(len(intensity_peaks[i])):
            if j in m_peaks[i-8]:
                if (m_peaks[i-8].index(j)+1 == m):
                    left_split = intensity_peaks[i][j-1]
                    m_line = intensity_peaks[i][j]
                    dR = (left_split-m_line)**2
    else:
        dR = 0
    
    delta_R.append(dR)

delta_R = delta_R[8:]; B = B[8:]

a,b = np.polyfit(B,delta_R,1)
dR_exp = a * B + b

print("params:", a, b)

plt.plot(B, delta_R, 'k.')
plt.plot(B, dR_exp, 'r')
plt.xlabel('B (G)')
plt.ylabel('${\Delta}$R$^2$')
plt.title('${\Delta}$R$^2$ over B for m=3')
plt.show()





