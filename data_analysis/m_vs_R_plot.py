import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import chisquare
import os
import matplotlib.ticker as ticker


num_files = 17

#indices where splitting starts
split_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16]

#indices corresponding to m values
m_peaks = [[0,1,3,4], [0,2,4,5], [0,3,6,9], [0,3,6,9], [0,3,6,9], [0,3,6,9], [1,4,7,10], [1,4,7,10], [1,4,7,10]]

peak_err = 1.57   #sigma R
B_err = 100; I_err = 0.025

lambda_0 = 585e-9  #m
h = 6.626e-34   #J/Hz
c = 3e8   #m/s
t = 0.01 #1 cm
g = 1
mu_B = 9.274e-24    #J/T
m = 3   #peak of interest

print(f"Parameters: \n h = {h} J*s \n c = {c:.2e} m/s \n t = {t} m \n g = {g} \n m = {m} \n")

def process_image(file_pattern='pattern_{}.jpg', num_files=num_files, angle = -35, h_center = 675):
    intensity_peaks = []; mean_diffs = []
    total_count = 0
    
    for i in range(num_files):
        file_name = file_pattern.format(i)
        # change the file directory to images/file_name
        print('file_name: ', file_name)
        file_name = os.path.join('images', file_name)
        image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        
        #crop the top 20% of the image
        height, width = image.shape[:2]
        crop_height = int(0.2 * height)  # Calculate the height to crop
        cropped_image = image[crop_height:, :]  # Crop from 20% to the bottom

        #rotate the image clockwise
        center = (width // 2, (height - crop_height) // 2)  # Center of rotation
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        
        #calculate the new bounding dimensions of the rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_width = int((height - crop_height) * sin + width * cos)
        new_height = int((height - crop_height) * cos + width * sin)

        #adjust the rotation matrix to account for translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]

        #perform the rotation
        rotated_image = cv2.warpAffine(cropped_image, rotation_matrix, (new_width, new_height))
        new_height, new_width = rotated_image.shape[:2]
        rotated_image = rotated_image[int(new_height*0.3):int(new_height*0.75), int(new_width*0.35):int(new_width*0.65)]
        
        #crop RHS
        height, width = rotated_image.shape[:2]
        crop_width = int(0.75 * width)
        cropped_image = rotated_image[:, :crop_width]  
        
        '''
        plt.figure()
        plt.imshow(cropped_image, cmap='gray')
        plt.axvline(cropped_image.shape[1] // 2)
        plt.axhline(y=h_center)
        plt.title(f'rotate image {i}')
        plt.savefig(f'line_of_intensity_{i}.jpg')
        plt.show() '''
        
        #get the intensity of middle column (vertical line)
        intensity = cropped_image[h_center:, cropped_image.shape[1] // 2]
        reveresed_intensity = intensity[::-1]
        
        peaks, _ = find_peaks(intensity, width = 10, distance = 40, prominence=5)
        reversed_peaks, _ = find_peaks(reveresed_intensity, width = 10, distance = 40, prominence=5)
        peaks = list(peaks)
        reveresed_peaks = list(reversed_peaks)

        #change the reversed_peaks back to the original orientation
        reveresed_peaks = [len(intensity) - p for p in reveresed_peaks]
        
        #remove fringe values
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
        total_count += len(differences)

        mean_diffs.append(np.mean(np.array(differences)))
        
        intensity_peaks.append(peaks)
        
    mean_diffs = [i for i in mean_diffs if i < 100]
    print(f'Total Count: {total_count:.2f}')
    print(f'Standard Deviation: {np.std(np.array(mean_diffs)):.2f} \n')
    
    return intensity_peaks

def red_chi_sq(y_obs, y_exp):
    chi = chisquare(y_obs, y_exp)[0]
    print(chi)
    return chi/(len(y_obs)-2)
    
def find_slope_R_difference(peak, peak_err, split_indices=split_indices, m_peaks=m_peaks, num_files=num_files):
    a_params = []
    b_params = []
    sig_a = []
    sig_b = []
    # SET THE SIZE TO (10, 6) INCHES
    plt.figure(figsize=(10, 6))
    m_obs = [[], [], [], []]

    # Set up a colormap for the trials.
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=num_files - 1)

    # Loop over trials.
    for i in range(num_files):
        X = []
        Y = []
        
        if i in split_indices:
            # Only take the peaks that match the split level.
            for j in range(len(peak[i])):
                if j in m_peaks[split_indices.index(i)]:
                    # Get the m-index from the m_peaks list.
                    X.append(m_peaks[split_indices.index(i)].index(j) + 1)
                    Y.append(peak[i][j]**2)
        else:
            for j in range(len(peak[i])):
                X.append(j + 1)
                Y.append(peak[i][j]**2)
            
        x = np.array(X)
        y = np.array(Y)
        
        # Find the slope and intercept for this trial.
        p, cov = np.polyfit(x, y, 1, cov=True)
        a = p[0]
        b = p[1]
        sig_a.append(np.sqrt(cov[0][0]))
        sig_b.append(np.sqrt(cov[1][1]))
        
        y_est = a * x + b
        
        # Append the observed y-values to compute the average later.
        for j in range(len(m_obs)):
            m_obs[j].append(y[j])
            
        a_params.append(a)
        b_params.append(b)
        
        # Determine the color for this trial from the colormap.
        color = cmap(norm(i))
        
        # Plot the trendline and errorbar using the same color.
        plt.plot(x, y_est, color=color)
        plt.errorbar(x, y, yerr=2 * np.sqrt(y) * peak_err, fmt='.', color=color)
    
    # Compute the mean for each m-value.
    for i in range(len(m_obs)):
        m_obs[i] = np.mean(m_obs[i])
        
    # Compute average parameters and errors.
    average_slope = np.mean(a_params)
    average_intercept = np.mean(b_params)
    slope_err = np.mean(sig_a)
    intercept_err = np.mean(sig_b)
    
    m = np.array([1, 2, 3, 4])
    print(m_obs)
    print(average_slope * m + average_intercept)
    print(red_chi_sq(m_obs, average_slope * m + average_intercept))
    
    # Plot the average trendline and error bounds.
    plt.plot(m, average_slope * m + average_intercept,
             label='Average Slope: $R^2$ = ({:.2e})m - {:.2e}'.format(average_slope, -average_intercept),
             linestyle='dashed', color='k')
    plt.plot(m, (average_slope + slope_err) * m + average_intercept,
             label='Error Bounds', linestyle='dashed', color='b')
    plt.plot(m, (average_slope - slope_err) * m + average_intercept,
             linestyle='dashed', color='b')
    
    plt.xlabel('m')
    plt.ylabel('$R^2$ [$Pixels^2$]')
    
    # Instead of many trial legends, create a ScalarMappable for the colorbar.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Get the current axes and attach the colorbar to it.
    ax = plt.gca()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    cbar = plt.colorbar(sm, ax=ax, label='Trial Number')
    
    # Add legend only for the average and error bounds.
    plt.legend(loc='upper left', frameon=False)
    plt.annotate("$\u03c7_\u03bd^2$ = {:.2f}".format(red_chi_sq(m_obs, average_slope * m + average_intercept)), (0, 1))
    plt.savefig('R2_vs_m.jpg', dpi=300)
    plt.show()
    
    return a_params, b_params, average_slope, average_intercept, slope_err, intercept_err

intensity_peaks = process_image()

intensity_slope, intensity_intercept, average_slope, average_intercept, slope_err, intercept_err = find_slope_R_difference(intensity_peaks, peak_err)


