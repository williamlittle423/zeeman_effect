import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error
from scipy.stats import chisquare
import os

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
        
def red_chi_sq (y_obs, y_exp):
    chi = chisquare(y_obs, y_exp)[0]
    print(chi)
    return chi/(len(y_obs)-2)
    
def find_slope_R_difference(peak, peak_err, split_indices=split_indices, m_peaks=m_peaks, num_files=num_files):
    a_params = []; b_params = []; sig_a = []; sig_b = []
    plt.figure()
    m_obs = [[], [], [], []]

    #remove the split level peaks
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
        
        #find the slope of the m vs. R^2 data
        p,cov = np.polyfit(x,y,1, cov=True)
        a=p[0]; b=p[1]
        sig_a.append(np.sqrt(cov[0][0])); sig_b .append(np.sqrt(cov[1][1]))
        
        y_est = a * x + b
        
        for j in range(len(m_obs)):
            m_obs[j].append(y[j])
            
        a_params.append(a); b_params.append(b)
        
        #plot the points and their trendlines
        fig = plt.plot(x, y_est)
        plt.errorbar(x, y, yerr = 2*np.sqrt(y)*peak_err, fmt='.', color = fig[-1].get_color(), label=f'Trial {i}')
        
    for i in range(len(m_obs)):
        m_obs[i] = np.mean(m_obs[i])
    #finding average and standard deviation
    average_slope = np.mean(a_params); average_intercept = np.mean(b_params)
    slope_err = np.mean(sig_a); intercept_err = np.mean(sig_b)
    #np.std(a_params); intercept_err = np.std(b_params)
    
    m = np.array([1,2,3,4])
    print(m_obs)
    print(average_slope*m+average_intercept)
    print(red_chi_sq(m_obs, average_slope*m+average_intercept))

    return a_params, b_params, average_slope, average_intercept, slope_err, intercept_err

def plot_BvsI (B_err, I_err):
    '''
    #manually recorded data for magentic field from supplied current
    B = [330, 1120, 1363, 2100, 2730, 3370, 3950, 4520, 5150, 5650, 6240, 6850, 7300, 7750, 8080, 8450, 8750] # Guass
    I = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80] # Amps
    V =[0, 15, 20, 25, 37, 42.5, 55, 65, 75, 85, 95, 105, 115, 125, 135, 145, 155] # V

    #fit a line to the data
    p,cov = np.polyfit(I[:-3], B[:-3], 1, cov=True)
    a=p[0]; b=p[1]
    sig_a = np.sqrt(cov[0][0]); sig_b = np.sqrt(cov[1][1])
    
    y_err = np.sqrt((sig_a*np.array(I))**2 + (I_err*a)**2 + sig_b**2)
    
    hyst_err = root_mean_squared_error(B, a*np.array(I) + b)

    plt.figure(figsize=(12, 6))
    plt.plot(I, a*np.array(I) + b, 'b-', label='Linear fit y = {:.5f}x - {:.2f}'.format(a,-b))
    plt.errorbar(I, B, yerr=B_err, xerr=I_err, fmt='k.', label='Experimental Data')

    plt.ylabel('Magnetic Field (Gauss)'); plt.xlabel('Current (Amps)')
    plt.title('Magnetic vs Current Field')
    plt.legend()
    plt.savefig('current_vs_magnetic_field.png')
    plt.show() '''
    
    
    B_up = [34, 1476, 2700, 3950, 4980, 6230, 7220, 8070, 8740]
    I_up = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    I_down = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    B_down = [8200, 6610, 6150, 5350, 4150, 3010, 1580, 350]
    
    B_tot = B_up + B_down
    I_tot = I_up + I_down
    
    a, b = np.polyfit(I_tot, B_tot, 1)
    p, cov = np.polyfit(I_tot, B_tot, 1, cov=True)
    a=p[0]; b=p[1]
    sig_a = np.sqrt(cov[0][0]); sig_b = np.sqrt(cov[1][1])
    
    #error calculations
    y_err = np.sqrt((sig_a*np.array(I_tot))**2 + (I_err*a)**2 + sig_b**2)
    hyst_err = root_mean_squared_error(B_tot, a*np.array(I_tot) + b)
    
    return a, b, hyst_err, y_err, sig_a, sig_b


intensity_peaks = process_image()

intensity_slope, intensity_intercept, average_slope, average_intercept, slope_err, intercept_err = find_slope_R_difference(intensity_peaks, peak_err)


print(f"$R^2$ = ({average_slope:.2e} +/- {slope_err:.1e}) * m + ({average_intercept:.2e} +/- {intercept_err:.2e}) \n")

B_slope, B_intercept, hyst_err, B_calc_err, B_slope_err, B_intercept_err =  plot_BvsI(B_err, I_err)

print(f"B = ({B_slope:.2e} +/- {B_slope_err:.0e}) * I + ({B_intercept:.2e} +/- {B_intercept_err:.2e}) G")
print(f"Hystersis error: {hyst_err:.2f} G")
print(f"Calculated B error: {np.mean(B_calc_err):.2f} G \n")


#m = 3 peak 
delta_R = []
I = np.linspace(0, 0.8, num_files)

#find values of m=3 and slpit lines
for i in range(num_files):
    if i in (10,11,12,13,14,15,16):
        for j in range(len(intensity_peaks[i])):
            if j in m_peaks[i-8]:
                if (m_peaks[i-8].index(j)+1 == m):
                    left_split = intensity_peaks[i][j-1]**2
                    m_line = intensity_peaks[i][j]**2
                    right_split = intensity_peaks[i][j+1]**2
                    dR = right_split-left_split
                    
    elif i in (8, 9):
        for j in range(len(intensity_peaks[i])):
            if j in m_peaks[i-8]:
                if (m_peaks[i-8].index(j)+1 == m):
                    left_split = intensity_peaks[i][j-1]**2
                    m_line = intensity_peaks[i][j]**2
                    dR = (m_line-left_split)*2
    else:
        dR = 0
    
    delta_R.append(dR)

#remove zero values
delta_R = delta_R[8:]; I = I[8:]
sig_dR = 2*np.sqrt(delta_R)*peak_err

#linear fit
a,b = np.polyfit(I,delta_R,1)
dR_exp = a * I + b

#convert to B
B = B_slope*I + B_intercept

delta_R = np.array(delta_R)
#convert to delta lambda
delta_lambda = lambda_0*delta_R/average_slope

sig_dL = np.sqrt((sig_dR*lambda_0/average_slope)**2 + (slope_err * delta_R*lambda_0/(average_slope**2))**2)

B = B*10**(-4); B_err = B_err*10**(-4)
#linear fit
a,b = np.polyfit(B,delta_lambda,1)
dL_exp = a * B + b

#convert to delta E
delta_E = g*h*c*delta_lambda/(2*t*lambda_0)
sig_E = g*h*c*sig_dL/(2*t*lambda_0)

#linear fit
p,cov = np.polyfit(B,delta_E,1, cov=True)
a=p[0]; b=p[1]
sig_a = np.sqrt(cov[0][0]); sig_b = np.sqrt(cov[1][1])

a_rescaled = a * 1e24    # now 1.23 (if a = 1.23e-24)
sig_a_rescaled = sig_a * 1e24  # now 0.46 (if sig_a = 4.56e-25)

dE_exp = a * B + b

#plot delta E values
plt.figure(figsize=(10, 6))
plt.errorbar(B, delta_E, xerr=B_err, yerr=sig_E, fmt='k.')
#plt.plot(B, delta_E, 'k.')
plt.plot(B, dE_exp, 'r',
         label='$\u0394$E = ({:.2f} ± {:.2f})e-25 B - {:.2e} J'.format(a_rescaled, sig_a_rescaled, -b))
plt.xlabel('B (T)'); plt.ylabel('$\u0394$E (J)')
plt.legend(frameon=False)
plt.savefig('m=3_B_vs_deltaE.png')
plt.show()


print(f"\u0394E = ({a:.2e} +/- {sig_a:.0e}) * B + ({b:.2e} +/- {sig_b:.2e}) J\n")
print(f"Bohr magneton: {a:.2e} +/- {sig_a:.1e} J/T")

print(f"Percent error: {(abs((a-mu_B)/mu_B)*100):.2e}%")

within_err = a < mu_B < (a+sig_a) or a > mu_B > (a-sig_a)

print(f"Falls within error: {within_err}")


