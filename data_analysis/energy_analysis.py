import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from sklearn.metrics import root_mean_squared_error

num_files = 17

#indices where splitting starts
split_indices = [8, 9, 10, 11, 12, 13, 14, 15, 16]

#indices corresponding to m values
m_peaks = [[0,1,3,4], [0,2,4,5], [0,3,6,9], [0,3,6,9], [0,3,6,9], [0,3,6,9], [1,4,7,10], [1,4,7,10], [1,4,7,10]]

peak_err = 1.57   #sigma R
B_err = 100; I_err = 0.025

lambda_0 = 585e-9  #m
h = 6.626e-34   #J/Hz
c = 3e8   #m/s^2

def process_image(file_pattern='pattern_{}.jpg', num_files=num_files, angle = -35, h_center = 675):
    intensity_peaks = []
    
    for i in range(num_files):
        file_name = file_pattern.format(i)
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
        plt.show() '''
        
        #get the intensity of middle column (vertical line)
        intensity = cropped_image[h_center:, cropped_image.shape[1] // 2]
        
        #find the peaks of the intensity plots
        peaks, _ = find_peaks(intensity, width = 10, distance = 40, prominence=5)
        peaks = list(peaks)
        
        #remove peaks on the fringe
        j = 0
        while j < len(peaks):
            if (peaks[j] < 475 or peaks[j] > 1900):
                peaks.pop(j)
            else: 
                j +=1
        
        '''
        #plot the intensity values
        plt.figure()
        plt.plot(intensity, 'k.')
        plt.title('Vertical Line Pixel Intensity in Centre of Image')
        for p in peaks:
            plt.axvline(x=p, color='red', linestyle='--')
        plt.xlabel('Pixel Position (Y)')
        plt.ylabel('Intensity Value')
        plt.grid()
        plt.show()
        plt.savefig('intensity_trial{}.jpg'.format(i)) '''
        
        intensity_peaks.append(peaks)
        
    return intensity_peaks

def find_slope(peak, peak_err, split_indices=split_indices, m_peaks=m_peaks, num_files=num_files):
    a_params = []; b_params = []
    plt.figure(figsize=(10,8))

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
        a,b = np.polyfit(x,y,1)
        y_est = a * x + b
        
        for j in range(len(peak[i])):
            X.append(j+1)
            Y.append(peak[i][j]**2)
            
        a_params.append(a); b_params.append(b)
        
        #plot the points and their trendlines
        fig = plt.plot(x, y_est)
        plt.errorbar(x, y, yerr = 2*np.sqrt(y)*peak_err, fmt='.', color = fig[-1].get_color(), label=f'trial {i}')
        
    #finding average and standard deviation
    average_slope = np.mean(a_params); average_intercept = np.mean(b_params)
    slope_err = np.std(a_params); intercept_err = np.std(b_params)
    
    m = np.array([1,2,3,4])
    plt.plot(m, average_slope*m+average_intercept, label ='average', linestyle = 'dashed', color = 'k')
    plt.xlabel('m')
    plt.ylabel('R $^2$')
    plt.title('Distance between m level relationship')
    plt.legend(); plt.show()
    plt.savefig('R2_vs_m.jpg')
    
    return a_params, b_params, average_slope, average_intercept, slope_err, intercept_err

def plot_BvsI (B_err, I_err):
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

    plt.ylabel('Magnetic Field (Gauss)')
    plt.xlabel('Current (Amps)')
    plt.title('Magnetic vs Current Field')
    plt.legend(); plt.show()
    plt.savefig('current_vs_magnetic_field.png')
    
    return a,b,hyst_err,y_err

def m_line_current_vs_delta_R(B_slope, B_intercept, B_err, average_slope, slope_err, I_err=I_err, lambda_0=lambda_0):
    #m = 3 peak 
    m = 3
    delta_R = []
    I = np.linspace(0, 0.8, num_files)

    plt.figure()
    
    #find values of m=3 and slpit lines
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

    #remove zero values
    delta_R = delta_R[8:]; I = I[8:]
    sig_dR = 2*np.sqrt(delta_R)*peak_err

    #linear fit
    a,b = np.polyfit(I,delta_R,1)
    dR_exp = a * I + b

    #plot I values
    plt.errorbar(I, delta_R, xerr=I_err, yerr=sig_dR, fmt='k.')
    plt.plot(I, dR_exp, 'r')
    plt.xlabel('I (A)')
    plt.ylabel('${\Delta}$R$^2$')
    plt.title('${\Delta}$R$^2$ over I for m=3')
    plt.show()
    plt.savefig('m=3_I_vs_deltaR^2.png')
    
    #convert to B
    B = B_slope*I + B_intercept; 
    #linear fit
    a,b = np.polyfit(B,delta_R,1)
    dR_exp = a * B + b

    #plot B values
    plt.errorbar(B, delta_R, xerr=B_err, yerr=sig_dR, fmt='k.')
    plt.plot(B, dR_exp, 'r')
    plt.xlabel('B (G)')
    plt.ylabel('${\Delta}$R$^2$')
    plt.title('${\Delta}$R$^2$ over B for m=3')
    plt.show()
    plt.savefig('m=3_B_vs_deltaR^2.png')
    
    
    delta_R = np.array(delta_R); average_slope = np.array(average_slope)
    #convert to delta lambda
    delta_lambda = (lambda_0*delta_R)/(m*average_slope);
    
    sig_dL = np.sqrt((sig_dR*(2*lambda_0*np.sqrt(delta_R))/(m*average_slope))**2 + (slope_err * (lambda_0*delta_R)/(m*average_slope**2))**2)
    
    #linear fit
    a,b = np.polyfit(B,delta_lambda,1)
    dL_exp = a * B + b
    
    #plot delta lambda values
    #plt.errorbar(B, delta_lambda, xerr=B_err, yerr=sig_dL, fmt='k.')
    plt.plot(B, delta_lambda, 'o')
    plt.plot(B, dL_exp, 'r')
    plt.xlabel('B (G)')
    plt.ylabel('${\Delta \lambda}$')
    plt.title('${\Delta \lambda}$ over B for m=3')
    plt.show()
    plt.savefig('m=3_B_vs_deltaL.png')
    
    return a,b


intensity_peaks = process_image()

intensity_slope, intensity_intercept, average_slope, average_intercept, slope_err, intercept_err = find_slope(intensity_peaks, peak_err)

print("m vs. R^2 average slope, intercept:", average_slope, average_intercept)
print("slope, intercept error:", slope_err, intercept_err, "\n")

B_slope, B_intercept, hyst_err, B_calc_err =  plot_BvsI(B_err, I_err)

print("B vs I params:", B_slope, B_intercept)
print("hystersis error:", hyst_err)
print("calculated B error:", np.mean(B_calc_err), '\n')

deltaR_slope, deltaR_intercept = m_line_current_vs_delta_R(B_slope, B_intercept, np.mean(B_calc_err), average_slope, slope_err)

print("m = 3 params:", deltaR_slope, deltaR_intercept)







