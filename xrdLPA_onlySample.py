# This script uses variance method to determine dislocation density from xrd peaks. 
# xrd data file will be asked in main() and base line corrected. Pseudovoigt function is used to 
# determine gaussian width, w_h and fraction, eta_h. 
# Gaussian width, w_g and fraction, eta_g for instrument need to be provided. Check 5th line in main()
# These values are then used to determine dislocation density, strain and crystallite size.
# For details of the method, please refer to supplementary file of https://doi.org/10.1016/j.jallcom.2024.174497 
# Values need to be proveded/checked: w_g, eta_h, K, L, b, lamda - default values are given
# 
import numpy as np
import matplotlib.pyplot as plt
import variance_utils
import checkData


pi_value = np.pi
def main():
    
    file_path_sample =  checkData.select_file("Please select the sample data file in .txt") 
    #  get the peak data and correct base line
    corrected_sample_data_x, corrected_sample_data_y, Two_theta_o_sample = base_correct_data(file_path_sample)
    # fit pseudoVoigt function on sample data and get w_h and eta_h
    w_h, eta_h = variance_utils.fit_pseudoVoigt(corrected_sample_data_x, corrected_sample_data_y)
    # calculate Wo and k for sample -refer to the paper above
    Wo_h, k_h = calculate_Wo_k(w_h, eta_h)
    # user need to provide this data for the instrument. fit_pseudoVoigt can be used to get this from instrement 
    # data for the closest peak
    w_g, eta_g = (0.001172343, 0.3)
    Wo_g, k_g = calculate_Wo_k(w_g, eta_g)
    rho, strain, D = get_size_strain(Wo_h,Wo_g,k_h,k_g,Two_theta_o_sample)
    print(f"Wo = {Wo_h} and \n k_h ={k_h}")
    print(f"rho = {rho} \n strain ={strain} \n D= {D}")

def calculate_Wo_k(w, eta):
    beta=w*(pi_value*eta+(1-eta)*np.sqrt(pi_value/np.log(2)))
    W_o =(w**3)*(1/beta)*(0.5*np.sqrt(pi_value)*(1-eta)*((np.log(2))**(-3/2))-pi_value*eta)
    k= 2*eta*w*w/beta
    return W_o, k
def base_correct_data(filename):
    TwoTheta_all, Intensity_all = get_data(filename)
    x, y, o= select_a_peak(TwoTheta_all, Intensity_all)
    base_correct_y = correctBaseline(x,y,o) 
    max_intensity = np.max(base_correct_y)
    normalized_intensity = base_correct_y/max_intensity  
    return x,normalized_intensity,o    

def get_size_strain(Wo_h,Wo_g,k_h,k_g,Two_theta_o_sample):
    
    Wo_f =Wo_h - Wo_g + 0.5*pi_value*pi_value*k_h*k_g
    k_f = k_h-k_g
    # K is variance Scherrer constant and L is tape parameter; both are shape parameters 
    # for sphereical shape: K = 1.2090  and L=0 (K=0.9?)
    K=1.2090
    L=0
    b=2.86e-10
    lamda_Cu = 1.54e-10#(0.000000000154
    lamda = lamda_Cu
    D= (K*lamda)/((pi_value**2)*(np.cos((Two_theta_o_sample*pi_value/180)*k_f)))
    strain = np.sqrt((1/(4*(np.tan(Two_theta_o_sample*pi_value/180))**2))*(Wo_f + (lamda**2)*L/(4*(pi_value**2)*((np.cos(Two_theta_o_sample*pi_value/180)**2)*D**2))))
    # moment = base_correct_y*(delta_theta**2)    
    rho=(3.46*strain)/(D*b)
    return rho, strain, D

def get_data(file):
    with open(file, 'r') as file:
        data = file.read()

    # Assuming data format: col1 col2 col3
    lines = data.split()#('\n')  # Split into lines
    TwoTheta =[]
    Intensity =[]
    for index,line in enumerate(lines):
        x,y = line.split(',')  # Split each line into columns
        x =float(x)
        y=float(y)
        TwoTheta.append(x)
        Intensity.append(y)
        # setxrd.append([x,y]) 
    TwoTheta=np.array(TwoTheta)
    Intensity =np.array(Intensity)
    return TwoTheta, Intensity

def select_a_peak(TwoTheta_all,Intensity_all):
#  It is recommended to first have a look at the xrd profile and get a rough idea of the 2theta
#  values within which the peak lies. Peaks occur at well established values for a material. So 
#  just select a wide enough range which only contains one peak.
    lim_low = 35
    lim_up = 43
    select_peak = (lim_low<=TwoTheta_all[:])&(TwoTheta_all[:]<=lim_up)
    tmp1 = np.where(TwoTheta_all == lim_low)
    peak_centre = TwoTheta_all[tmp1[0][0]+np.argmax(Intensity_all[select_peak])]
    TwoTheta_lower_limit = peak_centre - 3.0
    TwoTheta_upper_limit = peak_centre + 3.0
    select_peak1 = (TwoTheta_lower_limit<=TwoTheta_all[:])&(TwoTheta_all[:]<=TwoTheta_upper_limit)
    peak_x = TwoTheta_all[select_peak1]
    peak_y = Intensity_all[select_peak1]
    return peak_x, peak_y, peak_centre

def correctBaseline(x,y,peak_centre):
    # Here you have to select four points through which the line/polynomial curve is 
    # supposed to pass. In future cubic spline etc options will be available to draw a base line.
    plt.plot(x,y,'-r')
    plt.show()
    plt.pause(0.5)
    inputs = input("Enter the four end points of peak - left&right separated by a space: ")
    A, B, C, D = inputs.split()
    A = float(A)
    index_A = np.where(x ==A)
    index_A = int(index_A[0][0])
    B = float(B)
    index_B = np.where(x ==B)
    index_B = int(index_B[0][0])
    C = float(C)
    index_C = np.where(x ==C)
    index_C = int(index_C[0][0])
    D = float(D)
    index_D = np.where(x ==D)
    index_D = int(index_D[0][0])
    BaseLine_Data_x = np.concatenate((x[index_A:index_B],
                                    x[index_C:index_D]))
    BaseLine_Data_y = np.concatenate((y[index_A:index_B],
                                    y[index_C:index_D]))
    plt.plot(BaseLine_Data_x,BaseLine_Data_y,'-r',marker='o', markersize=1)
    plt.show()
    polynomial_fit_order = 1
    fit_Poly = variance_utils.fit_baseline(BaseLine_Data_x, BaseLine_Data_y,polynomial_fit_order)
    y_corrected = y - fit_Poly(x)
    return y_corrected

if __name__ == "__main__":
    main()



