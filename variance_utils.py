# Fit a cubic spline

import numpy as np
def fit_baseline(x,y,order,num=50):

    # Fit a straight line to the data
    coefficients = np.polyfit(x, y, order)
    polynomial = np.poly1d(coefficients)

    # Generate a fine grid of x values
    # x_fine = np.linspace(min(x), max(x), num)
    # Compute the y values on the fine grid
    # y_fine = polynomial(x_fine)
    return polynomial
    # Plot the original data and the fitted line
    # plt.figure(figsize=(8, 6))
    # plt.plot(x, y, 'o', label='Original data')
    # plt.plot(x_fine, y_fine, '-', label='Fitted line')
    # plt.legend()
    # plt.show()

from scipy.interpolate import CubicSpline
def compute_spline_and_interpolate(x_data, y_data, num_points=50):
    # Fit a cubic spline
    cs = CubicSpline(x_data, y_data)

    # Generate a fine grid of x values
    x_fine = np.linspace(min(x_data), max(x_data), num_points)

    # Compute the y values on the fine grid
    y_fine = cs(x_fine)

    return x_fine, y_fine


# Plot the original data and the spline
# plt.figure(figsize=(8, 6))
# plt.plot(BaseLine_Data_x, BaseLine_Data_y, 'o', label='Original data')
# plt.plot(x_fine, y_fine, '-', label='Cubic spline')
# plt.legend()
# plt.show()

# fit pseudoVoigt
from lmfit.models import PseudoVoigtModel

def fit_pseudoVoigt(x_data, y_data): 
        # Create Pseudo-Voigt model
    mod = PseudoVoigtModel()

    # Guess initial parameters
    pars = mod.guess(y_data, x=x_data)

    # Fit the Pseudo-Voigt curve
    out = mod.fit(y_data, pars, x=x_data)

    # Print fitting report
    print(out.fit_report(min_correl=0.25))

    # Extract fitted parameters
    amplitude = out.params['amplitude'].value
    center = out.params['center'].value
    sigma = out.params['sigma'].value
    eta = out.params['fraction'].value
    fwhm = out.params['fwhm'].value
    height = out.params['height'].value
    # fwhm = 2*w
    return fwhm/2, eta
    # Calculate Lorentzian and Gaussian widths
    # lorentzian_width = 2 * sigma/np.sqrt(2*np.log(2))
    # gaussian_width = 2 * sigma
    # # Calculate areas under the curves
    # area_gaussian = amplitude_gaussian * np.sqrt(2 * np.pi) * sigma
    # area_lorentzian = amplitude_gaussian * 2 * gamma
    # # Calculate fractions
    # total_area = area_gaussian + area_lorentzian
    # fraction_gaussian = area_gaussian / total_area
    # fraction_lorentzian = area_lorentzian / total_area
    # data = [amplitude_gaussian, center,sigma,lorentzian_width,gaussian_width, 
    # area_gaussian, area_lorentzian, fraction_gaussian, fraction_gaussian,total_area]
  
    # print(f"Lorentzian Width: {lorentzian_width:.2f}")
    # print(f"Gaussian Width: {gaussian_width:.2f}")
    # print(f"amplitude: {amplitude_gaussian}")
    # print(f"center: {center}")
    # print(f"sigma: {sigma}")

    # Plot data and fit
    # plt.plot(x_data, y_data, label="Experimental data")
    # plt.plot(x_data, out.best_fit, label="Pseudo-Voigt fit")
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.legend()
    # plt.show()