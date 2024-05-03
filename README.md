# Variance-Method
Python scripts for XRD line profile analysis using variance method
This script uses variance method to determine dislocation density from xrd peaks. 
xrd data file will be asked in main() and base line corrected. Pseudovoigt function is used to determine gaussian width, w_h and fraction, eta_h.Gaussian width, w_g and fraction, eta_g for instrument need to be provided. Check 5th line in main()
These values are then used to determine dislocation density, strain and crystallite size.
For details of the method, please refer to supplementary file of https://doi.org/10.1016/j.jallcom.2024.174497 
Values need to be proveded/checked: w_g, eta_h, K, L, b, lamda - default values are given
