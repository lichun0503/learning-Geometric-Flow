import SimpleITK as sitk
import numpy as np
import os
from scipy import ndimage as nd
import scipy.io as sio
from skimage import io
import skimage
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.metrics import normalized_root_mse
subject_list = [30, 60, 99];
# subject_list = [1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,29,30,31,32,33,35,36,37,38,40,41,42,43,44,45,46,47,48,49,50,51,52,
#                53,54,55,56,57];


image = skimage.io.imread("/home/lichun2020/PycharmProjects/PythonProject/Learning-Variational-Problem/Figures-TSC/Condition-3/Figures-3-00/u_1099.png")
image_noisy = skimage.io.imread("/home/lichun2020/PycharmProjects/PythonProject/Learning-Variational-Problem/Figures-TSC/Condition-3/Figures-3-0.9/u_1099.png")

psnr_noisy = peak_signal_noise_ratio(image, image_noisy)
ssim = ssim(image, image_noisy, multichannel=True)
NRMSE = normalized_root_mse(image, image_noisy)
print("SSIM:", ssim)
print("PSNR:", psnr_noisy)
print("NRMSE:", NRMSE)


