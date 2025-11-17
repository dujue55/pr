"""
This is the script for Convolution and Filtering part of the assignment.
"""

from typing import Tuple
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt



class Filter():
    """
     Class containing the basic filtering functions
    """
    @staticmethod
    def convolve2d(image: np.ndarray, image_filter: np.ndarray, mode: str ='same') -> np.ndarray:
        """
        To compute the convolution of the image with the filter
        Input:
            image : image of the form K x K, where KxK is the dimension of the image
            imageFilter : filter of the form k x k, where kxk is the dimension of the filter
            mode : mode of the convolution, either 'valid' or 'same'
        Output:
            convolvedImage : convolved image of the form K x K, where KxK is the dimension of the image if mode is 'same' 
            and (K-k+1) x (K-k+1) if mode is 'valid'

        Convolve the image using zero padding and the for loops. For the border mode same, we need to pad the image with zeros.
        This is to ensure that the filter is applied to all the pixels of the image.
        Hint: 
            1. You can use the function np.pad to pad the image with zeros
            2. You can use the function np.flip to flip the filter
            3. The convolution operator requires the filter to be flipped, here.
        Note: In general the filters need not be flipped, but here we need to flip the filter
               to get the correct output and follow the definition of convolution
        """
        #Complete the code here
        # flip the filter
        filter = np.flip(image_filter)
        K = image.shape[0]
        k = filter.shape[0]

        # padding if mode == 'same'
        if mode == 'same': 
            pad = k // 2
            image_padded = np.pad(image,pad_width=pad, mode='constant', constant_values=0)
            output_shape = (K, K)
        else:
            image_padded = image
            output_shape = (K - k + 1, K - k +1)

        # initialize output
        output = np.zeros(output_shape)

        # loop
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                region = image_padded[i : i + k, j : j+k]
                output[i,j] = np.sum(region * filter)

        return output

    @staticmethod
    def convolve2d_fft(image: np.ndarray, image_filter: np.ndarray, mode: str ='same')-> np.ndarray:      
        """
        To compute the convolution of the image with the filter using fft
        Input:
            image : image of the form K x K, where KxK is the dimension of the image
            imageFilter : filter of the form k x k, where kxk is the dimension of the filter
            mode : mode of the convolution, either 'valid' or 'same' 
        Output:
            convolvedImage : convolved image of the form K x K, where KxK is the dimension of the image if mode is 'same'
                                 and (K-k+1) x (K-k+1) if mode is 'valid'
        Convolve the image using fft. For the border mode same, we need to pad the image with zeros. This
        is to ensure the fourier based convolution is same as the for loop based convolution in the boundary regions.
        Hint:
            1. You can use the function np.pad to pad the image with zeros
            2. Fliping the filter is not required here as multiplication in frequency 
                domain implies convolution in spatial domain
            3. Make sure to return the real part of the output
        """

        # Complete the code here
        # 做 FFT 卷积前，一定要：1.对 image 和 filter 都 pad；2.pad 到相同大小 (K+k-1, K+k-1)；3.padding 补在右边和下边；4.然后再 FFT → 乘法 → iFFT → 裁剪。
        # 计算目标大小，防止循环卷积
        K = image.shape[0]
        k = image_filter.shape[0]
        fft_shape = (K + k - 1, K + k - 1)

        # 分别padding
        image_padded = np.pad(image, ((0, fft_shape[0] - K), (0, fft_shape[1] - K)))
        filter_padded = np.pad(image_filter, ((0, fft_shape[0] - k), (0, fft_shape[1] - k)))

        # 分别进行傅里叶
        F_image = np.fft.fft2(image_padded)
        F_filter = np.fft.fft2(filter_padded)

        #频域相乘（对应时域卷积）
        F_result = F_image * F_filter

        # 做逆运算
        conv_full = np.fft.ifft2(F_result)
        conv_full = np.real(conv_full)  # 去掉虚数部分

        # 根据mode剪裁结果
        if mode == 'same':
            start = k // 2
            end = start + K
            conv_result = conv_full[start:end, start:end]
        elif mode == 'valid':
            conv_result = conv_full[k-1:K, k-1:K]
        else:  # 'full'
            conv_result = conv_full
        return conv_result
    

    @staticmethod
    def gaussian_lowpass( eta: float) -> np.ndarray:
        """
        To generate the Gaussian Low Pass Filter
        Input:
            eta : standard deviation of the gaussian distribution
        Output:
            gaussianLowPassFilter : gaussian low pass filter of the form (2m+1) x (2m+1), where (2m+1) x (2m+1) is the dimension of the filter and 
                                    m = 4*eta. IF m is not an integer, then round it to smallest integer greater than m.
        Generate the gaussian low pass filter of the given shape and sigma. 
        Note:
            1.The filter should be normalized such that the sum of all the elements is 1.
            2.The pdf of the Gaussian is sampled at interger values between -m to m, where m is the size of the filter.
            3. We choose odd sized filters to ensure that the filter is symmetric about the center pixel.
        Hint:
            1. You can use the function np.meshgrid to generate the grid and samples along the grid.

        """
        #Complete the code here
        # 1. 根据题意计算 m
        m = int(np.ceil(4 * eta))   # 向上取整，保证是整数
        size = 2 * m + 1            # 滤波器大小为奇数，保证中心像素对称

        # 2. 生成二维网格坐标 n1, n2 ∈ [-m, m]
        n = np.arange(-m, m + 1)
        n1, n2 = np.meshgrid(n, n)

        # 3. 按高斯分布计算滤波器权重
        gaussian = np.exp(-(n1**2 + n2**2) / (2 * eta**2))

        # 4. 归一化（使得所有元素之和为1）
        gaussian /= np.sum(gaussian)

        return gaussian

    

    @staticmethod
    def gaussian_highpass(eta: float) -> np.ndarray:
        """
        To generate the Gaussian High Pass Filter
        Input:
            eta : standard deviation of the gaussian distribution
        Output:
            gaussianHighPassFilter : gaussian high pass filter of the form (2m+1) x (2m+1),  where (2m+1) x (2m+1) is the dimension of the filter and
                                       m = 4*eta. IF m is not an integer, then round it to smallest integer greater than m.
        Hint:
            1.Use the relation gaussianHighPassFilter = delta - gaussianLowPassFilter.
            2. Delta is filter with only the center element as 1 and rest as 0.
        """
        #Complete the code here
        # 1. 先生成低通滤波器
        low_pass = Filter.gaussian_lowpass(eta)

        # 2. 创建 delta（中心为1，其余为0）
        delta = np.zeros_like(low_pass)
        center = low_pass.shape[0] // 2
        delta[center, center] = 1

        # 3. 计算高通滤波器
        high_pass = delta - low_pass

        return high_pass



    


if __name__ == '__main__':

    # Note: AutoGrader will not run this section of the code
    # You can use this to test your code and import any libraries here
    # Autograder doesnt have skimage or scipy installed
    from skimage.data import brick
    from scipy.signal import convolve2d,fftconvolve
    image = brick()

    # plt.imshow(image)
    # plt.show()

    # Test convolve2d
    image_filter = np.array([[1,2,3],[4,5,6],[7,8,9]])
    convolved_image_same = Filter.convolve2d(image,image_filter,mode='same')
    convolved_image_valid = Filter.convolve2d(image,image_filter,mode='valid')

    

    convolved_image_scipy_same = convolve2d(image,image_filter,mode='same')
    convolved_image_scipy_valid = convolve2d(image,image_filter,mode='valid')

    # Difference between your convolve2d and scipy convolve2d should be very small ideally zero
    difference = np.sum(np.abs(convolved_image_same-convolved_image_scipy_same))
    print('Convole2D (same) difference: ',difference)

    difference = np.sum(np.abs(convolved_image_valid-convolved_image_scipy_valid))
    print('Convole2D (valid) difference: ',difference)

    # Test convolve2d_fft
    convolved_image_fft_same = Filter.convolve2d_fft(image,image_filter,mode='same')
    convolved_image_fft_scipy_same = fftconvolve(image,image_filter,mode='same')

    convolved_image_fft_valid = Filter.convolve2d_fft(image,image_filter,mode='valid')
    convolved_image_fft_scipy_valid = fftconvolve(image,image_filter,mode='valid')

    # Difference between your convolve2d_fft and scipy convolve2d_fft should be very small ideally zero
    difference = np.sum(np.abs(convolved_image_fft_same-convolved_image_fft_scipy_same))
    print('Convole2D FFT (same) difference: ',difference)

    difference = np.sum(np.abs(convolved_image_fft_valid-convolved_image_fft_scipy_valid))
    print('Convole2D FFT (valid) difference: ',difference)


    # Test gaussian_lowpass
    eta_Val = 0.2
    gaussian_lowpass_filter = Filter.gaussian_lowpass(eta_Val)
    # Note this is for the case when eta = 0.2
    gaussian_expected = np.array([[1.38877368e-11, 3.72659762e-06, 1.38877368e-11],
       [3.72659762e-06, 9.99985094e-01, 3.72659762e-06],
       [1.38877368e-11, 3.72659762e-06, 1.38877368e-11]])
    

    
    if np.all(np.isclose(gaussian_lowpass_filter, gaussian_expected)):
        print("Passed the gaussian lowpass filter test")

    # Check if it sums to 1
    if np.isclose(np.sum(gaussian_lowpass_filter),1):
        print("Passed the gaussian lowpass filter normalization test")

    # Test gaussian_highpass
    gaussian_highpass_filter = Filter.gaussian_highpass(eta_Val)

    # Note this is for the case when eta = 0.2
    highpass_expected = np.array([[-1.38877368e-11, -3.72659762e-06, -1.38877368e-11],
       [-3.72659762e-06,  1.49064460e-05, -3.72659762e-06],
       [-1.38877368e-11, -3.72659762e-06, -1.38877368e-11]])
    
    if np.all(np.isclose(gaussian_highpass_filter, highpass_expected)):
        print("Passed the gaussian highpass filter test")

    # Check if it sums to 0
    if np.isclose(np.sum(gaussian_highpass_filter),0):
        print("Passed the gaussian highpass filter normalization test")





 