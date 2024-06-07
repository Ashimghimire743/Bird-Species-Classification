import numpy as np
import cv2

def relu(x):
    #max(0,x)
    return x if x>0 else 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def conv2d(input_image, num_filters, kernel_size, filter_kernel, activation="relu", kernel_regularizer=None):
    # Extract input image dimensions
    img_height, img_width, img_channels = input_image.shape
    
    # Extract kernel size
    kernel_height, kernel_width = kernel_size
    
    # Initialize list to store output feature maps
    output_feature_maps = []
    
    # Perform convolution for each filter
    for _ in range(num_filters):
        # Initialize feature map for current filter
        feature_map = np.zeros((img_height - kernel_height + 1, img_width - kernel_width + 1))
        
        # Perform convolution operation
        for i in range(feature_map.shape[0]):
            for j in range(feature_map.shape[1]):
                # Extract current window from input image
                window = input_image[i:i+kernel_height, j:j+kernel_width]
                # Perform element-wise multiplication with filter kernel
                output_pixel = np.sum(window * filter_kernel)
                # Store result in feature map
                feature_map[i, j] = output_pixel
        
        # Apply activation function
        if activation == "relu":
            feature_map = np.maximum(0, feature_map)
        
        # Store feature map in output list
        output_feature_maps.append(feature_map)
    
    # Convert output feature maps to numpy array
    output_feature_maps = np.array(output_feature_maps)
    
    return output_feature_maps



if __name__ == '__main__':
    # print(relu(-10))
    # logits = np.array([[2.0, 1.0, 0.1],
    #                 [0.9, 1.5, 2.5],
    #                 [1.5, 2.0, 1.0]])
    # softmax_output = softmax(logits)
    # print(softmax_output)

    input_image = np.random.rand(500, 500, 1)
    num_filters = 1
    kernel_size = (3, 3)
    activation = "relu"
    kernel_regularizer = None

    # Define the convolution kernel
    filter_kernel = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]])

    # Apply Conv2D layer
    output_feature_maps = conv2d(input_image, num_filters, kernel_size, filter_kernel, activation, kernel_regularizer)

    # Print output feature map
    # print("Output feature map:")
    # print(output_feature_maps[0])

    cv2.imshow('image',output_feature_maps[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

