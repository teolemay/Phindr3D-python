"""
Teophile Lemay, 2022-06-14

"""

import numpy as np
import matplotlib.pyplot as plt

#   getImageWithSVMVOverlay.m
def getImageWithSVMVOverlay(IM, param, type):
    """
    IM is rgb 2d image
    param should be taken from Phindr analysis or the desired 
    SV/MV dimensions can be added in another way

    Some default values for the param attributes
    % param.tileX = 10;
    % param.tileY = 10;
    % param.megaVoxelTileX = 5;
    % param.megaVoxelTileY = 5;

    """
    if type == 'SV':
        IM[:param.tileX:, :, :] = 0.7
        IM[:, :param.tileY:, :] = 0.7
    else:
        IM[:param.tileX*param.megaVoxelTileX:, :, :] = 1
        IM[:, :param.tileY*param.megaVoxelTileY:, :] = 1
    return IM


def multiChannelColorImage(images, colors):
    """
    use colors to represent multichannel images in single 2d rgb format image (numpy array)

    images: list or array of single channel 2d images to be combined.
    colors: list or array of rgb colors for each channel
    """
    #each channel -> multiply with rgb color
    colorChans = []
    for i in range(len(images)):
        colorChans.append( np.outer(images[i], colors[i]).reshape( (images[i].shape[0], images[i].shape[1], 3) ) )
    colorChans = np.array(colorChans)

    #add all the channels together to mix the colors
    colorIM = np.sum(colorChans, axis=0)
        
    #rescale for matplotlib to read as rgb -> it works with floats in range [0, 1] or integers in range [0, 255]
    colormins = np.zeros(3)
    colormaxs = np.zeros(3)
    for i in range(3):
        colormins[i] = np.min(colorIM[:, :, i])
        colormaxs[i] = np.max(colorIM[:, :, i])
    colorIM = ((colorIM - colormins)/(colormaxs - colormins) * 255).astype('uint8')  # [0, 255] mode right now.

    # # optional plotting
    # plt.figure(figsize=(10,10))
    # plt.imshow(colorIM)
    # plt.show()
    return colorIM

#   mergeChannels.m
def mergeChannels(imMultiChannel, colors):
    """
    same function as above but the way santosh wrote it in matlab.
    """
    shape = imMultiChannel.shape
    im3D = np.zeros(shape)
    for jChannels in range(0, shape[2]):
        maxVal = np.max(imMultiChannel[:, :, jChannels])
        minVal = np.min(imMultiChannel[:, :, jChannels])
        maxMin = np.abs(maxVal - minVal)
        im = (imMultiChannel[:, :, jChannels] - minVal)/maxMin
        im[im>1] = 1
        im[im<0] = 0
        im3D[:, :, 0] = im3D[:, :, 0] + colors[jChannels, 0]*im
        im3D[:, :, 1] = im3D[:, :, 1] + colors[jChannels, 2]*im
        im3D[:, :, 2] = im3D[:, :, 2] + colors[jChannels, 3]*im
    return im3D

# in matlab this function shows the image in the desired format when selected from the view results window.
def setImageView(imageFileNames, channelColorValue, selectedViewType, selectedChannels, axisHandle):
    """
    sets up viewing for selected images
    has some imshow calls, with parameters changed to show with desired view setting
    low-key is gui related
    """
    imageFileNames = imageFileNames[:, selectedChannels]
    channelColorValue = channelColorValue[selectedChannels.T, :] #transpose propably not needed
    imageInfo = imfinfo(imageFileNames[0, 0])
    (m, n) = imageFileNames.shape
    if m == 1:
        selectedViewType = 'Slice'
    if (selectedViewType == 'Slice') or (selectedViewType == 'MIP'):
        image2Show = np.zeros((imageInfo.height, imageInfo.Width, np.sum(selectedChannels)))
        for i in range(0, n):
            image2Show[:, :, i] = getMIPImage(imageFileNames[:, i])
        image2Show = mergeChannels(image2Show, channelColorValue)
        plt.figure()
        plt.imshow(image2Show)
        plt.show()
    elif selectedViewType == 'Montage':
        image2Show = np.zeros((imageInfo.Height, imageInfo.Width, 3, imageFileNames.shape[0]))
        for i in range(0, m):
            imtmp = np.zeros((imageInfo.height, imageInfo.Width, np.sum(selectedChannels)))
            for j in range(0, n):
                imtmp[:, :, j] = io.imread(imageFileNames[i, j], imageInfo.Format)
            image2Show[:, :, :, i] = mergeChannels(imtmp, channelColorValue)
        plt.figure() #one big figure with many smaller subplots inside could be ok, montage seems to just tile images together ~ similar to subplots
        for i in range(1, m*n): #i think starting at 1 is ok, but will have to see
            plt.subplot(m, n, i)
            plt.imshow(image2Show[:, :, :, i])
            plt.xticks([])
            plt.yticks([])
        plt.show()
    return None
