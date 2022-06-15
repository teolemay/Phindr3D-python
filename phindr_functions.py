"""
Teophile Lemay, 2022

This file contains functions needed to replicate the Phindr3D tool in python. 
code is directly copied/inspired (only modified so that it works in python instead of matlab) from https://github.com/DWALab/Phindr3D .

lets do this.
"""

from mahotas.features import texture
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import time
import tifffile as tf
import skimage.io as io
import glob
from sklearn import cluster
from skimage import feature as feat
import mahotas as mt

#my own functions:

Generator = np.random.default_rng()

def random_cmap(map_len=40, black_background=True):
    """
    this function creates a random color map, useful in segmentation maps

    :param map_len: optional. length of color map. default is 256
    
    :return: random color map.
    """
    from matplotlib import colors
    temp_cmap = np.random.rand(map_len, 3)
    if black_background:
        temp_cmap[0] = 0
    return colors.ListedColormap(temp_cmap)


# start with lib folder:

#classes that seem to be made/needed.
class param_class:
    """
    this class holds the random parameters that seem to be passed along everywhere
    """
    def __init__(self):
        self.tileX = 10
        self.tileY = 10
        self.tileZ = 3
        self.intensityThresholdTuningFactor = 0.5 
        self.numVoxelBins = 20
        self.numSuperVoxelBins = 15
        self.numMegaVoxelBins = 40
        self.minQuantileScaling = .5
        self.maxQuantileScaling = .5
        self.randTrainingSuperVoxel = 10000
        self.superVoxelThresholdTuningFactor = 0.5 #change back to 5 later! 
        self.megaVoxelTileX = 5
        self.megaVoxelTileY = 5
        self.megaVoxelTileZ = 2
        self.countBackground = False
        self.megaVoxelThresholdTuningFactor = .5
        self.pixelsPerImage = 200 #for trainin i guess
        self.randTrainingPerTreatment = 1
        self.randTrainingFields = 5 
        self.startZPlane = 1
        self.endZPlane = 500
        self.numRemoveZStart = 1
        self.numRemoveZEnd = 1
        self.computeTAS = False
        self.showImage = False
        self.showChannels = False
        self.trainingPerColumn = False
        self.intensityNormPerTreatment = False
        self.treatmentCol = ''
        self.trainingColforImageCategories = []
        self.superVoxelPerField = self.randTrainingSuperVoxel//self.randTrainingFields
        self.lowerbound = [0, 0, 0]
        self.upperbound = [1, 1, 1]
        self.numChannels = 3 #keep this here for now since it doesnt seem to be computed early enough in my implementation
        self.svcolormap = random_cmap(map_len=self.numSuperVoxelBins+1)
        self.mvcolormap = random_cmap(map_len=self.numMegaVoxelBins+1)
        self.textureFeatures = False

class par_class:
    def __init__(self):
        self.training = None #unknown
        self.test = None #unknown

#intermediate translation functions (for when python doesnt seem to have a direct translation for a matlab function.)
def mat_dot(A, B, axis=0):
    """
    equvalent to dot product for matlab (can choose axis as well)
    """
    return np.sum(A.conj() * B, axis=axis)

def regexpi(astr, expression, fmt='names'): 
    """ 
    named tokens in matlab regular expressions are equivalent to named groups in python regular expressions.
    regexpi should use re.IGNORECASE bc matlab regexpi is case invariant

    astr is the string to search in
    expression is the set of groups to use for the search.
    """
    groups = re.compile(expression)
    if fmt == 'names':
        m = groups.search(astr, flags=re.IGNORECASE) # i think that this is an appropriate translation of regexpi
        return m.groupdict()
    elif fmt == 'tokenExtents':
        m = groups.search(astr, flags=re.IGNORECASE)
        spans = []
        for group in groups:
            if m.span(group) != (-1, -1):
                spans.append(m.span)
        return np.array(spans)
    elif fmt == 'split':
        return groups.split(astr)

def imfinfo(filename):
    class info:
        pass
    info = info()
    tif = tf.TiffFile(filename)
    file = tif.pages[0]
    metadata = {}
    for tag in file.tags.values():
        metadata[tag.name] = tag.value
    info.Height = metadata['ImageLength']
    info.Width = metadata['ImageWidth']
    info.Format = 'tif'
    return info 


def im2col(img, blkShape):
    #this function is modified from https://github.com/Mullahz/Python-programs-for-MATLAB-in-built-functions/blob/main/im2col.py
    #provides same functionality as matlab's im2col builtin function in distinct mode
    #actuall tested and compared to matlab version this time. produces nice results.
    imgw = img.shape[0]
    imgh = img.shape[1]
    blk_sizew = blkShape[0]
    blk_sizeh = blkShape[1]
  
    mtx = img
    m1c = (imgw*imgh)//(blk_sizew*blk_sizeh)
    m1 = ((blk_sizew*blk_sizeh), m1c)
    blk_mtx = np.zeros(m1)
  
    itr = 0
    for i in range(1,imgw,blk_sizew):
        for j in range(1,imgh,blk_sizeh):
            blk = mtx[i-1:i+blk_sizew-1, j-1:j+blk_sizeh-1].ravel()
            itr = itr+1
            blk_mtx[:,itr-1] = blk
    return blk_mtx


#functions that we seem to need (only start with calculation functions, no gui stuff).

#   extractImageLevelTextureFeatures.m
def extractImageLevelTextureFeatures(mData, param, outputFileName='imagefeatures.csv', outputDir=''):
    if param.countBackground:
        totalBins = param.numMegaVoxelBins + 1
    else:
        totalBins = param.numMegaVoxelBins
    uniqueImageID = np.unique(mData[param.imageIDCol])
    resultIM = np.zeros((len(uniqueImageID), totalBins)) #for all images: put megavoxel frequencies
    resultRaw = np.zeros((len(uniqueImageID), totalBins))
    if param.textureFeatures:
        textureResults = np.zeros((len(uniqueImageID), 4))
    useTreatment=False
    if len(param.treatmentCol) > 0:
        useTreatment = True
        Treatments = []
    timeupdates = len(uniqueImageID)//5
    for iImages in range(len(uniqueImageID)):
        if (iImages == 1) or ((iImages > 3) and ((iImages+1) % timeupdates == 0)):
            print(f'Remaining time estimate ... {round(timeperimage * (len(uniqueImageID)-iImages)/60, 2)} minutes')
        if iImages == 0:
            a = time.time()
        id = uniqueImageID[iImages]
        tmpmdata = mData.loc[mData[param.imageIDCol[0]] == id]
        d = getImageInformation(tmpmdata , param.channelCol[0])
        param = getTileInfo(d, param)
        superVoxelProfile, fgSuperVoxel = getTileProfiles(tmpmdata, param.pixelBinCenters, param)
        megaVoxelProfile, fgMegaVoxel = getMegaVoxelProfile(superVoxelProfile, fgSuperVoxel, param)
        imgProfile, rawProfile, texture_features = getImageProfile(megaVoxelProfile, fgMegaVoxel, param)
        resultIM[iImages, :] = imgProfile
        resultRaw[iImages, :] = rawProfile
        if param.textureFeatures: 
            textureResults[iImages, :] = texture_features
        if useTreatment:
            Treatments.append(tmpmdata[param.treatmentCol].values[0])
        if iImages == 0:
            timeperimage = time.time() - a
    print('Writing data to file ...')
    numRawMV = np.sum(resultRaw, axis=1) #one value per image, gives number of megavoxels
    dictResults = {
        'ImageID':uniqueImageID
    }
    if useTreatment:
        dictResults['Treatment'] = Treatments
    else:
        dictResults['Treatment'] = np.full((len(uniqueImageID), ), 'RR', dtype='object')
    dictResults['NumMV'] = numRawMV
    for i in range(resultIM.shape[1]):
        mvlabel = f'MV{i+1}'
        dictResults[mvlabel] = resultIM[:, i] #e.g. mv cat 1: for each image, put here frequency of mvs of type 1.
    if param.textureFeatures:
        for i, name in enumerate(['text_ASM', 'text_entropy', 'text_info_corr1', 'text_infor_corr2']):
            dictResults[name] = textureResults[:, i]
    df = pd.DataFrame(dictResults)
    csv_name = outputFileName
    if len(outputDir) > 0:
        csv_name = outputDir + '\\' + csv_name
    if csv_name[-4:] != '.csv':
        csv_name = csv_name + '.csv'
    df.to_csv(csv_name) 
    print('\nAll done.')
    return param, resultIM, resultRaw, df #, metaIndexTmp


#   getImageInformation.m
def getImageInformation(tmpdf, chan):
    """called in getPixelBinCenters"""
    """called in extractImageLevelTextureFeatures"""
    """
    gets image dimensions from file names.
    tmpdf is a (truncated) metadata dataframe containg a single shared image ID across all rows
    channelcol is column index of all channel columns
    stackcol is column index of stack column
    """
    imFileName = tmpdf[chan][tmpdf.index[0]]
    #want to get to single filename (first file is convenient.)
    d = np.ones(3, dtype=int)
    d[2] = len(tmpdf)
    info = imfinfo(imFileName) #imfinfo is matlab built-in.
    d[0] = info.Height 
    d[1] = info.Width
    return d

#   getImageProfile.m
def getImageProfile(megaVoxelProfile, fgMegaVoxel, param):
    """called in extractImageLevelTextureFeatures"""
    """
    provides multi-parametric representation of image based on megavoxel categories
    """
    tmp1 = np.array([mat_dot(param.megaVoxelBincenters, param.megaVoxelBincenters, axis=1)]).T 
    tmp2 = mat_dot(megaVoxelProfile[fgMegaVoxel], megaVoxelProfile[fgMegaVoxel], axis=1) 
    a = np.add(tmp1, tmp2).T - (2*(megaVoxelProfile[fgMegaVoxel] @ param.megaVoxelBincenters.T)) 
    minDis = np.argmin(a, axis=1) + 1 
    x = np.zeros(megaVoxelProfile.shape[0], dtype='uint8')
    x[fgMegaVoxel] = minDis
    numbins = param.numMegaVoxelBins
    tmp = np.zeros(numbins+1)
    for i in range(0, numbins+1):
        tmp[i] = np.sum(x[fgMegaVoxel] == (i))
    imageProfile = tmp
    if param.showImage:
        mv_show = np.reshape(x, (param.numMegaVoxelZ, param.numMegaVoxelX, param.numMegaVoxelY))
        for i in range(mv_show.shape[0]):
            plt.figure()
            title = f'Megavoxel image'
            plt.title(title)
            plt.imshow(mv_show[i, :, :], param.mvcolormap)  ###################correct one
            # plt.imshow(mv_show[i, :, :], 'viridis') ############# viridis to compare with matlab easier.
            plt.colorbar()
            plt.show()
            # np.savetxt(r'C:\Users\teole\anaconda3\envs\phy479\pytmvim.csv', mv_show[i, :, :], delimiter=',') ################ pixel pixel comparisons

    if param.textureFeatures: ###########lets put this here. 
        mv_image = np.reshape(x, (param.numMegaVoxelZ, param.numMegaVoxelX, param.numMegaVoxelY))
        total_mean_textures = np.full((param.numMegaVoxelZ, 4), np.nan)
        for i in range(mv_image.shape[0]):
            texture_features = np.full((3, 13), np.nan)
            try:
                texture_features[1, :] = mt.features.haralick(mv_image[i, :, :], distance=1, ignore_zeros=True, return_mean=True)
            except ValueError:
                pass
            try:
                texture_features[1, :] = mt.features.haralick(mv_image[i, :, :], distance=2, ignore_zeros=True, return_mean=True)
            except ValueError:
                pass
            try:
                texture_features[2, :] = mt.features.haralick(mv_image[i, :, :], distance=3, ignore_zeros=True, return_mean=True)
            except ValueError:
                pass
            texture_features = texture_features[:, [0, 8, 11, 12]]
            texture_features = texture_features[~np.isnan(texture_features).any(axis=1), :]
            if len(texture_features) > 1:
                texture_features = np.mean(texture_features, axis=0)
            if texture_features.size > 0:
                total_mean_textures[i, :] = texture_features
        total_mean_textures = total_mean_textures[~np.isnan(total_mean_textures).any(axis=1), :]
        texture_features = np.mean(total_mean_textures, axis=0)
        if texture_features.size == 0:
            param.texture_features = False
            print(f'Texture feature extraction failed for image {imagename}. continuing with default phindr3D')
            texture_features = None
    else:
        texture_features = None
    if not param.countBackground:
        rawProfile = imageProfile[1:].copy()
        imageProfile = imageProfile[1:]
    else:
        rawProfile = imageProfile.copy()
    imageProfile = imageProfile / np.sum(imageProfile) #normalize the image profile
    return imageProfile, rawProfile, texture_features

#   getImageThreshold.m
def getImageThreshold(IM):
    """called in getIndividualChannelThreshold"""
    maxBins = 256
    freq, binEdges = np.histogram(IM.flatten(), bins=maxBins)
    binCenters = binEdges[:-1] + np.diff(binEdges)/2
    meanIntensity = np.mean(IM.flatten())
    numThresholdParam = len(freq)
    binCenters -= meanIntensity
    den1 = np.sqrt((binCenters**2) @ freq.T)
    numAllPixels = np.sum(freq) #freq should hopefully be a 1D vector so summ of all elements should be right.
    covarMat = np.zeros(numThresholdParam)
    for iThreshold in range(numThresholdParam):
        numThreshPixels = np.sum(freq[binCenters > binCenters[iThreshold]])
        den2 = np.sqrt( (((numAllPixels - numThreshPixels)*(numThreshPixels))/numAllPixels) )
        if den2 == 0:
            covarMat[iThreshold] = 0 #dont want to select these, also want to avoid nans
        else:
            covarMat[iThreshold] = (binCenters @ (freq * (binCenters > binCenters[iThreshold])).T) / (den1*den2) #i hope this is the right mix of matrix multiplication and element-wise stuff.
    imThreshold = np.argmax(covarMat) #index makes sense here.
    imThreshold = binCenters[imThreshold] + meanIntensity
    return imThreshold

#   getImageThresholdValues.m
def getImageThresholdValues(mData, param):
    """
    get image threshold values for dataset.  
    """
    intensityThresholdValues = np.full((5000, param.numChannels), np.nan) #not sure why we want 5000 rows
    startVal = 0
    endVal = 0
    for id in param.randFieldID: #for each of the randomly selected images chosen earlier:
        tmpmdata = mData.loc[mData[param.imageIDCol[0]] == id].reset_index(drop=True) #dict of all the slices corresponding to image with imageID ii
        d = getImageInformation(tmpmdata, param.channelCol[0]) #xx is the list of slices.
        param = getTileInfo(d, param)
        iTmp = getIndividualChannelThreshold(tmpmdata, param) #just giving the list of slices for an imageID
        intensityThresholdValues[startVal:endVal+iTmp.shape[0], :] = iTmp
        startVal += iTmp.shape[0]
        endVal += iTmp.shape[0]
    # ii  = (intensityThresholdValues[:, 0] == np.nan) == False #ii is where not nan. Im not sure why they chose to write it like this tho.
    # jj = np.isfinite(intensityThresholdValues)
    param.intensityThresholdValues = intensityThresholdValues[np.isfinite(intensityThresholdValues).any(axis=1)] # remember everything gets rescaled from 0 to 1 #drop rows containing nan, then take medians for each channel#intensityThresholdValues[ii]
    # param.intensityThresholdValues = intensityThresholdValues[jj]
    return param
 
#   getIndividualChannelThreshold.m
def getIndividualChannelThreshold(tmpmdata, param):
    """called in getImageThresholdValues""" 
    """
    tmpmdata is the truncated metadata dataframe for a single image id.
    """
    thresh = np.zeros((len(tmpmdata), param.numChannels))
    if param.intensityNormPerTreatment:
        grpVal = np.argwhere(param.allTreatments == tmpmdata[param.treatmentCol].values[0])
    for iImages in range(len(tmpmdata)):
        for iChannels in range(param.numChannels):
            IM = io.imread(tmpmdata.loc[iImages, param.channelCol[iChannels]]) 
            xEnd = -param.xOffsetEnd
            if xEnd == -0:
                xEnd = None   #if the end index is -0, you just index from 1 to behind 1 and get an empty array. change to 0 if the dimOffsetEnd value is 0.
            yEnd = -param.yOffsetEnd
            if yEnd == -0:
                yEnd = None
            IM = IM[param.xOffsetStart:xEnd, param.yOffsetStart:yEnd]
            if param.intensityNormPerTreatment:
                IM = rescaleIntensity(IM, low=param.lowerbound[grpVal, iChannels], high=param.upperbound[grpVal, iChannels])
            else:
                IM = rescaleIntensity(IM, low=param.lowerbound[iChannels], high=param.upperbound[iChannels])
            thresh[iImages, iChannels] = getImageThreshold(IM.astype('float64')) #want double precision here. not sure if python can handle this since rounding error occurs at 1e-16, but will make float64 anyway
    #they choose to clear IM here, but if its getting overwritten every for loop, its probably fine.
    return thresh

######never used.
#   getIntensityFeatures.m 

#never used
#   getMIPImage.m

#   getMegaVoxelBinCenters.m
def getMegaVoxelBinCenters(mData, param):
    """
    compute bincenters for megaVoxels

    % mData  - Metadata
    % allImageID - Image IDs of each image stack
    % param - All parameters
    % Output
    % param - Appended parameters
    """
    MegaVoxelsforTraining = []
    for id in param.randFieldID:
        tmpmdata = mData.loc[mData[param.imageIDCol[0]] == id]
        d = getImageInformation(tmpmdata, param.channelCol[0])
        param = getTileInfo(d, param)
        superVoxelProfile, fgSuperVoxel = getTileProfiles(tmpmdata, param.pixelBinCenters, param)
        megaVoxelProfile, fgMegaVoxel = getMegaVoxelProfile(superVoxelProfile, fgSuperVoxel, param)
        if len(MegaVoxelsforTraining) == 0:
            MegaVoxelsforTraining = megaVoxelProfile[fgMegaVoxel]
        else:
            MegaVoxelsforTraining = np.concatenate((MegaVoxelsforTraining, megaVoxelProfile[fgMegaVoxel]))
    param.megaVoxelBincenters = getPixelBins(MegaVoxelsforTraining, param.numMegaVoxelBins)
    # visualization
    if param.showBincenters:
        from phindr_clustering import sammon
        try:
            S, E = sammon(MegaVoxelsforTraining, 2)
            sam=True
        except ValueError:
            sam = False
            from sklearn.decomposition import KernelPCA
            pca = KernelPCA(n_components=2, kernel='rbf')
            S = pca.fit(MegaVoxelsforTraining).transform(MegaVoxelsforTraining)
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_title('Training mega-voxel distribution and bin centers')
        ax.scatter(S[:, 0], S[:, 1], color='tab:blue', label='Training mega-voxels')
        if sam:
            from scipy.spatial.distance import cdist
            for i in range(param.megaVoxelBincenters.shape[0]):
                closest_index = np.argmin(cdist(MegaVoxelsforTraining, np.atleast_2d(param.megaVoxelBincenters[i])))
                if i == 0:
                    ax.scatter(S[closest_index, 0], S[closest_index, 1], color='r', label='Approx. mega voxel bin centers')
                else:
                    ax.scatter(S[closest_index, 0], S[closest_index, 1], color='r')
                ax.text(S[closest_index, 0], S[closest_index, 1], f'{i+1}', zorder=1, color='k')
        else:
            bc = pca.transform(param.megaVoxelBincenters)
            ax.scatter(bc[:, 0], bc[:, 1], color='r', label='Bin centers')
            for i in range(param.megaVoxelBincenters.shape[0]):
                ax.text(bc[i, 0], bc[i, 1], f'{i+1}', zorder=1, color='k')
        ax.legend()
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        plt.show()
    return param

#   getMegaVoxelProfile.m
def getMegaVoxelProfile(tileProfile, fgSuperVoxel, param):
    """called in extractImageLevelTextureFeatures"""
    """called in getMegaVoxelBinCenters"""
    temp1 = np.array([mat_dot(param.supervoxelBincenters, param.supervoxelBincenters, axis=1)]).T
    temp2 = mat_dot(tileProfile[fgSuperVoxel], tileProfile[fgSuperVoxel], axis=1)
    a = np.add(temp1, temp2).T - 2*(tileProfile[fgSuperVoxel] @ param.supervoxelBincenters.T)
    minDis = np.argmin(a, axis=1) + 1 #mindis+1 here
    x = np.zeros(tileProfile.shape[0], dtype='uint8') #x is the right shape
    x[fgSuperVoxel] = minDis
    #had to change x shape here from matlab form to more numpy like shape. 
    x = np.reshape(x, (int(param.croppedZ/param.tileZ), int(param.croppedX/param.tileX), int(param.croppedY/param.tileY))) #new shape (z, x, y)
    if param.showImage:
        for i in range(x.shape[0]):
            plt.figure()
            title = f'Supervoxel image'
            plt.title(title)
            plt.imshow(x[i, :, :], param.svcolormap) ##############################This is the correct one
            # plt.imshow(x[i, :, :], 'viridis') #########viridis to compare to matlab map
            plt.colorbar()
            plt.show()
            # np.savetxt(r'C:\Users\teole\anaconda3\envs\phy479\pytsvim.csv', x[i, :, :], delimiter=',') ################ pixel pixel comparisons

    #pad first dimension 
    x = np.concatenate([ np.zeros((param.superVoxelZAddStart, x.shape[1], x.shape[2])), x, np.zeros((param.superVoxelZAddEnd, x.shape[1], x.shape[2])) ], axis=0) #new (z, x, y) shape
    #pad second dimension
    x = np.concatenate([ np.zeros((x.shape[0], param.superVoxelXAddStart, x.shape[2])), x, np.zeros((x.shape[0], param.superVoxelXAddEnd, x.shape[2])) ], axis=1) #new (z, x, y) shape
    #pad third dimension
    x = np.concatenate([ np.zeros((x.shape[0], x.shape[1], param.superVoxelYAddStart)), x, np.zeros((x.shape[0], x.shape[1], param.superVoxelYAddEnd)) ], axis=2) #for new (z, x, y) shape
    x = x.astype(np.uint8)
    param.numMegaVoxelX = x.shape[1]//param.megaVoxelTileX
    param.numMegaVoxelY = x.shape[2]//param.megaVoxelTileY
    param.numMegaVoxelZ = x.shape[0]//param.megaVoxelTileZ
    param.numMegaVoxelsXY = int(x.shape[1] * x.shape[2] / (param.megaVoxelTileY * param.megaVoxelTileX)) #for new shape
    param.numMegaVoxels = int((param.numMegaVoxelsXY*x.shape[0])/param.megaVoxelTileZ)
    sliceCounter = 0
    startVal = 0
    endVal = param.numMegaVoxelsXY
    try:
         megaVoxelProfile = np.zeros((param.numMegaVoxels, param.numSuperVoxelBins+1))
    except Exception as e:
        print(e)
    fgMegaVoxel = np.zeros(param.numMegaVoxels)
    tmpData = np.zeros((param.numMegaVoxelsXY, int(param.megaVoxelTileX*param.megaVoxelTileY*param.megaVoxelTileZ)))
    startCol = 0
    endCol = (param.megaVoxelTileX*param.megaVoxelTileY)
    for iSuperVoxelImagesZ in range(0, x.shape[0]):
        sliceCounter += 1
        tmpData[:, startCol:endCol] = im2col(x[iSuperVoxelImagesZ, :, :], (param.megaVoxelTileX, param.megaVoxelTileY)).T #changed which axis is used to iterate through z.
        startCol += (param.megaVoxelTileX*param.megaVoxelTileY)
        endCol += (param.megaVoxelTileX*param.megaVoxelTileY)
        if sliceCounter == param.megaVoxelTileZ:
            fgMegaVoxel[startVal:endVal] = (np.sum(tmpData!= 0, axis=1)/tmpData.shape[1]) >= param.megaVoxelThresholdTuningFactor
            for i in range(0, param.numSuperVoxelBins+1):
                megaVoxelProfile[startVal:endVal, i] = np.sum(tmpData == i, axis=1) #value of zeros means background supervoxel
            sliceCounter = 0
            tmpData = np.zeros((param.numMegaVoxelsXY, param.megaVoxelTileX*param.megaVoxelTileY*param.megaVoxelTileZ))
            startCol = 0
            endCol = (param.megaVoxelTileX*param.megaVoxelTileY)
            startVal += param.numMegaVoxelsXY
            endVal += param.numMegaVoxelsXY
    if not param.countBackground:
        megaVoxelProfile = megaVoxelProfile[:, 1:]
    megaVoxelProfile = np.divide(megaVoxelProfile, np.array([np.sum(megaVoxelProfile, axis=1)]).T) #dont worry about divide by zero here either
    fgMegaVoxel = fgMegaVoxel.astype(bool) 
    return megaVoxelProfile, fgMegaVoxel

# Never used
#   getMerged3DImage.m

#   getPixelBinCenters.m
def getPixelBinCenters(mData, param):
    """
    compute bincenters for pixels

    #bincenters should be pixel categories

    allImageId is image ids of each image stack
    mData is metadata.
    param should be some parameter class object.
    """
    pixelsForTraining = np.zeros((300000, param.numChannels)) # long array [channel 1[very long zeros...], channel2[very long zeros...], channel3[very long zeros ...]] by 3 channels
    startVal = 0
    endVal = 0
    for id in param.randFieldID: #for each i in range (length of training image set)  #image name
        tmpmdata = mData.loc[mData[param.imageIDCol[0]] == id]
        d = getImageInformation(tmpmdata, param.channelCol[0])
        param = getTileInfo(d, param)
        param.randZForTraining = len(tmpmdata)//2 #want to take half (floor) of the z slices
        iTmp = getTrainingPixels(tmpmdata, param) #load correct 3d 3 channel image. should be 3 channels with flatenned image in each.
        pixelsForTraining[startVal:endVal+iTmp.shape[0], :] = iTmp #add to list.
        startVal += iTmp.shape[0]
        endVal += iTmp.shape[0]
    pixelsForTraining = pixelsForTraining[np.sum(pixelsForTraining, axis=1) > 0, :] #this step gets rid of trailing zeros left over at the end.
    param.pixelBinCenters = getPixelBins(pixelsForTraining, param.numVoxelBins) 
    ## end of function. everything between here and return is to show the bincenters. ## 

    if param.showBincenters:
        if param.numChannels == 3:#should be 3. changed here to test something #3d projection, otherwise back to sammon mapping.
            fig = plt.figure()
            ax = fig.add_subplot(projection = '3d')
            ax.set_title('Training voxel distribution')
            ch1 = pixelsForTraining[:, 0]
            ch2 = pixelsForTraining[:, 1]
            ch3 = pixelsForTraining[:, 2]
            ax.scatter(ch1, ch2, ch3)
            ax.set_xlabel('Channel 1 intensity')
            ax.set_ylabel('Channel 2 intensity')
            ax.set_zlabel('Channel 3 intensity')

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(projection = '3d')
            ax2.set_title('Voxel bin center distribution')
            ch1 = param.pixelBinCenters[:, 0]
            ch2 = param.pixelBinCenters[:, 1]
            ch3 = param.pixelBinCenters[:, 2]
            for i in range(len(param.pixelBinCenters)):
                ax2.scatter(ch1[i], ch2[i], ch3[i])
                ax2.text(ch1[i], ch2[i], ch3[i], f'{i+1}', zorder=1,  color='k')
            ax2.set_xlabel('Channel 1 intensity')
            ax2.set_ylabel('Channel 2 intensity')
            ax2.set_zlabel('Channel 3 intensity')
            plt.show()
        else:
            from phindr_clustering import sammon
            try:
                S, E = sammon(pixelsForTraining, 2)
                sam=True
            except ValueError:
                sam=False
                from sklearn.decomposition import KernelPCA
                pca = KernelPCA(n_components=2, kernel='rbf')
                S = pca.fit(pixelsForTraining).transform(pixelsForTraining)
                print(S.shape)
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Training Voxel distribution and bin centers')
            ax.scatter(S[:, 0], S[:, 1], color='tab:blue', label='Training Voxels')
            if sam:
                from scipy.spatial.distance import cdist
                for i in range(param.supervoxelBincenters.shape[0]):
                    closest_index = np.argmin(cdist(pixelsForTraining, np.atleast_2d(param.pixelBinCenters[i])))
                    ax.scatter(S[closest_index, 0], S[closest_index, 1], color='r')
                    ax.text(S[closest_index, 0], S[closest_index, 1], f'{i+1}', zorder=1, color='k')
            else:
                bc = pca.transform(param.pixxelBinCenters)
                ax.scatter(bc[:, 0], bc[:, 1], color='r', label='Bin centers')
                for i in range(param.pixelBinCenters.shape[0]):
                    ax.text(bc[i, 0], bc[i, 1], f'{i+1}', zorder=1, color='k')
            ax.legend()
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            plt.show()
    return param

#   getPixelBins.m
def getPixelBins(x, numBins):
    """called in getPixelBinCenters"""
    """called in getSuperVoxelBinCenters"""
    """called in getMegaVoxelBinCenters"""
    """
    %getPixelBins Get pixel centers from training images
    % For each voxel, assign categories

    % Inputs:
    % x - m x n (m is the number of observations and n could be number of channels or category fractions)
    % numBins - Number of categories

    % Outputs
    % binCenters - (numBins+1) x n (The first centroid are zeros- indicating background)

    % numBins = param.numVoxelBins;
    % Use kmeans clustering to get  (looks like it is using kmeans++ algorithm) # use sklearn kmeans (gives same results as matlab with enough repeats!)
    """
    m = x.shape[0]
    if m > 50000:
        samSize = 50000
    else:
        samSize = m
    if m > samSize:
        numRandRpt = 10
        binCenters = np.zeros((numBins, x.shape[1], numRandRpt))
        sumD = np.zeros(numRandRpt)
        for iRandCycle in range(0, numRandRpt):
            randpermX = np.array([x[j] for j in Generator.choice(m, size=samSize, replace=False, shuffle=False)  ])
            kmeans = cluster.KMeans(n_clusters=numBins, init='k-means++', n_init=100, max_iter=100).fit(randpermX) #max_iter used to be 100. changed because bin-centers don't always match up to real values.
            binCenters[:, :, iRandCycle] = kmeans.cluster_centers_
            temp1 = np.add(np.array([mat_dot(binCenters[:, :, numRandRpt-1], binCenters[:, :, numRandRpt-1], axis=1)]).T, mat_dot(x, x, axis=1)).T #still not sure which one of this or the next should be transposed
            temp2 = 2*(x @ binCenters[:, :, numRandRpt-1].T)
            a = (temp1 - temp2)
            sumD[iRandCycle] = np.sum(np.amin(a, axis=1))
        minDis = np.argmin(sumD)
        binCenters = binCenters[:, :, minDis]
    else: 
        kmeans = cluster.KMeans(n_clusters=numBins, init='k-means++', n_init=100, max_iter=100).fit(x) #max iter used to be 100
        binCenters = kmeans.cluster_centers_ 
    return np.abs(binCenters)

#   getScalingFactorforImages.m
def getScalingFactorforImages(metadata, param):
    """
    compute lower and higher scaling values for each image
    param: structure of parameter value
    metadata: Metadata

    """
    randFieldIDforNormalization = getTrainingFields(metadata, param) #choose images for scaling
    if param.intensityNormPerTreatment:
        grpVal = np.zeros(randFieldIDforNormalization.size)
    minChannel = np.zeros((randFieldIDforNormalization.size, param.numChannels)) #min values of all selected images in all channels
    maxChannel = np.zeros((randFieldIDforNormalization.size, param.numChannels)) #max values of all selected images in all channels
    numImages = randFieldIDforNormalization.size
    for i in range(0, numImages):
        # which images 
        id = randFieldIDforNormalization[i] # which 3d image
        tmpmdata = metadata.loc[metadata[param.imageIDCol[0]]==id]
        zStack = np.ravel(tmpmdata[param.stackCol])
        depth = len(zStack)
        #used to be a getTileInfo here.
        randHalf = int(depth//2 )
        randZ = [zStack[j] for j in Generator.choice(depth, size=randHalf, replace=False, shuffle=False)] #choose half of the stack, randomly
        minVal = np.zeros((randHalf, param.numChannels))
        maxVal = np.zeros((randHalf, param.numChannels))
        for j in range(randHalf): 
            for k in range(param.numChannels):
                IM = io.imread(tmpmdata.loc[tmpmdata[param.stackCol[0]]==randZ[j], param.channelCol[k]].values[0]) 
                minVal[j, k] = np.quantile(IM, 0.01)
                maxVal[j, k] = np.quantile(IM, 0.99)
        minChannel[i, :] = np.amin(minVal, axis=0)
        maxChannel[i, :] = np.amax(maxVal, axis=0)
        if param.intensityNormPerTreatment:
            #index of the treatment for this image in the list of all treatment
            grpVal[i] = np.argwhere(param.allTreatments == tmpmdata[param.treatmentCol].values[0]) #tmpdata[param.treatmentCol[0]][0] is the treatment of the current image
    if param.intensityNormPerTreatment:
        uGrp = np.unique(grpVal)
        param.lowerbound = np.zeros((uGrp.size, param.numChannels))
        param.upperbound = np.zeros((uGrp.size, param.numChannels))
        for i in range(0, uGrp.size):
            ii = grpVal == uGrp[i]
            if np.sum(ii) > 1:
                param.lowerbound[i, :] = np.quantile(minChannel[grpVal == uGrp[i], :], 0.01)
                param.upperbound[i, :] = np.quantile(maxChannel[grpVal == uGrp[i], :], 0.99)
            else:
                param.lowerbound[i, :] = minChannel[grpVal == uGrp[i], :]
                param.upperbound[i, :] = maxChannel[grpVal == uGrp[i], :]
    else:
        param.lowerbound = np.quantile(minChannel, 0.01, axis = 0)
        param.upperbound = np.quantile(maxChannel, 0.99, axis = 0)
    param.randFieldID = randFieldIDforNormalization #added this here because I dont know where else this would be determined.
    return param

#   getSuperVoxelBinCenters.m
def getSuperVoxelBinCenters(mData, param):
    """
    compute bin centers for super voxels
    % mData  - Metadata
    % allImageID - Image ID's of each image stack
    % param - All parameters
    """
    param.pixelBinCenterDifferences = np.array([mat_dot(param.pixelBinCenters, param.pixelBinCenters, axis=1)]).T  # do the array of list of array trick to increase dimensionality of pixelBinCenterDiff  so that the transpose is actually different from the original array. 
    tilesForTraining = []   #(cont. above) this trick lets me broadcast it together with another array of different length in np.add()
    for id in param.randFieldID: # for each 3d 3 channel image in the training set
        tmpmdata = mData.loc[mData[param.imageIDCol[0]]== id]
        d = getImageInformation(tmpmdata, param.channelCol[0])
        param = getTileInfo(d, param)
        superVoxelProfile, fgSuperVoxel = getTileProfiles(tmpmdata, param.pixelBinCenters, param)
        tmp = superVoxelProfile[fgSuperVoxel] 
        if tmp.size != 0:
            if len(tilesForTraining) == 0:
                tilesForTraining = tmp
            if param.superVoxelPerField > tmp.shape[0]:
                tilesForTraining = np.concatenate((tilesForTraining, tmp))
            else: 
                tmp2Add = np.array([tmp[i, :] for i in Generator.choice(tmp.shape[0], size=param.superVoxelPerField, replace=False, shuffle=False)])
                tilesForTraining = np.concatenate((tilesForTraining, tmp2Add))
    if len(tilesForTraining) == 0:
        print('\nNo foreground super-voxels found. consider changing parameters')
    param.supervoxelBincenters = getPixelBins(tilesForTraining, param.numSuperVoxelBins)
    #below is just visualization.
    if param.showBincenters:
        toobig2show = False
        from phindr_clustering import sammon
        try:
            S, E = sammon(tilesForTraining, 2)
            sam = True
        except ValueError or MemoryError:
            sam=False
            try:
                from sklearn.decomposition import KernelPCA
                pca = KernelPCA(n_components=3, kernel='rbf')
                S = pca.fit(tilesForTraining).transform(tilesForTraining)
            except MemoryError:
                toobig2show = True
                print('Super voxel array too big for visualization')
        if sam and (not toobig2show):
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title('Training super-voxel distribution and bin centers')
            ax.scatter(S[:, 0], S[:, 1], color='tab:blue', label='Training super voxels')
            from scipy.spatial.distance import cdist
            for i in range(param.supervoxelBincenters.shape[0]):
                closest_index = np.argmin(cdist(tilesForTraining, np.atleast_2d(param.supervoxelBincenters[i])))
                ax.scatter(S[closest_index, 0], S[closest_index, 1], color='r')
                ax.text(S[closest_index, 0], S[closest_index, 1], f'{i+1}', zorder=1, color='k')
        elif not toobig2show:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_title('Training super-voxel distribution')
            ax.scatter(S[:, 0], S[:, 1], S[:, 2], color='tab:blue', label='Training super voxels')
            ax.set_zlabel('PCA 3')

            fig2 = plt.figure()
            ax2 = fig2.add_subplot(projection='3d')
            ax2.set_title('Super voxel bin centers')
            bc = pca.transform(param.supervoxelBincenters)
            bc1 = bc[:, 0]
            bc2 = bc[:, 1]
            bc3 = bc[:, 2]
            for i in range(param.supervoxelBincenters.shape[0]):
                ax2.scatter(bc1[i], bc2[i], bc3[i])
                ax2.text(bc1[i], bc2[i], bc3[i], f'{i+1}', zorder=1, color='k')
            ax2.set_xlabel('PCA 1')
            ax2.set_ylabel('PCA 2')
            ax2.set_zlabel('PCA 3')
            ax.legend()
        if not toobig2show:
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            plt.show()
    return param

#   getTileInfo.m
def getTileInfo(dimSize, param):
    """called in getPixelBinCenters"""
    """called in extractImageLevelTextureFeatures"""
    """called in getImageThresholdValues"""
    """called in getSuperVoxelbinCenters"""
    """
    computes how many pixels and stacks that need to be retained based on user choices.
    """
    xOffset = dimSize[0] % param.tileX
    yOffset = dimSize[1] % param.tileY
    zOffset = dimSize[2] % param.tileZ

    if xOffset % 2 == 0:
        param.xOffsetStart = int(xOffset/2 + 1)-1 #remember 0 indexing in python
        param.xOffsetEnd = int(xOffset/2)
    else:
        param.xOffsetStart = int(xOffset//2 + 1)-1
        param.xOffsetEnd = int(-(-xOffset//2 ))  #ceiling division is the same as upside-down floor division. 
    if yOffset % 2 == 0:
        param.yOffsetStart = int(yOffset/2 + 1)-1
        param.yOffsetEnd = int(yOffset/2)
    else:
        param.yOffsetStart = int(yOffset//2 + 1)-1
        param.yOffsetEnd = int(-(-yOffset//2 )) 
    if zOffset % 2 == 0:
        param.zOffsetStart = int(zOffset/2 + 1)-1
        param.zOffsetEnd = int(zOffset/2)
    else:
        param.zOffsetStart = int(zOffset//2 + 1)-1
        param.zOffsetEnd = int(-(-zOffset//2 ) )   

    param.croppedX = dimSize[0] - param.xOffsetStart - param.xOffsetEnd
    param.croppedY = dimSize[1] - param.yOffsetStart - param.yOffsetEnd
    param.croppedZ = dimSize[2] - param.zOffsetStart - param.zOffsetEnd
    
    superVoxelXOffset = (param.croppedX/param.tileX) % param.megaVoxelTileX
    superVoxelYOffset = (param.croppedY/param.tileY) % param.megaVoxelTileY
    superVoxelZOffset = (param.croppedZ/param.tileZ) % param.megaVoxelTileZ
    param.origX = dimSize[0]
    param.origY = dimSize[1]
    param.origZ = dimSize[2]

    if superVoxelXOffset % 2 == 0:
        param.superVoxelXOffsetStart = superVoxelXOffset/2 + 1
        param.superVoxelXOffsetEnd = superVoxelXOffset/2
    else:
        param.superVoxelXOffsetStart = superVoxelXOffset//2 + 1
        param.superVoxelXOffsetEnd = -(-superVoxelXOffset//2) #same floor division trick.
    if superVoxelXOffset != 0: #add pixel rows if size of supervoxels are not directly visible
        numSuperVoxelsToAddX = param.megaVoxelTileX - superVoxelXOffset
        if numSuperVoxelsToAddX % 2 == 0:
            param.superVoxelXAddStart = int(numSuperVoxelsToAddX/2)
            param.superVoxelXAddEnd = int(numSuperVoxelsToAddX/2)
        else:
            param.superVoxelXAddStart = int(numSuperVoxelsToAddX // 2)
            param.superVoxelXAddEnd = int(-(-numSuperVoxelsToAddX // 2 ))
    else:
        param.superVoxelXAddStart = int(0)
        param.superVoxelXAddEnd = int(0)
    #same along other axes.
    if superVoxelYOffset != 0:
        numSuperVoxelsToAddY = param.megaVoxelTileY - superVoxelYOffset
        if numSuperVoxelsToAddY % 2 == 0:
            param.superVoxelYAddStart = int(numSuperVoxelsToAddY/2)
            param.superVoxelYAddEnd = int(numSuperVoxelsToAddY/2)
        else:
            param.superVoxelYAddStart = int(numSuperVoxelsToAddY //2)
            param.superVoxelYAddEnd = int(-(- numSuperVoxelsToAddY //2))
    else:
        param.superVoxelYAddStart = int(0)
        param.superVoxelYAddEnd = int(0)
    if superVoxelZOffset != 0:
        numSuperVoxelsToAddZ = param.megaVoxelTileZ - superVoxelZOffset
        if numSuperVoxelsToAddZ % 2 == 0:
            param.superVoxelZAddStart = int(numSuperVoxelsToAddZ/2)
            param.superVoxelZAddEnd = int(numSuperVoxelsToAddZ/2)
        else:
            param.superVoxelZAddStart = int(numSuperVoxelsToAddZ //2)
            param.superVoxelZAddEnd = int(-(-numSuperVoxelsToAddZ//2))
    else:
        param.superVoxelZAddStart = int(0)
        param.superVoxelZAddEnd = int(0)
    #continue first part of supervoxels offset parity with other axes
    if superVoxelYOffset % 2 == 0:
        param.superVoxelYOffsetStart = int(superVoxelYOffset/2 + 1) - 1 
        param.superVoxelYOffsetEnd = int(superVoxelYOffset/2)
    else:
        param.superVoxelYOffsetStart = int(superVoxelYOffset//2 + 1) - 1
        param.superVoxelYOffsetEnd = int(-(-superVoxelYOffset//2))
    if superVoxelZOffset % 2 ==0:
        param.superVoxelZOffsetStart = int(superVoxelZOffset/2 + 1) - 1
        param.superVoxelZOffsetEnd = superVoxelZOffset/2
    else:
        param.superVoxelZOffsetStart = int(superVoxelZOffset//2 +1) - 1
        param.superVoxelZOffsetEnd = int(-(-superVoxelZOffset//2)) 

    param.numSuperVoxels = (param.croppedX*param.croppedY*param.croppedZ)//(param.tileX*param.tileY*param.tileZ) #supposed to be all elementwise operations (floor division too)
    param.numSuperVoxelsXY = (param.croppedX*param.croppedY)/(param.tileX*param.tileY)

    tmpX = (param.croppedX/param.tileX) + superVoxelXOffset
    tmpY = (param.croppedY/param.tileY) + superVoxelYOffset
    tmpZ = (param.croppedZ/param.tileZ) + superVoxelZOffset

    param.numMegaVoxels = int((tmpX*tmpY*tmpZ) // (param.megaVoxelTileX*param.megaVoxelTileY*param.megaVoxelTileZ))
    param.numMegaVoxelsXY = int((tmpX*tmpY)/(param.megaVoxelTileX*param.megaVoxelTileY))
    
    return param

#   getTileProfiles.m
def getTileProfiles(tmpmdata, pixelBinCenters, param):
    """called in extractImageLevelTextureFeatures"""
    """called in getMegaVoxelBinCenters"""
    """called in getSuperVoxelBinCenters"""
    """
    computes low level categorical features for supervoxels
    function assigns categories for each pixel, computes supervoxel profiles for each supervoxel
    % Inputs:
    % filenames - Image file names images x numchannels
    % pixelBinCenters - Location of pixel categories: number of bins x number
    % of channels
    param: parameter object
    ii: current image id
    % Output:
    % superVoxelProfile: number of supervoxels by number of supervoxelbins plus
    % a background
    % fgSuperVoxel: Foreground supervoxels - At lease one of the channles
    % should be higher than the respective threshold
    % TASScores: If TAS score is selected
    """
    numTilesXY = int((param.croppedX*param.croppedY)/(param.tileX*param.tileY)) #why not just use param.numSuperVoxelsXY, I Have not idea. the calculation is the exact same.
    zEnd = -param.zOffsetEnd
    if zEnd == -0:
        zEnd = None

    slices = np.sort(tmpmdata[param.stackCol[0]].values) 
    slices = slices[param.zOffsetStart:zEnd] #keep z stacks that are divisible by stack count
    sliceCounter = 0
    startVal = 0
    endVal=numTilesXY
    startCol= 0
    endCol = param.tileX*param.tileY
    if param.intensityNormPerTreatment:
        grpVal = np.argwhere(param.allTreatments == tmpmdata[param.treatmentCol].values[0])
    superVoxelProfile = np.zeros((param.numSuperVoxels, param.numVoxelBins+1))
    fgSuperVoxel = np.zeros(param.numSuperVoxels)
    if param.computeTAS:
        categoricalImage = np.zeros((param.croppedX, param.croppedY, param.croppedZ))
    #loop over file names and extract super voxels
    tmpData = np.zeros((numTilesXY, int(param.tileX*param.tileY*param.tileZ))) #dimensions: number of supervoxels in a 2D cropped image x number of voxels in a supervoxel.
    #tmpData holds the binned pixel image (ONE LAYER OF SUPERVOXELS AT A TIME.), but in a weird format right now.
    for iImages, zslice in enumerate(slices):
        sliceCounter += 1
        croppedIM = np.zeros((param.origX, param.origY, param.numChannels)) #just one slice in all channels
        for jChan in range(param.numChannels):
            try:
                if param.intensityNormPerTreatment:
                    croppedIM[:,:, jChan] = rescaleIntensity(io.imread(tmpmdata.loc[tmpmdata[param.stackCol[0]] == zslice, param.channelCol[jChan]].values[0], 'tif'), low=param.lowerbound[grpVal, jChan], high=param.upperbound[grpVal, jChan])
                else:
                    croppedIM[:,:, jChan] = rescaleIntensity(io.imread(tmpmdata.loc[tmpmdata[param.stackCol[0]] == zslice, param.channelCol[jChan]].values[0], 'tif'), low=param.lowerbound[jChan], high=param.upperbound[jChan])
            except Exception as e:
                print(e)
                print('Error: file ->', filenames[zslice][jChan+1])
        xEnd = -param.xOffsetEnd
        if xEnd == -0:
            xEnd = None   #if the end index is -0, you just index from 1 to behind 1 and get an empty array. change to 0 if the dimOffsetEnd value is 0.
        yEnd = -param.yOffsetEnd
        if yEnd == -0:
            yEnd = None
        #crop image to right dimensions for calculating supervoxels
        croppedIM = croppedIM[param.xOffsetStart:xEnd, param.yOffsetStart:yEnd, :] #z portion of the offset has already been done by not loading the wrong slices

        if param.showImage: 
            if param.showChannels or param.numChannels != 3:
                fig, ax = plt.subplots(1, int(param.numChannels))
                for i in range(param.numChannels):
                    ax[i].set_title(f'Channel {i+1}')
                    ax[i].imshow(croppedIM[:, :, i], 'gray')
                    ax[i].set_xticks([])
                    ax[i].set_yticks([])
            elif param.numChannels == 3:
                plt.figure()
                title = f'slice {zslice}'
                plt.title(title)
                plt.imshow(croppedIM) #leaving it in multichannel gives rgb correctly for 3 channel image. WILL Fail for numChannel != 3
            plt.show()

        x = np.reshape(croppedIM, (param.croppedX*param.croppedY, param.numChannels)) #flatten image, keeping channel dimension separate   
        fg = np.sum(x > param.intensityThreshold, axis=1) >= 1 #want to be greater than threshold in at least 1 channel
        pixelCategory = np.argmin(np.add(param.pixelBinCenterDifferences, mat_dot(x[fg,:], x[fg,:], axis=1)).T - 2*(x[fg,:] @ pixelBinCenters.T), axis=1) + 1
        x = np.zeros(param.croppedX*param.croppedY, dtype='uint8')
        x[fg] = pixelCategory #assign voxel bin categories to the flattened array

        ## uncomment for testing if needed.
        # x_show = np.reshape(x, (param.croppedX, param.croppedY))
        # np.savetxt(r'C:\Users\teole\anaconda3\envs\phy479\pytvoxelim.csv', x_show, delimiter=',')

        #here, x can be reshaped to croppedX by croppedY and will give the map of pixel assignments for the image slice
        if param.computeTAS:
            categoricalImage[:, :, iImages] = np.reshape(x, param.croppedX, param.croppedY)
        # del fg, croppedIM, pixelCategory #not 100 on why to delete al here since things would just be overwritten anyway, but why not right, also, some of the variables to clear where already commented out so I removed them from the list
        if sliceCounter == param.tileZ:
            #add the tmpData that has been accumulating for the past  to the fgsupervoxel
            fgSuperVoxel[startVal:endVal] = (np.sum(tmpData != 0, axis=1)/tmpData.shape[1]) >= param.superVoxelThresholdTuningFactor
            for i in range(0, param.numVoxelBins+1):
                superVoxelProfile[startVal:endVal, i] = np.sum(tmpData == i, axis=1) #0 indicates background
            #reset for next image
            sliceCounter = int(0)
            startVal += numTilesXY
            endVal += numTilesXY
            startCol = 0
            endCol = param.tileX*param.tileY
            tmpData = np.zeros((numTilesXY, param.tileX*param.tileY*param.tileZ))
        else:
            tmpData[:, startCol:endCol] = im2col(np.reshape(x, (param.croppedX, param.croppedY)), (param.tileX, param.tileY)).T
            startCol += (param.tileX*param.tileY)
            endCol += (param.tileX*param.tileY)
    if not param.countBackground:
        superVoxelProfile = superVoxelProfile[:, 1:]
    superVoxelProfile = np.divide(superVoxelProfile, np.array([np.sum(superVoxelProfile, axis=1)]).T) #dont worry about divide by zero errors, they are supposed to happen here!
    superVoxelProfile[superVoxelProfile == np.nan] = 0
    fgSuperVoxel = fgSuperVoxel.astype(bool)
    return superVoxelProfile, fgSuperVoxel ##fgSuperVoxel used to be fgSuperVoxel.T

#   getTrainingfields.m
def getTrainingFields(metadata, param):
    """
    called in getScalingFactorforImages

    get smaller subset of images (usually 10) to define parameters for further analysis
        (nly used for scaling factors to scale down intensities from 0 to 1)
    """
    uniqueImageID = np.unique(metadata[param.imageIDCol])
    if not param.intensityNormPerTreatment:
        randFieldID = np.array([uniqueImageID[i] for i in Generator.choice(uniqueImageID.size, size=param.randTrainingFields, replace=False, shuffle=False)])
    else:
        #have different treatments, want to choose training images from each treatment.
        uTreat = np.unique(metadata[param.treatmentCol].values)
        numtreatments = len(uTreat)
        param.randTrainingPerTreatment = -(-param.randTrainingFields//numtreatments) #ceiling division
        randFieldID = []
        for treat in uTreat:
            treatmentIDs = np.unique(metadata.loc[metadata[param.treatmentCol[0]] == treat, param.imageIDCol])  #all image IDs corresponding to this treatment
            randFieldID = randFieldID + [treatmentIDs[j] for j in Generator.choice(len(treatmentIDs), size=param.randTrainingPerTreatment, replace=False, shuffle=False)]
        randFieldID = np.array(randFieldID)
    return randFieldID 

#   getTrainingPixels.m
def getTrainingPixels(tmpmdata, param):
    """called in getPixelBinCenters"""
    """
    tmpmdata is truncated metadata dataframe, limited to a single image id
    param is param file
    ii is the image ID for the image that was passed in.
    """
    slices = tmpmdata[param.stackCol[0]].values
    #uncomment this row for proper function.
    slices = np.array([slices[i] for i in Generator.choice(len(slices), size=param.randZForTraining, replace=False, shuffle=False)]) #shuffle the slices
    trPixels = np.zeros((param.pixelsPerImage*param.randZForTraining, param.numChannels))
    startVal = 0
    if param.intensityNormPerTreatment:
        grpVal = np.argwhere(param.allTreatments == tmpmdata[param.treatmentCol].values[0])
    slices = slices[0:(len(slices)//2)] ###################### revert by removing the :3]#
    for zplane in slices:
        croppedIM = np.zeros((param.origX, param.origY, param.numChannels))
        for jChan in range(param.numChannels):
            if param.intensityNormPerTreatment:
                croppedIM[:,:, jChan] = rescaleIntensity(io.imread(tmpmdata.loc[tmpmdata[param.stackCol[0]] == zplane, param.channelCol[jChan]].values[0], 'tif'), low=param.lowerbound[grpVal, jChan], high=param.upperbound[grpVal, jChan])
            else:
                croppedIM[:,:, jChan] = rescaleIntensity(io.imread(tmpmdata.loc[tmpmdata[param.stackCol[0]] == zplane, param.channelCol[jChan]].values[0], 'tif'), low=param.lowerbound[jChan], high=param.upperbound[jChan])
        #crop parameters
        xEnd = -param.xOffsetEnd
        if xEnd == -0:
            xEnd = None   #if the end index is -0, you just index from 1 to behind 1 and get an empty array. change to 0 if the dimOffsetEnd value is 0.
        yEnd = -param.yOffsetEnd
        if yEnd == -0:
            yEnd = None
        croppedIM = croppedIM[param.xOffsetStart:xEnd, param.yOffsetStart:yEnd, :] #crop
        croppedIM = np.reshape(croppedIM, (param.croppedX*param.croppedY, param.numChannels)) #flatten
        croppedIM = croppedIM[np.sum(croppedIM > param.intensityThreshold, axis=1) >= param.numChannels/3, :] #must be greater than threshold in at least one channel.
        croppedIM = selectPixelsbyweights(croppedIM)
        if croppedIM.shape[0] >= param.pixelsPerImage:
            trPixels[startVal:startVal + param.pixelsPerImage, :] = np.array([croppedIM[i, :] for i in Generator.choice(croppedIM.shape[0], size=param.pixelsPerImage, replace=False, shuffle=False) ])
            startVal += param.pixelsPerImage
        else:
            trPixels[startVal:(startVal+croppedIM.shape[0])] = croppedIM
            startVal += croppedIM.shape[0]
    if trPixels.size == 0:
        trPixels = np.zeros((param.pixelsPerImage*param.randZForTraining, param.numChannels))
    return trPixels

#   initParameters.m
def initParameters():
    """
    initiates parameters and returns structured array
    """
    param = param_class()
    return param

#   rescaleIntensity.m
def rescaleIntensity(im, low=0, high=1):
    """called in getIndividualChannelThreshold"""
    """called in getTileProfiles"""
    """
    rescales intensity of image based on lower and upper bounds
    """
    im = im.astype(np.float64)
    diffIM = high - low
    im = (im - low)/diffIM
    im[im>1] = 1
    im[im<0] = 0
    return im

#   selectPixelsbyweight.m
def selectPixelsbyweights(x):
    """called in getTrainingPixels"""
    n, bin_edges = np.histogram(x, bins=(int(1/0.025) + 1), range=(0,1), )
    q = np.digitize(x, bin_edges)
    n = n / np.sum(n)
    p = np.zeros(q.shape)
    for i in range(0, n.shape[0]):
        p[q==i] = n[i]
    p = 1 - p
    p = np.sum(p>np.random.random((q.shape)), axis=1) #q shape may or may not be correct
    p = p != 0
    p = x[p, :]
    return p

####  NEVER USED ############
#   setImageView.m  

#   getGroupIndices.m

#NOT Called.
#   getImageIDfromMetadata.m

# never used
#   AssignTilestoBins.m

# never used
#   equalTrainingSamplePartition.m

# never used/not needed by me
#   getExtractedMetadata.m


## never used
#   findClosestPoint2Line.m

## never used
#   getDistancePointFromLine.m

# Never called
#   getImageWithSVMVOverlay.m

## never used
#   writestr.m

#never used (i think it's related to getting specific image from the view results window.)
#   getImage2Display.m

#need to find these ones

# def getPlateInfoFromMetadatafile(metadataFilenameLabel, param):
#     """called in getIntensityFeatures"""
#     """function not defined in lib, third party/clustering, organoidCSApp folders"""
#     return metadataLabel, unknown, unknown, imageIDLabel

#never used
# getCategoricalTASScores


#gui function to fill in later:

#   PhinDR3D_Main.m

#   PhindrViewer.m

#   colorpicker.m

#   findjobj.m
#      very intense gui stuff. apparently matlab gui is built on java, this matlab function does works through handles, other gui things 
#      straight up gui building, user interactions, etc.

#   imageViewer.m

#   metadataExtractor.m

#   setChannelInformation

#   setParameterValues.m
























