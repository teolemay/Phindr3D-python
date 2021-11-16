"""
Teophile Lemay, 2021

This file contains functions needed to replicate the Phindr3D tool in python. 
code is directly copied/inspired (only modified so that it works in python instead of matlab) from https://github.com/DWALab/Phindr3D .

lets do this.

repository structure:
    executables:
        installation .exe files
    lib:
        lots of files
    manuals:
        phindr3D manual .pdf
        organoid contour segmentation manual .pdf
    phindr3d-organoidCSApp:
        files for organoid contour segmentation
    ThirdParty/Clustering:
        clustering algos.
"""

"""
metadata format notes.upper():

mData[ii, 0] is the same as fileNames.


"""

"""
FORMAT TRANSLATION NOTES:

rules of thumb from stackOverflow:
    matlab array -> python numpy array

    matlab cell array -> python list

    matlab structure -> python dict


if indexing doesnt work properly for indexing with some other logical array, should try indexing with np.ix_()

VERY POSSIBLY, I SHOULD USE DICTIONARIES / ORDERED DICTIONARIESTO REPLACE THE "CELL" COMPONENTS USED TO HOLD METADATA.

ALSO POSSIBLE THAT ALL THE THINGS CURRENTLY SET AS CLASSES (PARAM, PAR, C) SHOULD ALSO BE REPLACED WITH DICTIONARIES
"""

import numpy as np
from numpy.core.fromnumeric import size
import scipy as sc
import pandas as pd
import time
import matplotlib.pyplot as plt
import re
import os
import tifffile as tf
import skimage.io as io
import cv2 as cv

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
        self.intensityThresholdTuningFactor = .5
        self.numVoxelBins = 20
        self.numSuperVoxelBins = 15
        self.numMegaVoxelBins = 40
        self.minQuantileScaling = .5
        self.maxQuantileScaling = .5
        self.randTrainingSuperVoxel = 10000
        self.superVoxelThresholdTuningFactor = .5
        self.megaVoxelTileX = 5
        self.megaVoxelTileY = 5
        self.megaVoxelTileZ = 2
        self.countBackground = False
        self.megaVoxelThresholdTuningFactor = .5
        self.pixelsPerImage = 200
        self.randTrainingPerTreatment = 1
        self.randTrainingFields = 5
        self.showImage = 0
        self.startZPlane = 1
        self.endZPlane = 500
        self.numRemoveZStart = 1
        self.numRemoveZEnd = 1
        self.computeTAS = 0
        self.showImage = 0
        self.trainingPerColumn = False
        self.intensityNormPerTreatment = False
        self.treatmentColNameForNormalization = ''
        self.trainingColforImageCategories = ''
        self.superVoxelPerField = self.randTrainingSuperVoxel//self.randTrainingFields
    

class C_class:
    def __init__(self):
        self.minClsSize = 5
        self.maxCls = 10
        self.minCls = 1
        self.S = []
        self.pmin = 0
        self.pmax = 0
        self.pmed = 0

class par_class:
    def __init__(self):
        self.training = None #unknown
        self.test = None #unknown

#intermediate translation functions (for when python doesnt seem to have a direct translation for a matlab function.)
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

def im2col(arr, shape, type='distinct'): # this implementation inspired from https://stackoverflow.com/questions/25449279/efficient-implementation-of-im2col-and-col2im
    """
    im2col is a matlab builtin function
    param arr: 2D array 
    param shape: tuple, kernel shape (rows, cols)
    param type: kernel motion type (distinct or sliding kernels)
    """
    if type == 'distinct':
        nrows = shape[0]
        ncols = shape[1]
        nele = nrows*ncols
        row_ext = arr.shape[0] % nrows #need to padd array if kernel doesnt fit nicely into array shape
        col_ext = arr.shape[1] % ncols
        A1 = np.zeros((arr.shape[0]+row_ext, arr.shape[1]+col_ext))
        A1[:arr.shape[0], :arr.shape[1]] = arr

        t1 = np.reshape(A1, (nrows, A1.shape[0]/nrows, -1))
        t1tilde = np.transpose(t1.copy(), (0, 2, 1))
        t2 = np.reshape(t1tilde, (t1.shape[0]*t1.shape[2], -1))
        t3 = np.transpose(np.reshape(t2, (nele, t2.shape[0]/nele, -1)), (0, 2, 1))
        return np.reshape(t3, (nele, -1))

#NOT done with this one yet
x =  #bad code here to make unfinished function easy to find in long file.
y = 
z = 
def kmeans(x, *args):
    """called in getpixelBins"""
    #matlab has a built-in k means clustering function. this is a placeholder until I choose how to write it in python.
    # bin centers is: "the k cluster centroid locations in the k-by-p matrix C." as written at https://www.mathworks.com/help/stats/kmeans.html#bues2hs
    return something_discarded, binCenters





#functions that we seem to need (only start with calculation functions, no gui stuff yet).

#   AssignTilestoBins.m
def assignTilestoBins(tiledImage, binCenters):
    """
    assigns tiles to bins
    """
    numBins = binCenters.shape[0] 
    numTiles, numPixelTile, numChannels = tiledImage.shape
    tiledImageDis = np.zeros((numTiles, numPixelTile, numBins))
    for iBins in range(numBins):
        tmp = np.tile(binCenters[iBins, :], (numPixelTile, 1)) # i believe this is the correct translation.
        tmp =  tiledImage - tmp #apply operation @minus element wise to tiledIMage and tmp; minus(A, B) = A - B
        tmp = tmp**2
        tiledImageDis[:, :, iBins] = sum(tmp, axis=2)
    
    I = np.argmin(tiledImageDis, axis=2) #seems like we want the indices of the minimums along axis=2
    tiledImageDis = I
    return tiledImageDis 

#   clsIn.m
def clsIn(data, beta=0.05, dis='euclidean'):
    """
    dis can be: "euclidean" OR "cosine" OR "hamming"
    """
    dis = dis.lower()
    if data.size == 0: #check if data is empty array
        print('Data is empty')
        return None
    C = C_class()
    C.minClsSize = 5
    C.maxCls = 10
    C.minCls = 1
    C.S = []
    C.pmin = 0
    C.pmax = 0
    C.pmed = 0
    sim = sc.spatial.distance.cdist(data, data, dis)
    if dis == 'euclidean':
        sim = -1*sim
    elif (dis == 'cosine') or (sim == 'hamming'):
        sim = 1 - sim
    x_x = np.tril(np.ones((sim.shape[0], sim.shape[0]), -1, dtype=bool)) #lower triangular matrix True below the diagonal, false elsewhere.
    C.pmed = np.median(sim[x_x], axis=1) #i think this is the right axis, but very unsure. Should be median along rows of 2d matrix since sim[x_x] is 2d matrix however, numpy rows and cols are different than in matlab.
    C.pmin, C.pmax = preferenceRange(sim)
    C.S = sim
    return C

#   computeClustering.m
def computeClustering(data, numberClusters, type='AP'):
    C = clsIn(data)
    if type == 'AP':
        clusterResult = apclusterK(C.S, numberClusters)
    else:
        clusterResult = np.arange(0, data.shape[0])
    return clusterResult

#   equalTrainingSamplePartition.m
def equalTrainingSamplePartition(groups, samplesize):
    par = par_class()
    par.training = np.full((groups.size, 1), False)
    par.test = np.full((groups.size, 1), False)
    uniqueGroups = np.unique(groups[groups > 0])[0]
    for iGroups in range(0, uniqueGroups.size):
        ind = np.argwhere(groups == uniqueGroups[iGroups])
        p = np.random.Generator.permutation(ind.size)
        par.training[ind[p[:samplesize]], 0] = True #may or may not have to remove the ,0 at the end.
        par.test[ind[p[samplesize:]], 0] = True
    return par

#   extractImageLevelTextureFeatures.m
def extractImageLevelTextureFeatures(mData, allImageId, param, outputFileName, outputDir):
    param.correctshade = 0
    if param.countBackground:
        totalBins = param.numMegaVoxelBins + 1
    else:
        totalBins = param.numMevaVoxelBins
    uniqueImageID = np.unique(allImageId)[0]
    resultIM = np.zeros((uniqueImageID.size, totalBins))
    resultRaw = np.zeros((uniqueImageID.size, totalBins))
    metaIndexTmp = [[[] for j in range(mData.shape[1])] for i in range(uniqueImageID.size)] #nested lists seems like a ok but not the cleanest way to do cell()
    averageTime = 0
    for iImages in range(0, uniqueImageID.size):
        tImageAnal = time.time() #want to time this for some reason
        ii = allImageId == uniqueImageID[iImages]
        d, param.fmt = getImageInformation( mData[ii, 0] )
        param = getTileInfo(d, param)
        tmpInfoTable = mData[ii, 0:param.numchannels] #this type of indexing might not work.
        superVoxelProfile, fgSuperVoxel = getTileProfiles(tmpInfoTable, param.pixelBinCenters, param, ii)
        megaVoxelProfile, fgMegaVoxel = getMegaVoxelProfile(superVoxelProfile, fgSuperVoxel, param)
        resultIM[iImages, :], resultRaw[iImages] = getImageProfile(megaVoxelProfile, fgMegaVoxel, param)
        tmp = mData[allImageId == uniqueImageID, :] #indexing may be wrong here too.
        for k in range(mData.shape[1]):
            metaIndexTmp[iImages][k] = tmp[1][k]
        averageTime = averageTime + (time.time() - tImageAnal)
        print('time remaining:', (uniqueImageID.size -iImages-1)*(averageTime/(iImages+1)), 's')  #time left update i guess
    numRawMV = np.sum(resultRaw, axis=1)
    dataHeaderIM = [[] for l in range(resultIM.shape[1])]
    for i in range(0, resultIM.shape[1]):
        dataHeaderIM[0][i] = f'MV {i}'
    #want to remove ImageID if existis in Metadata
    ii = param.metaDataHeader == 'ImageID'
    param.metaDataHeader = param.metaDataHeader
    #write outputs to file. pd.DataFrame would be VERY good for this part. have to figure out how to integrate properly
    print('NEED TO PROPERLY FORMAT OUTPUT TO FILE HERE!!!!')

    return resultIM, resultRaw, metaIndexTmp

#   findClosestPoint2Line.m
def findClosestPoint2Line(pts, v1, v2):
    d = np.zeros((pts.shape[0], 1))
    if pts.shape[1] == 2:
        pts = np.array([pts, d])
    for i in range(0, pts.shape[0]):
        d[i] = getDistancePointFromLine(pts[i, :], v1, v2)
    closestIndex = np.argmin(d)
    return closestIndex

#   getBestPreference.m
def getBestPreference(x, y, pl=False):
    """
    % %getBestPreference Perform knee point detection
    % to get the best clusters
    % "Knee Point Detection in BIC for Detecting the Number of Clusters"
    % Input:
    %       x - X axis values
    %       y - y axis values
    %       pl - toggle plotting option
    """
    yp = 0 #i think that these are supposed to be indices, so i set to 0. if not, then it should be 1
    xp = np.argwhere(y ==1)
    if x.shape[0] != y.shape[0]:
        print('Error')
        return None
    ys = y
    pp = 3
    maxabd = np.abs(y[2:] + y[0:-2] - (2*y[1:-1]))
    ix = np.zeros((maxabd.shape[0], 1))
    uMaxabd = np.unique(maxabd)[::-1].sort() #sort the array in descending order. This format should work
    uMaxabd = uMaxabd[1:]
    cnt = 0
    for i in range(0, uMaxabd.size):
        ii = np.argwhere(maxabd == uMaxabd[i])
        ix[cnt:cnt+ii.size]  = ii[::-1].sort() #i dont think i need to put -1 after cnt + ii.size bc python omits last index already and I feel like matlab doesnt.
        cnt += ii.size
    n = x.size//2
    ix = ix[0:n]
    ix = ix + 1
    mangle = np.zeros((n, 1))
    for i in range(0, n):
        if ix[i] > 1:
            sl1 = np.divide( (y[ix[i]] - y[ix[i]-1]), (x[ix[i]] - x[ix[i]-1]) ) #pretty sure this is right. we should be doing proper matrix division
            sl2 = np.divide( (y[ix[i]+1] - y[ix[i]]), (x[ix[i]+1] - x[ix[i]]) ) #same here.
            mangle[i] = np.arctan( np.abs((sl1 + sl2)/(1 - (sl1*sl2))) )
    maxMangle = np.max(mangle)
    uI = mangle == maxMangle
    im = np.min(ix[uI])
    ii = im-1
    xp = x[ii]
    yp = ys[ii]
    y = ys
    #plotting here
    if pl:
        xCent = np.min(x) + (np.max(x) + np.min(x))/2
        yCent = (y[-1] - y[0])/2
        optText = f'Estimated Optimal Cluster -- {yp}'
        plt.figure()
        plt.plot(x, y, '-r', markercolor='r', lable='# Clusters')
        plt.plot(xp, yp, '-bo', lable='Optimal Cluster')
        plt.ylabel('Number of clusters')
        plt.xlabel('Preference')
        plt.text(xCent, yCent, optText)
        plt.legend()
    return yp, xp

#   getDistancePointFromLine.m
def getDistancePointFromLine(pt, v1, v2):
    """called in findClosestPoint2Line"""
    a = v1 - v2
    b = pt - v2
    d = np.linalg.norm(np.divide(np.cross(a, b), np.norm(a)))
    return d

#NOT FINISHED WITH THIS ONE YET.
#   getExtractedMetadata.m
def getExtractedMetadata(pth, astr, exprIM, opFilename):
    """
    function creates metadata from folder i think.

    need better understanding of what actually is supposed to be in the metadata. maybe would be usefull to understand what gets parsed from metadata.
    """
    m = regexpi(str, exprIM, 'Names') #want name and text of each named token. 
    #If str and expression are both character vectors or string scalars, 
    # the output is a 1-by-n structure array, where n is the number of matches. 
    # The structure field names correspond to the token names.

    if len(m) == 0:
        print('Regular expression mismatch')
        return None

#   getGroupIndices.m
def getGroupIndices(textData, grpNames, Narg=None):
    if Narg == None:
        output = True
    else:
        output = False
    if (len(textData) == 0) or (len(grpNames) == 0):
        print('No data')
        return None
    numGrps = len(grpNames)
    grps = np.zeros((len(textData), 1))
    for i in range(numGrps):
        p = grpNames[i] == textData[:, 0]
        idx = np.nonzero(p)
        if output:
            print(f'NUmber of Points in {grpNames[i]} is {idx}')
        if idx.size == 0:
            pass
        grps[idx, 0] = i
    return grps
    
#NOT FINISHED WITH THIS ONE YET. HAS SOME DETAILS ABOUT METADATA CELL FORMAT THAT COULD BE USEFUL.
#   getImage2Display.m
def getImage2Display(mData, metaHeader, param, ImageID2View, imageID, chanInfo, stack2View, typeofView):
    """
    select image based on specific image ID (support for GUI program.)

    % mData -Metadata cell array consisting of individual image information
    %           in each row
    %
    % metaHeader - Associated metadata header
    %
    % param - Parameter file (Structure variable)
    % 
    % imageID2View - Image id selected to view
    % 
    % imageID - List of all image ID. Column vector
    % 
    % chanInfo - Structure variable having specific information about channels
    % 
    % stack2View - If viewing single slice then the stack number to view
    % typeOfView - Montage, MIP. Any other choice defaults to single plane
     """
    #  get image from image Id, sort z stacks, norm things, show image projection based on user choices.
    x = 
    y =
    z = 
    return image2Display

#   getImageIDfromMetadata.m
def getImageID(metadata, rootDir, exprIM):
    """
    gets unique image ID
    """
    uCol = metadata[:, 0] #first column is always one of the channels
    astr = uCol[10, :] #just pick any row (what does this mean??)
    astr = re.sub(rootDir, '', astr) #replace all text in astr that matches with rootDir with '' (nothing).
    m1 = regexpi(astr, exprIM, 'names') #I think this returns a dictionary of {name:token text, ...}
    ff = m1.keys() #list of fieldnames (dictionary keys) 
    ff = [key.lower() for key in ff]#make lowercase for easier list comprehension.
    try:
        stackNuminRegexp = ff.index('stack')
    except ValueError:
        stackNuminRegexp = ff.index('stacks')
    for i in range(uCol.size):
        astr = re.sub(rootDir, '', uCol[i, :])
        mm = regexpi(astr, exprIM, fmt='tokenExtents')
        mm = mm[0, 0]
        mm = mm[stackNuminRegexp, :]
        uCol[i, :] = astr[0:mm[0]-1, mm[1]+1:] #may be wrong here.
    uUCol = np.unique(uCol)[0]
    imageIDcol = np.zeros((metadata.shape[0], 1))
    cnt = 1
    numUCol = uUCol.size
    for i in range(0, numUCol):
        ii = uCol == uUCol[i, :]
        imageIDcol[ii, 0] = cnt
        cnt += 1
    return imageIDcol

#   getImageInformation.m
def getImageInformation( fileNames ):
    """called in getPixelBinCenters"""
    """called in extractImageLevelTextureFeatures"""
    """
    gets image dimensions from file names.
    """
    if len(fileNames) == 0:
        print('Fi,e name empty')
    d = np.ones(1, 3)
    fmt = 'tif'
    imFileName = np.unique(fileNames[:, 0])[0]  #could be wrong
    imFileName = imFileName[0, 0] # i think we just want the name of the first file.
    d[0, 2] = fileNames.shape[0]
    info = imfinfo(imFileName) #imfinfo is matlab built-in.
    d[0, 0] = info.Height
    d[0, 1] = info.width
    return d, fmt 

#   getImageProfile.m
def getImageProfile(megaVoxelProfile, fgMegaVoxel, param):
    """called in extractImageLevelTextureFeatures"""
    """
    provides multi-parametric representation of image based on megavoxel categories
    """
    tmp1 = np.tensordot(param.megaVoxelBincenters, param.megaVoxelBincenters, axis=1).T 
    tmp2 = np.tensordot(megaVoxelProfile[fgMegaVoxel,:], megaVoxelProfile[fgMegaVoxel,:], axis=1) - 2*(megaVoxelProfile[fgMegaVoxel, :] @ param.megaVoxelBincenters.T)
    a = tmp1 + tmp2
    minDis = np.argmin(a, axis=1)
    x = np.zeros((megaVoxelProfile.shape[0], 1))
    x[fgMegaVoxel, 0] = minDis
    numbins = param.numMegaVoxelBins
    tmp = np.zeors((1, numbins+1))
    for i in range(0, numbins+1):
        tmp[:, i] = np.sum(x[fgMegaVoxel, 0] == (i-1))
    imageProfile = tmp
    if not param.countBackground:
        rawProfile = imageProfile[:, 1:].copy()
        imageProfile = imageProfile[:, 1:]
    else:
        rawProfile = imageProfile.copy(imageProfile)
    imageProfile = imageProfile / np.sum(imageProfile, axis=1) #elementwise right array division.
    return imageProfile, rawProfile

#   getImageThreshold.m
def getImageThreshold(IM):
    maxBins = 256
    freq, binEdges = np.histogram(IM.flatten(), bins=maxBins) 
    binCenters = binEdges[:-1] + np.diff(binEdges)/2
    meanIntensity = np.mean(IM.flatten())
    numThresholdParam = len(freq)
    binCenters -= meanIntensity
    den1 = np.sqrt((binCenters**2) @ freq.T)
    numAllPixels = np.sum(freq) #freq should hopefully be a 1D vector so summ of all elements should be right.
    covarMat = np.zeros(numThresholdParam, 1)
    for iThreshold in range(numThresholdParam):
        numThreshPixels = np.sum(freq[binCenters > binCenters[iThreshold]])
        den2 = np.sqrt( (((numAllPixels - numThreshPixels)*(numThreshPixels))/numAllPixels) )
        covarMat[iThreshold, 0] = (binCenters @ (freq * (binCenters > binCenters[iThreshold])).T) / (den1*den2) #i hope this is the right mix of matrix multiplication and element-wise stuff.
    imThreshold = np.argmax(covarMat) #index makes sense here.
    imThreshold = binCenters[imThreshold] + meanIntensity
    return imThreshold

#   getImageThresholdValues.m
def getImageThresholValues(mData, allImageId, param):
    """
    get image threshold values for dataset.  
    """
    intensityThresholdValues = np.full((5000, param.numChannels), np.nan)
    startVal = 0
    endVal = 0
    for iImages in range(0, param.randFieldID.size):
        ii = allImageId == param.randFieldID[iImages]
        xx = mData[ii, 0]
        if xx.size == 0:
            print('SSS') #dont know about this one boss
        d, param.fmt = getImageInformation(mData[ii, 0])
        param = getTileInfo(d, param)
        iTmp = getIndividualChannelThreshold(mData[ii, 0:param.numChannels], param, ii=ii)
        intensityThresholdValues[startVal:endVal+iTmp.shape[0]-1, :] = iTmp
        startVal += iTmp.shape[0]
        endVal += iTmp.shape[0]
    ii  = (intensityThresholdValues[:, 0] == np.nan) == False #ii is where not nan. Im not sure why they chose to write it like this tho.
    param.intensityThresholdValues = intensityThresholdValues[ii, :]
    return param

#   getImageWithSVMVOverlay.m
def getImageWithSVMVOverlay(IM, param, type):
    """
    I assume this means get image with superVoxel or megaVoxel overlay.

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
 
#   getIndividualChannelThreshold.m
def getIndividualChannelThreshold(filenames, param, ii=None):
    """called in getImageThresholdValues""" 
    
    numberChannels = filenames.shape[1]
    thresh = np.zeros(filenames.shape)
    if ii == None:
        ii = np.full(filenames.shape[0], True)
    if param.intensityNormPerTreatment:
        grpVal = np.unique(param.grpIndicesForIntensityNormalization[ii])[0] 
    for iChannels in range(0, numberChannels):
        for iImages in range(0, filenames.shape[0]):
            IM = io.imread(filenames[iImages, iChannels]) #might need to add something to properly include .tif at the end of the filenames.
            IM = IM[param.XOffsetStart:(-param.XOffsetEnd), param.YOffsetStart:(-param.YOffsetEnd)]
            if param.intensityNormPerTreatement:
                IM = rescaleIntensity(IM, low=param.lowerbound[grpVal, iChannels], high=param.upperbound[grpVal, iChannels])
            else:
                IM = rescaleIntensity(IM, low=param.lowerbound[:, iChannels], high=param.upperbound[:, iChannels])
            thresh[iImages, iChannels] = getImageThreshold(IM.astype('float64')) #want double precision here. not sure if python can handle this since rounding error occurs at 1e-16, but will make float64 anyway
    #they choose to clear IM here, but if its getting overwritten every for loop, its probably fine.
    return thresh

#   getIntensityFeatures.m
def getIntesityFeatures(metadataFileRaw, metadataFilenameLabel, uImageID, param): #added param to function parameters because it is somehow called here.
    param.channelDiscarded = [] #empty matrix -> empty list should be fine.
    metadataLabel, discard1, discard2, imageIDLabel = getPlateInfoFromMetadatafile(metadataFilenameLabel, param)
    metadataRaw, headerRaw, discard1, imageIDRaw = getPlateInfoFromMetadatafile(metadataFileRaw, param)
    stkCol = headerRaw == 'Stack'
    allFeatures = np.full((uImageID.size, 4), np.nan)
    for i in range(0, uImageID.size):
        ii = imageIDRaw = uImageID[i]
        if np.sum(ii) == 0:
            continue
        stk = [int(stackstr) for stackstr in metadataRaw[ii, stkCol]] #these values are presumably str values in some array container. want to convert each eeach string to int.
        stk2pick = (np.max(stk) - np.min(stk))//2
        jj = stk == np.min(stk) + stk2pick
        channelCol = metadataRaw[ii, 0:3]
        channelCol = channelCol[jj, :]
        #label image
        kk = imageIDLabel == uImageID[i]
        labelImage = metadataLabel[kk, 0]
        labelImage = io.imread(labelImage)
        labelImage = labelImage > 0
        labelImage = cv.erode(labelImage, np.ones((20,20)))
        labelImageEroded = cv.erode(labelImage, np.ones((20,20))) #erode twice
        #in their code, There is commented out imshow() stuff to show the eroded images here.

        for k in range(0,3):
            rawImage = io.imread(channelCol[0, k]).astype(np.float64) #remember it is '.Tiff' file.
            im1 = labelImageEroded * rawImage #elementwise multiplication is correct here
            int1 = np.sum(im1) / np.sum(labelImageEroded)
            im1 = (labelImage - labelImageEroded) * rawImage #elementwise multiplication is correct
            l = labelImage - labelImageEroded
            int2 = np.sum(im1)/np.sum(l)
            allFeatures[i, k] = int2/int1
        allFeatures[i, 4] = np.max(stk) - np.min(stk)
        #get MIP
    allFeaturesInt = np.full((uImageID.size, 3), np.nan)
    for i in range(0, uImageID.size):
        ii = imageIDRaw == uImageID[i]
        if np.sum(ii) == 0:
            continue
        stk = [int(stackstr) for stackstr in metadataRaw[ii, stkCol]]
        stk2pick = (np.max(stk) - np.min(stk))//2
        jj = stk == (np.min(stk)+stk2pick)
        channelCol = metadataRaw[ii, 1:3]
        #label image
        kk = imageIDLabel == uImageID[i]
        labelImage = metadataLabel[kk, 0]
        labelImage = io.imread(labelImage)
        labelImage = labelImage > 0
        #commented code: eroded image
        for k in range(0, 3):
            rawImage = getMIPImage(channelCol[:, k])
            im1 = labelImage * rawImage
            allFeaturesInt[i, k] = np.sum(im1) / np.sum(labelImage)
    allHeader = ['Ch1Int', 'Ch2Int', 'Ch3Int', 'Ch1Ratio', 'Ch2Ratio', 'Ch3Ratio', 'NumPlanes']
    allData = [allFeaturesInt, allFeatures]
    return allHeader, allData

#   getMIPImage.m
def getMIPImage(imfiles):
    """called in getIntensityFeatures"""
    """
    getMIPImage Creates maximum intensity projected images for a set of image
    files in imfiles
    """
    format = regexpi(imfiles[0, 0], r'\.', fmt='split')
    format = format[0, -1]
    numzPlanes = imfiles.shape[0]
    info = imfinfo(imfiles[0,0], format)
    imMIP = np.ones((info.Height, info.Width)) * (-1) * np.inf #why not just initialize with 0, idk, but this works
    for i in range(0, numzPlanes):
        tmp = io.imread(imfiles[i, 0], format).astype(np.float64)
        tmp = sc.signal.medfilt2d(tmp, kernel_size=3)
        imMIP = np.maximum(imMIP, tmp)
    return imMIP

#   getMegaVoxelBinCenters.m
def getMegaVoxelBinCenters(mData, allImageId, param):
    """
    compute bincenters for megaVoxels

    % mData  - Metadata
    % allImageID - Image IDs of each image stack
    % param - All parameters
    % Output
    % param - Appended parameters
    """
    MegaVoxelsforTraining = []
    totalIterations = param.randFieldID.size+1
    for iImages in range(0, param.randFieldID.size):
        ii = allImageId == param.randFieldID[iImages]
        d, param.fmt = getImageInformation(mData[ii, 0])
        param = getTileInfo(d, param)
        tmpInfoTable = mData[ii, 0:param.numChannels]
        superVoxelProfile, fgSuperVoxel = getTileProfiles(tmpInfoTable, param.pixelBinCenters, param, ii)
        megaVoxelProfile, fgMegaVoxel = getMegaVoxelProfile(superVoxelProfile, fgSuperVoxel, param)
        megaVoxelforTraining = [MegaVoxelsforTraining, megaVoxelProfile[fgMegaVoxel, :]] # i dont think this list creation process is right. matlab format: [megaVoxelforTraining;megaVoxelProfile(fgMegaVoxel,:)
        param.megaVoxelBincenters = getPixelBins(megaVoxelforTraining, param.numMegaVoxelBins)
    return param

#   getMegaVoxelProfile.m
def getMegaVoxelProfile(tileProfile, fgSuperVoxel, param):
    """called in extractImageLevelTextureFeatures"""
    """called in getMegaVoxelBinCenters"""
    a =  np.tensordot(param.supervoxelBincenters, param.supervoxelBincenters, axis=1).T + np.tensordot(tileProfile[fgSuperVoxel, :], tileProfile[fgSuperVoxel, :], axis=1) + (-2)*(tileProfile[fgSuperVoxel, :] @ param.supervoxelBincenters.T)  #elementwise summation 
    minDis = np.argmin(a, axis=2)
    x = np.zeros((tileProfile.shape[0], 1))
    x[fgSuperVoxel, 0] = minDis
    x = np.reshape(x, (param.croppedX/param.tileX, param.croppedY/param.tileY, param.croppedZ/param.tileZ))
    if param.showImage:
        plt.figure()
        plt.imshow(x[:, :, 2], 'gray')
        plt.show()
    x = np.concatenate(np.zeros((param.superVoxelXAddStart, x.shape[1], x.shape[2])), x, np.zeros((param.superVoxelXAddEnd, x.shape[1], x.shape[2])), axis=0)
    x = np.concatenate(np.zeros((x.shape[0], param.superVoxelYAddStart, x.shape[2])), x, np.zeros((x.shape[0], param.superVoxelYAddEnd, x.shape[2])), axis=1)
    x = np.concatenate(np.zeros((x.shape[0], x.shape[1], param.superVoxelZAddStart)), x, np.zeros((x.shape[0], x.shape[1], param.superVoxelZAddEnd)), axis=2)
    x = x.astype(np.uint8)
    param.numMegaVoxelsXY = x.shape[0] * x.shape[2] / (param.megaVoxelTileY @ param.megaVoxelTileX)
    param.numMegaVoxels = (param.numMegaVoxelsXY*x.shape[2])/param.megaVoxelTileZ
    sliceCounter = 0
    tmp = []
    startVal = 0
    endVal = param.numMegaVoxelsXY
    try:
         megaVoxelProfile = np.zeros((param.numMegaVoxels, param.numSuperVoxelBins+1))
    except Exception as e:
        print('e')
    fgMegaVoxel = np.zeros((param.numMegaVoxels, 1))
    for iSuperVoxelImagesZ in range(0, x.shape[2]):
        sliceCounter += 1
        tmp1 = im2col(x[:, :, iSuperVoxelImagesZ], (param.megaVoxelTileX, param.megaVoxelTileY), type='distinct')
        tmp = [tmp, tmp1]
        if sliceCounter == param.megaVoxelTileZ:
            for i in range(0, param.numSuperVoxelBins+1):
                megaVoxelProfile[startVal:endVal, i] = np.sum(tmp == i-1, axis=1) #value of zeros means background supervoxel
            fgMegaVoxel[startVal:endVal, 0] = (np.sum(tmp!= 0, axis=1)/tmp.shape[1]) >= param.megaVoxelThresholdTuningFactor
            sliceCounter = 0
            tmp = []
            startVal += param.numMegaVoxelsXY
            endVal += param.numMegaVoxelsXY
    if not param.countBackground:
        megaVoxelProfile = megaVoxelProfile[:, 1:]
    megaVoxelProfile = megaVoxelProfile / np.sum(megaVoxelProfile, axis=2) #hopefully this works, they ask for elementwise, but the arrays seem to have different shapes.
    fgMegaVoxel = fgMegaVoxel.astype(bool) 
    return megaVoxelProfile, fgMegaVoxel

#   getMerged3DImage.m
def getMerged3DImage(filenames, colors, boundValues, thresholdValues):
    #this function makes a distinction about whether filenames is a matlab cell array or not.
    #hopefully, I can standardize things down the line so that this distinction is not needed.
    #here, it looks like cell format filenames means list/array of file names, non-cell format means filenames is actually already multichannel image
    
    #here, I have a poor workaround, but hopefully it holds for now. Will need to come back to this to FIX LATER
    if (filenames.shape[1] == colors.shape[0]): #hopefully neitther of the shape matches can happen coincidentally.
        numChannels = filenames.shape[1]
        info = imfinfo(filenames[0, 0])
        imWidth = info.Width
        imHeight = info.Height
        cellFile = True #unclear if this is necessary
    elif (filenames.shape[2] == colors.shape[1]):
        numChannels = filenames.shape[2]
        imWidth = filenames.shape[1]
        imHeight = filenames.shape[0]
        cellFile = False
    else: #i no colors and shape matches.
        im3D = np.ones((1000, 1000, 3))
        return im3D
    im3D = np.zeros((imHeight, imWidth, 3))
    #want to give image pseudocolor for each channel
    for jChannel in range(0, numChannels):
        if cellFile:
            im = io.imread(filenames[0, jChannel]).astype(np.float64)
        else:
            im = filenames[:, :, jChannel]
        maxMin = np.abs(boundValues[jChannel, 1]) - boundValues[jChannel, 0]
        im = (im - boundValues[jChannel, 0])/maxMin
        im[im>1] = 1
        im[im<0] = 0
        im = im * (im >= thresholdValues[jChannel]) #not sure how this works because im has already been converted to a binary array it seems
        im3D[:, :, 0] = im3D[:, :, 0] + colors[jChannel, 0]*im
        im3D[:, :, 1] = im3D[:, :, 1] + colors[jChannel, 1]*im
        im3D[:, :, 2] = im3D[:, :, 2] + colors[jChannel, 2]*im
    for i in range(0, 3):
        maxI = np.max(np.max(im3D[:, :, i]))
        minI = np.min(np.min(im3D[:, :, i]))
        if (maxI - minI) > 0:
            im3D[:, :, i] = (im3D[:, :, i] - minI)/(maxI - minI)
    return im3D

#   getPixelBinCenters.m
def getPixelBinCenters(mData, allImageId, param):
    """
    compute bincenters for pixels

    allImageId is image ids of each image stack
    mData is metadata.
    param should be some parameter class object.
    """
    pixelsForTraining = np.zeros(300000, param.numChanels)
    startVal = 0
    endVal = 0
    #they have waitbar (for gui, gives you progress update)
    # totalIterations = param.randFieldID.size + 1 #number of elements in array randFieldID +1 #only used for the waitbar
    for iImages in range(param.randFieldID.size):
        ii = allImageId == param.randFieldID[iImages] 
        d, param.fmt = getImageInformation( mData(ii, 0) )
        param = getTileInfo(d, param)
        param.randZForTraining = np.sum(ii)//2 #True = 1, False = 0; so we have number of true in ii.
        tmpInfoTable = mData
        iTmp = getTrainingPixels(tmpInfoTable, param, ii)
        pixelsForTraining[startVal:endVal+iTmp.shape[0]-1, :] = iTmp
        startVal += iTmp.shape[0]
        endVal += iTmp.shape[0]
    pixelsForTraining = pixelsForTraining[np.sum(pixelsForTraining, axis=1) > 0, :]
    param.pixelBinCenters = getPixelBins(pixelsForTraining, param.numVoxelBins)
    return param

#   getPixelBins.m
def getPixelBins(x, numBins):
    """called in getPixelBinCenters"""
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
    % Use kmeans clustering to get  (looks like it is using kmeans++ algorithm)
    """
    #matlab has a built-in k means clustering algorithm. Need to decide whether to use something like sklearn kmeans, or write own k means algo (might be slower)
    m = x.shape[0]
    if m > 50000:
        samSize = 50000
    else:
        samSize = m
    if m > samSize:
        numRandRpt = 10
        binCenters = np.zeros((numBins, x.shape[1], numRandRpt))
        sumD = np.zeros((numRandRpt, 1))
        for iRandCycle in range(0, numRandRpt):
            discard, binCenters[:, :, iRandCycle] = kmeans(x[np.random.rand(m, samSize)], )#OTHER PARAMETERS)

    return binCenters

#   getScalingFactorforImages.m
def getScalingFactorforImages(metadata, allImageID, param):
    """
    compute lower and higher scaling values for each image
    param: structure of parameter value
    metadata: Metadata
    """
    if param.intensityNormPerTreatment:
        randFieldIDforNormalization = getTrainingFields(metadata, param, allImageID, param.treatmentColNameForForNormalization)
        grpVal = np.zeros((randFieldIDforNormalization.size, 1))
    else:
        randFieldIDforNormalization = getTrainingFields(metadata, param, allImageID, param.trainingColforImageCategories)
    minChannel = np.zeros((randFieldIDforNormalization.size, param.numChannels))
    maxChannel = np.zeros((randFieldIDforNormalization.size, param.numChannels))
    numImages = randFieldIDforNormalization.size
    for i in range(0, numImages):
        ii = allImageID == randFieldIDforNormalization[i]
        filenames = metadata[ii, 1:param.numChannels] #whenever stuff like this pops up, probably need to change to something like np.choose
        if len(filenames) == 0:
            print('SSS (no filenames)')
        print(filenames[0, 0]) #in matlab this just returns the value at 0, 0 in the cell. for now, keep as print
        if i == 0:
            d, fmt = getImageInformation(filenames[:, 0])
        d[0, 2] = np.sum(ii)
        param = getTileInfo(d, param)
        randZ = np.sum(ii)//2
        randZ = np.random.Generator.integers(low=0, high=np.sum(ii), size=randZ)
        filenames = filenames[randZ, :]
        minVal = np.fill((filenames.shape[0], param.numChannels), np.inf)
        maxVal = np.fill((filenames.shape[0], param.numChannels), -1*np.inf)
        for j in range(0, param.numChannels):
            for k in range(0, param.numChannels):
                IM = io.imread(filenames[j, k], fmt)
                minVal[j, k] = min(minVal[j, k], np.quantile(IM, 0.01))
                maxVal[j, k] = max(maxVal[j, k], np.quantile(IM, 0.99))
        minChannel[i, :] = np.min(minVal)
        maxChannel[i, :] = np.max(maxVal)
        if param.intensityNormPerTreatment:
            grpVal[i] = np.unique(param.grpIndicesForIntensityNormalization[ii])
    if param.intensityNormPerTreatment:
        uGrp = np.unique(grpVal)
        param.lowerbound = np.zeros((uGrp.size, param.numChannnels))
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
        param.lowerbound = np.quantile(minChannel, 0.01)
        param.upperbound = np.quantile(maxChannel, 0.99)
    return param

#   getSuperVoxelBinCenters.m
def getSuperVoxelBinCenters(mData, allImageId, param):
    """
    compute bin centers for super voxels
    % mData  - Metadata
    % allImageID - Image ID's of each image stack
    % param - All parameters
    """
    param.pixelBinCenterDifferences = np.tensordot(param.pixelBinCenters, param.pixelBinCenters, axis=1).T
    tilesForTraining = []
    totalIterations = param.randFieldID.size + 1
    for iImages in range(0, param.randFieldID.size):
        ii = allImageId == param.randFieldID[iImages]
        d, param.fmt = getImageInformation(mData[ii, 0])
        param = getTileInfo(d, param)
        tmpInfoTable = mData[ii, 0:param.numChannels]
        superVoxelProfile, fgSuperVoxel = getTileProfiles(tmpInfoTable, param.pixelBinCenters, param, ii)
        tmp = superVoxelProfile[fgSuperVoxel, :] #i think fg means foreground
        if param.superVoxelPerField > tmp.shape[0]:
            tilesForTraining = [tilesForTraining, tmp[:, :]]
        else:
            selBlocks = np.random.Generator.integers(low=0, high=tmp.shape[0], size=param.superVoxelPerField)
            tilesForTraining = [tilesForTraining, tmp[selBlocks, :]]
    param.supervoxelBincenters = getPixelBins(tilesForTraining, param.numSuperVoxelBins)
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

    param.croppedX = dimSize[0] - xOffset
    param.croppedY = dimSize[1] - yOffset
    param.croppedZ = dimSize[2] - zOffset
    
    superVoxelXOffset = (param.croppedX/param.tileX) % param.megaVoxelTileX
    superVoxelYOffset = (param.croppedY/param.tileY) % param.megaVoxelTileY
    superVoxelZOffset = (param.croppedZ/param.tileZ) % param.megaVoxelTileZ
    param.origX = dimSize[0]
    param.origY = dimSize[1]
    param.origZ = dimSize[2]

    if xOffset % 2 == 0:
        param.xOffsetStart = xOffset/2 + 1
        param.xOffsetEnd = xOffset/2
    else:
        param.xOffsetStart = xOffset//2 + 1
        param.xOffsetEnd = -(-xOffset//2 )  #ceiling division is the same as upside-down floor division. 
    if yOffset % 2 == 0:
        param.yOffsetStart = yOffset/2 + 1
        param.yOffsetEnd = yOffset/2
    else:
        param.yOffsetStart = yOffset//2 + 1
        param.yOffsetEnd = -(-yOffset//2 )     
    if zOffset % 2 == 0:
        param.zOffsetStart = zOffset/2 + 1
        param.zOffsetEnd = zOffset/2
    else:
        param.zOffsetStart = zOffset//2 + 1
        param.zOffsetEnd = -(-zOffset//2 )    

    if superVoxelXOffset % 2 == 0:
        param.superVoxelXOffsetStart = superVoxelXOffset/2 + 1
        param.superVoxelXOffsetEnd = superVoxelXOffset/2
    else:
        param.superVoxelXOffsetStart = superVoxelXOffset//2 + 1
        param.superVoxelXOffsetEnd = -(-superVoxelXOffset//2) #same floor division trick.
    if superVoxelXOffset != 0: #add pixel rows if size of supervoxels are not directly visible
        numSuperVoxelsToAddX = param.megaVoxelTileX - superVoxelXOffset
        if numSuperVoxelsToAddX % 2 == 0:
            param.superVoxelXAddStart = numSuperVoxelsToAddX/2
            param.superVoxelXAddEnd = numSuperVoxelsToAddX/2
        else:
            param.superVoxelXAddStart = numSuperVoxelsToAddX // 2
            param.superVoxelXAddEnd = -(-numSuperVoxelsToAddX // 2 )
    else:
        param.superVoxelXAddStart = 0
        param.superVoxelXAddEnd = 0
    #same along other axes.
    if superVoxelYOffset != 0:
        numSuperVoxelsToAddY = param.megaVoxelTileY - superVoxelYOffset
        if numSuperVoxelsToAddY % 2 == 0:
            param.superVoxelYAddStart = numSuperVoxelsToAddY/2
            param.superVoxelYAddEnd = numSuperVoxelsToAddY/2
        else:
            param.superVoxelYAddStart = numSuperVoxelsToAddY //2
            param.superVoxelYAddEnd = -(- numSuperVoxelsToAddY //2)
    else:
        param.superVoxelYAddStart = 0
        param.superVoxelYAddEnd = 0
    if superVoxelZOffset != 0:
        numSuperVoxelsToAddZ = param.megaVoxelTileZ - superVoxelZOffset
        if numSuperVoxelsToAddZ % 2 == 0:
            param.superVoxelZAddStart = numSuperVoxelsToAddZ/2
            param.superVoxelZAddEnd = numSuperVoxelsToAddZ/2
        else:
            param.superVoxelZAddStart = numSuperVoxelsToAddZ //2
            param.superVoxelZAddEnd = -(-numSuperVoxelsToAddZ//2)
    else:
        param.superVoxelZAddStart = 0
        param.superVoxelZAddEnd = 0
    #continue first part of supervoxels offset parity with other axes
    if superVoxelYOffset % 2 == 0:
        param.superVoxelYOffsetStart = superVoxelYOffset/2 + 1
        param.superVoxelYOffsetEnd = superVoxelYOffset/2
    else:
        param.superVoxelYOffsetStart = superVoxelYOffset//2 + 1
        param.superVoxelYOffsetEnd = -(-superVoxelYOffset//2)
    if superVoxelZOffset % 2 ==0:
        param.superVoxelZOffsetStart = superVoxelZOffset/2 + 1
        param.superVoxelZOffsetEnd = superVoxelZOffset/2
    else:
        param.superVoxelZOffsetStart = superVoxelZOffset//2 +1
        param.superVoxelZOffsetEnd = -(-superVoxelZOffset//2)

    param.numSuperVoxels = (param.croppedX*param.croppedY*param.croppedZ)//(param.tileX*param.tileY*param.tileZ) #supposed to be all elementwise operations (floor division too)
    param.numSuperVoxelsXY = (param.croppedX*param.croppedY)/(param.tileX*param.tileY)

    tmpX = (param.croppedX/param.tileX) + superVoxelXOffset
    tmpY = (param.croppedY/param.tileY) + superVoxelYOffset
    tmpZ = (param.croppedZ/param.tileZ) + superVoxelZOffset

    param.numMegaVoxels = (tmpX*tmpY*tmpZ) // (param.megaVoxelTileX*param.megaVoxelTileY*param.megaVoxelTileZ)
    param.numMegaVoxelsXY = (tmpX*tmpY)/(param.megaVoxelTileX*param.megaVoxelTileY)

    param.superVoxelRow = np.linspace(0, param.croppedX, num=int(param.croppedX/param.tileX + 1)).astype(int) #I think it should be integer because it seems like we will be indexing with this
    param.superVoxelCol = np.linspace(0, param.croppedY, num=int(param.croppedY/param.tileY + 1)).astype(int)
    param.superVoxelZ = np.linspace(0, param.croppedZ, num=int(param.croppedZ/param.tileZ + 1)).astype(int)

    param.megaVoxelRow = np.linspace(0, param.croppedX, num=int(param.croppedX/param.megaVoxelTileX + 1)).astype(int)
    param.megaVoxelCol = np.linspace(0, param.croppedY, num=int(param.croppedY/param.megaVoxelTileY + 1)).astype(int)
    param.megaVoxelZ = np.linspace(0, param.croppedZ, num=int(param.croppedZ/param.megaVoxelTileZ + 1)).astype(int)
    
    return param

#   getTileProfiles.m
def getTileProfiles(filenames, pixelBinCenters, param, ii):
    """called in extractImageLevelTextureFeatures"""
    """called in getMegaVoxelBinCenters"""
    """
    computes low level categorical features for supervoxels
    function assigns categories for each pixel, computes supervoxel profiles for each supervoxel
    % Inputs:
    % filenames - Image file names images x numchannels
    % pixelBinCenters - Location of pixel categories: number of bins x number
    % of channels
    % Output:
    % superVoxelProfile: number of supervoxels by number of supervoxelbins plus
    % a background
    % fgSuperVoxel: Foreground supervoxels - At lease one of the channles
    % should be higher than the respective thrshold
    % TASScores: If TAS score is sleected
    """
    numTilesXY = (param.cropedX*param.croppedY)/(param.tileX*param.tileY) #why not just use param.numSuperVoxelsXY, I Have not idea. the calculation is the exact same.
    filenames = filenames[param.zOffsetStart:-param.zOffsetEnd, :] #keep z stacks that are divisible by stak count
    sliceCounter=0
    startVal=0
    endVal=numTilesXY

    startCol= 0
    endCol = param.tileX*param.tileY
    if param.intensitynormPerTreatment:
        grpVal = np.unique(param.grpIndicesForIntensityNormalization[ii])
    superVoxelProfile = np.zeros((param.numSuperVoxels, param.numVoxelBins+1))
    fgSuperVoxel = np.zeros((param.numSuperVoxels, 1))
    cnt = 1
    if param.computeTAS:
        categoricalImage = np.zeros((param.croppedX, param.croppedY, param.croppedZ))
    #loop over file names and extract super voxels
    tmpData = np.zeros((numTilesXY, param.tileX*param.tileY*param.tileZ))
    for iImages in range(0, filenames.shape[0]):
        sliceCounter += 1
        croppedIM = np.zeros((param.origX, param.origY, param.numChannels)) #just one slice in all 3 channels
        for jChannels in range(0, param.numChannels):
            if param.intensityNormPerTreatment:
                croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(filenames[iImages, jChannels], param.fmt), low=param.lowerbound[grpVal, jChannels], high=param.upperbound[grpVal, jChannels])
            else:
                croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(filenames[iImages, jChannels], param.fmt), low=param.lowerbound[:, jChannels], high=param.upperbound[:, jChannels])
        croppedIM = croppedIM[param.xOffsetStart:-param.xOffsetEnd, param.yOffsetStart:-param.xOffsetEnd, :] #z portio of the offset has already been done by not loading the wrong slices
        x = np.reshape(croppedIM, (param.croppedX*param.croppedY, param.numChannels))
        fg = np.sum(x > param.intensityThreshold, axis=1) >= param.numChannels/3 #a fancy way to say greater than 1?
        pixelCategory = np.argmin((param.pixelBinCenterDifferences + np.tensordot(x[fg,:], x[fg,:], axis=1)) - 2*(x[fg,:] @ pixelBinCenters.T), axis=1)
        x = np.zeros((param.croppedX, param.croppedY, 1), dtype='uint8')
        x[fg, :] = pixelCategory
        if param.computeTAS:
            categoricalImage[:, :, iImages] = np.reshape(x, param.croppedX, param.croppedY)
        del fg, croppedIM, pixelCategory #not 100 on why to delete al here since things would just be overwritten anyway, but why not right, also, some of the variables to clear where already commented out so I removed them from the list
        if sliceCounter == param.tileZ:
            fgSuperVoxel[startVal:endVal, 0] = (np.sum(tmpData != 0, axis=1)/tmpData.shape[1]) >= param.superVoxelThresholdTuningFactor
            for i in range(0, param.numVoxelBins+1):
                superVoxelProfile[startVal:endVal, i] = np.sum(tmpData == i-1, axis=1)
            sliceCounter = 0
            startVal += numTilesXY
            endVal += numTilesXY
            startCol = 0
            endCol = param.tileX*param.tileY
            tmpData = np.zeros((numTilesXY, param.tileX*param.tileY*param.tileZ))
        else:
            tmpData[:, startCol:endCol] = im2col(np.reshape(x, (param.croppedX, param.croppedY)), (param.tileX, param.tileY), type='distinct')
            startCol += (param.tileX*param.tileY)
            endCol += (param.tileX*param.tileY)
        cnt += 1
    if not param.countBackground:
        superVoxelProfile = superVoxelProfile[:, 1:]
    superVoxelProfile = superVoxelProfile / np.sum(superVoxelProfile, axis=1)
    superVoxelProfile[superVoxelProfile == np.nan] = 0
    if param.computeTAS:
        TASScores = getCategoricalTASScores[categoricalImage, param.numVoxelBins]
    else:
        TASScores = np.zeros((1, 27*param.numVoxelBins))
    fgSuperVoxel = fgSuperVoxel.astype(bool)
    return superVoxelProfile, fgSuperVoxel, TASScores #TASScores is optional

#   getTrainingfields.m
def getTrainingFields(metaInfo, param, allImageId, treatmentColumn):
    """called in getScalingFactorforImages"""
    randomImage = False
    if len(allImageId) == 0:
        randomImage = False
    if treatmentColumn.size == 0:
        randomImage = True
    uniqueImageID = np.unique(allImageId)
    if randomImage:
        randFieldID = uniqueImageID[np.random.Generator.integers(low=0, high=uniqueImageID.size, size=param.randTrainingFields)]
    else:
        if not np.array_equal(treatmentColumn, treatmentColumn.astype(bool)): #if not logical array
            treatmentColumn = param.metaDataHeader == treatmentColumn #want this to be case insensitive comparison of strings
        uTreat = np.unique(metaInfo[:, treatmentColumn])
        param.randTrainingPerTreatment = -(-param.randTrainingFields//uTreat.size) #ceiling division
        randFieldID = np.zeros((param.randTrainingPerTreatment, uTreat.size))
        for i in range(0, uTreat.size):
            ii = metaInfo[:, treatmentColumn] == uTreat[i, :]
            tmp = np.unique(allImageId[ii])
            randFieldID[:, i] = tmp[np.random.Generator.integers(low=0, high=tmp.size, size=param.randTrainingPerTreatment)]
    randFieldID = randFieldID.ravel()
    treatmentValues = []
    for i in range(0, randFieldID.size):
        if not randomImage:
            ii = metaInfo[allImageId == randFieldID[i], treatmentColumn]
            treatmentValues.append(ii[0,:])
        else:
            treatmentValues.append('RR')
    return randFieldID, treatmentValues

#   getTrainingPixels.m
def getTrainingPixels(filenames, param, ii):
    """called in getPixelBinCenters"""
    filenames = filenames[np.random.Generator.integers(low=0, high=filenames.shape[0], size=param.randZForTraining), :]
    trPixels = np.zeros((param.pixelsPerImage*param.randZForTraining, param.numChannels))
    startVal = 0
    if param.intensityNormPerTreatment:
        grpVal = np.unique(param.grpIndicesForIntensityNormalization[ii])
    filenames = filenames[0:(filenames.shape[0]//2), :]
    for iImages in range(0, filenames.shape[0]):
        croppedIM = np.zeros((param.origX, param.origY, param.numChannels))
        for jChannels in range(0, param.numChannels):
            if param.intensityNormPerTreatment:
                croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(filenames[iImages, jChannels], param.fmt), low=param.lowerbound[grpVal, jChannels], high=param.upperbound[grpVal, jChannels])
            else:
                croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(filenames[iImages, jChannels], param.fmt), low=param.lowerbound[:, jChannels], high=param.upperbound[:, jChannels])
        croppedIM = croppedIM[param.xOffsetStart:-param.xOffsetEnd, param.yOffsetStart:-param.yOffsetEnd, :]
        croppedIM = np.reshape(croppedIM, (param.croppedX*param.croppedY, param.numChannels))
        croppedIM = croppedIM[np.sum(croppedIM > param.intensityThreshold, axis=2) >= param.numChannels/3, :]
        croppedIM = selectPixelsbyweights(croppedIM)
        if croppedIM.shape[0] >= param.pixelsPerImage:
            trPixels[startVal:startVal + param.pixelsPerImage-1, :] = croppedIM[np.random.Generator.integers(low=0, high=croppedIM.shape[0], size=param.pizelsPerImage), :]
            startVal += param.pixelsPerImage
        else:
            trPixels[startVal:(startVal+croppedIM.shape[0]-1)] = croppedIM
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

#   mergeChannels.m
def mergeChannels(imMultiChannel, colors):
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

#   ParseMetadataFile.m
def parseMetaDataFile(metadatafilename):
    """called in getPixelBinCenters"""
    mData = []
    chanInfo = {} #since we call attributes/cell labels: try to use dictionary
    header = [] #all three of these are set up as cells
    try:
        with open(metadatafilename, mode='r+') as fid: #no truncation
            header = fid.readline().strip() #first line with leading and trailing whitespaces removed.
        header = regexpi(header, r'\t', fmt='split')
        if header != 'ImageID':
            print('\nERROR: please choose appropriate metadata file. \nThings will probably crash after this.\n\n')
            return None
        with open(metadatafilename) as fid:
            tmp = np.loadtxt(fid, delimiter='\t', skiprows=1, dtype={'names': [f'col{i}' for i in range(0, header.size)], 'formats': ['S4' for i in range(0, header.size)]} )
    except Exception as e:
        print(e)
        print('\nERROR reading metadata file did not work.\n\n')
    for i in range(0, tmp.shape[1]):
        mData.append(tmp[0, i]) #may or may not work properly.
    chanInfo['channelColNumber'] = input('Select channels (type indices of choice separated by spaces) :').split() #list dialog box pops up. shows options of list of header entries, allows multiple selection mode, prompts to select channels. returns indices of selected channels in header.
    chanInfo['channelColNumber'] = np.array([int(elem) for elem in chanInfo['channelColNumber']]) #array form of list of str numbers converted to int. (lets try this for now, seems close enough)
    chanInfo['channelNames'] = header[0, chanInfo['channnelColNumber']]
    chanInfo['channelColors'] = [f'color{i+1}' for i in range(0, chanInfo['channelColNumber'].size)]
    ii = header == 'ImageID' #should be case insensitive though
    imageID = mData[:, ii].astype(np.float64)
    stackCol = np.logical_or((header == 'Stacks'), (header == 'Stacks')) #should also be case insensitive unfortunately
    uImageID = np.unique(imageID)
    for i in range(0, uImageID.size):
        ii = imageID == uImageID[i]
        kk = mData[ii, stackCol].astype(np.float64)
        tmp = mData[ii, :]
        kk = np.argsort(kk)
        mData[ii, :] = tmp[kk, :]
    ii = header == 'ImageID' #also needs to be case insensitive
    imageID = mData[:, ii].astype(np.uint8) #hpe this works, might need to do as list comprehension instead
    return mData, imageID, header, chanInfo

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

#   setImageView.m
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

#   writestr.m
def writestr(fname, data, wflag):
    """
    % function to write a matrix to a file
    % usage flag=writedat(filename,data,write_flag)
    """
    n = data.shape
    flag = 1
    if n > 0:
    try:
        if wflag == 'Append':
            fid = open(fname, mode='a+')
        elif wflagg == 'Overwrite':
            fid = open(fname, mode='w+')
        else:
            #matlab code has a condition to see if: exist(fname, 'file') != False. not sure why this is that
            fid = open(fname, mode='w+')
    except Exception as e:
        print()
        print('ERROR problem opening file for writing')
        print(e)
        print()
        return None
    for i in range(0, n[0]):
        print('\n' + data[i, :].strip())
    fid.close()
    return flag



#need to find these ones

def getPlateInfoFromMetadatafile(metadataFilenameLabel, param):
    """called in getIntensityFeatures"""
    """function not defined in lib, third party/clustering, organoidCSApp folders"""
    return metadataLabel, unknown, unknown, imageIDLabel

def getCategoricalTASScores(categoricalImage, numVoxelBins):
    """called in getTileProfiles"""
    return TASScores




#gui function to fill in later:

#   PhinDR3D_Main.m
#    seems to be mostly gui stuff (lots of callbacks (function that executes on predefined action), calls to other functions to set things up)
def PhinDR3D_Main():
    return None

#   PhindrViewer.m
#     again, seems to be mostly gui stuff, img viewing i think. some functions to save cluster results to file.
def PhindrViewer():
    return None

#   colorpicker.m
def colorpicker():
    #gui stuff, I am pretty sure this lets the user pick what color is used to display each channel.
    #go back to later as needed.
    return None

#   findjobj.m
#      very intense gui stuff. apparently matlab gui is built on java, this matlab function does works through handles, other gui things 
#      straight up gui building, user interactions, etc.

#   imageViewer.m
def imageViewer():
    return None

#   metadataExtractor.m
def metadataExtractor():
    return None

#   setChannelInformation
def setChannelInformation(boxHandle, channelNames, channelColors, selectedChannels):
    """
    function seems to be gui related, should display information about the different channels
    related to boxHandle function of gui.
    """
    return None

#   setParameterValues.m
def setParametervalues(handles, param):
    """
    sets attributes in handles to specific values in param. won't touch this yet.
    may or may not be interface related
    """
    return None

#   setParameters.m
def setParameters(input):
    """GUI stuff. calls setParameterValues"""
    return None

#   setTextInformation.m
def setTextInformation(textHandle, imageFileName, channelNames, channelColors, selectedChannels, projectionType):
    """set text things on interface"""
    return None

#   updateAxisImage.m
def updateAxisImage(axisHandle, image2Display, typeOfview):
    """gui stuff, updates image to be displayed in specified axis"""
    return None

#   updateAxisPlot.m
def updateAxisPlot(data, groupValues, colorValues, axisHandle):
    """more gui stuff, update plots of data and groupValues that we are supposed to be plotting"""
    return None

#   viewScatterPie2.m
def viewScatterPie2(x, y, yhat, map, uY, uYhat):
    """plots data in x using groupings in y with colors in yhat"""
    """show scatter plot of pie plots i guess? there are scatterplots of pie plots in phindr paper"""
    return None























