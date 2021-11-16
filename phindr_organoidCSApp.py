"""
Teophile Lemay, 2021

functions from organoidCSApp folder ( https://github.com/DWALab/Phindr3D/tree/main/Phindr3D-OrganoidCSApp )
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import morphology as morph
from skimage import segmentation as seg
from skimage import filters
from scipy import ndimage
import phindr_functions as ph
import scipy.io as io

# builtin matlab functions to replicate
def imadjust(img, inrange=[0,1], outrange=[0,1], gamma=1):
    #convert image range from inrage to outrange # solution from ( https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python )
    a, b = inrange
    c, d = outrange
    adjusted = (((img - a)/(b-a)) ** gamma) * (d-c) + c

def imfill(img):
    """
    similar to matlab imfill('holes')
    flood fill holes in binary image that are separate from the border. 
    algo:
       padd outer borders to ensure flooding from corner goes all around
       flood from border, invert
       combine inverted flooded with binary map (OR operation)
    
    img should be binary map 
    """
    rows, cols = img.shape
    tmp = np.zeros((rows+2, cols+2))
    tmp[1:-1, 1:-1] = img #make a copy of img padded on all sides by zeros
    flooded = seg.flood_fill(tmp, (0,0), 1, in_place=True) #default is already full connectivity including diagonals as desired.
    inverted = 1 - flooded
    inverted = inverted[1:-1, 1:-1] #get rid of padding
    return np.logical_or(img, inverted) #works!

def bwareaopen(img, num, conn=8):
    """
    like morphological opening, but dont use a filter, just do connected components and remove small groups of connected components

    img: binary image
    num: number of components threshold for being allowed
    """
    #want fully connected components
    if conn == 8:
        struct = np.ones((3,3))
    elif conn == 4:
        struct = np.ones((3,3))
        struct[0, 0] = 0
        struct[-1, 0] = 0
        struct[0, -1] = 0
        struct[-1, -1] = 0 #take out corners
    labelled, ret = ndimage.label(img, structure = struct)
    labels = np.unique(labelled)
    for label in labels:
        if np.sum(labelled == label) < num:
            img[labelled == label] = 0 
    return img #works!

def imextendedmax(img, H):
    """
    similar to imextendedmaxima function from matlab
    H-maxima transform surpresses maxima to be 
    """
    #need to check textbook reference for this
    return None

def stdfilt(img, kernel_size):
    """
    apparently this is a faster way to compute std deviation filter ( https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html )
    """
    c1 = ndimage.filters.uniform_filter(img, kernel_size, mode='reflect')
    c2 = ndimage.filters.uniform_filter(img**2, kernel_size, mode='reflect')
    res = c2 - c1*c1
    res[res < 0] = 0 #hopefully this fixes the nan issue without needing to add small random numbers
    return np.sqrt(res)


# defined functions in the folder

#   getFocusplanesPerObjectMod.m
def getFocusplanesPerObjectMod(labelImage, fIndex, numZ=None):
    """
    computes optimal focus plane for each object in a 3d stack

    labelImage: bool image indicating presence of object
    """
    if numZ == None:
        numZ = np.max(fIndex)
    try:
        ll = np.zeros((1, 2))
        ii = fIndex[labelImage]
        ii = ii.ravel()
        n = np.digitize(ii, np.linspace(0.5, numZ-0.5, num=int(numZ))) # bins: 0.5 to numZ-0.5 with step size of 1 (just want the bin counts)
        n = n/np.sum(n)
        rndPoint = 1/numZ
        p = np.argwhere(n > rndPoint)
        if p.size == 0:
            minN = 0 #assume this is a indexing thing
            maxN = numZ 
        minN = max([1, np.min(p)-2])
        maxN = min([numZ, np.max(p)+2])

        ll[0] = minN
        ll[1] = maxN
    except Exception as e:
        print()
        print(e)
        print()
    return ll, n

#   getImageThreshold.m
#exactly the same function as defined in the lib folder. copy and pasted here: degenerate code.
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

#not done with this one yet. need to ask questions to matlab people/check textbook reference
#   getSegmentedOverlayImage.m
def getSegmentedOverlayImage(final_im, min_area_spheroid, radius_spheroid, smoothin_param, entropy_thresh, intensity_threshold, scale_spheroid):
    newfim = final_im.copy()
    SE = morph.disk(2*radius_spheroid)
    IM2 = morph.white_tophat(newfim, SE)
    IM4 = smoothImage(IM2, smoothin_param)
    minIM = np.min(IM4)
    maxIM = np.max(IM4)
    IM4 = IM4 - minIM 
    IM4 = IM4/(maxIM - minIM)
    IM4 = imadjust(IM4, gamma=0.5)
    IM6 = segmentImage(IM4, min_area_spheroid)
    IM6 = IM6 > 0
    IM6 = morph.binary_closing(IM6, footprint=np.ones(3,3))
    IM6 = morph.binary_closing(IM6, footprint=np.ones(3,3))
    IM6 = morph.binary_closing(IM6, footprint=np.ones(3,3)) #do 3 times.
    IM6 = imfill(IM6)
    IM6 = bwareaopen(IM6, 20) #open sets of connected components with less than 20 members
    IM7 = ndimage.distance_transform_edt(IM6)# matlab bwdist gives euclidean distance transform from non-zero elements. use on binary inverse of IM6 so distance transform from zero-elements. ndimage.distance_transform_edt is already distance from zero elements
    if scale_spheroid > 1:
        scale_spheroid = 1
    elif scale_spheroid <= 0:
        scale_spheroid = 0.1
    splitFactor = scale_spheroid * radius_spheroid
    IM9 = None# i think and truly hope that imextended maxima exists to threshold within rois.
    return segImage, L


#   getfsimage.m
def getfsimage(fnames):
    """
    % getfsimage - Outputs best focussed image from a set of 3D image slices
    % Input:
    % fnames - 3D image file names
    % Output:
    % final_image: Best focussed image
    """
    imInfo = ph.imfinfo(fnames[0, :])
    numZ = fnames.shape[0]
    kernel = np.ones((5,5))
    prevImage = np.fill((imInfo.Height, imInfo.Width), -1*np.inf)
    focusIndex = np.zeros((imInfo.Height, imInfo.Width))
    finalImage = np.zeros((imInfo.Height, imInfo.Width))
    for iFiles in range(0, numZ):
        IM = io.imread(fnames[iFiles, :], 'tiff')
        #some graphics stuff

        #imgradient actually returns both gradient magnitude and gradient direction. I have to assume that only gradietn magnitude is of interest here. by default, imgradient uses sobel method.
        xgrad = ndimage.sobel(IM, axis=0)
        ygrad = ndimage.sobel(IM, axis=1)
        tmp = np.sqrt( xgrad**2 + ygrad **2 ) #gradient magnitude
        ii = tmp >= prevImage
        focusIndex[ii] = iFiles
        finalImage[ii] = IM[ii]
        prevImage = np.maximum(tmp, prevImage)
    return finalImage, focusIndex
#in same file:
def binImage(I, bsize):
    m, n = I.shape
    B1 = ph.im2col(I, (bsize, bsize), type='distinct')
    B1 = np.mean(B1, axis=0) # I hope this is the right direction along which to take the mean, but I am not sure.
    im = np.reshape(B1, (int(m/bsize), int(n/bsize))) #if this doesnt work and throws an error, it is probably because the mean was taken along the wrong axis.
    return im

#another degenrate function:
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

#   removeBorderObjects.m
def removeBorderObjects(L, dis):
    """
    remove objects touching border of image
    """
    borderimage = np.zeros(L.shape)
    #some very strange indexing here, need to reference matlab to numpy guide.
    borderimage[:, np.r_[0:dis, borderimage.shape[1]-dis:borderimage.shape[1]]] = 1 #not quite sure this is correct
    borderimage[np.r_[0:dis, borderimage.shape[0]-dis:borderimage.shape[0]], :] = 1

    L2 = borderimage * L.astype(np.float64)
    uL = np.uniqueL2

    for i in range(0, uL.size):
        L[L == uL[i]] = 0
    
    L = resetLabelImage(L)
    return L

#   resetLabelImage.m
def resetLabelImage(L):
    """
    """
    uL = np.unique(L)
    uL = uL[uL > 0] #remove background
    for i in range(0, uL.size):
        ii = L == uL[i]
        L[ii] = i
    return L

#   segmentImage.m
def segmentImage(imfilename, minArea):
    if isinstance(imfilename, str):
        I = io.imread(imfilename)
    else:
        I = imfilename
    imthreshold = ph.getImageThreshold(I.astype(np.float64))
    bw = bwareaopen(I>imthreshold, minArea, conn=4)
    struct = np.ones((3,3))
    L, N = ndimage.label(bw, structure = struct)
    nI = np.zeros(I.shape)
    nI[:, np.r_[0, nI.shape[1]]] = 1 #hopefully this is correct indexing
    nI[np.r_[0, nI.shape[0]], :] = 1
    nL = L * nI
    uL = np.unique(nL)
    for i in range(0, uL.size):
        ii = L==uL[i]
        areaVal = np.sum(ii)
        if areaVal < minArea:
            L[ii] = 0
    N = np.unique(L)
    N = N[N>0] #get rid of background label
    for i in range(0, N.size):
        ii = L==N[i]
        L[ii] = i+1
    N = np.unique(L)
    N = np.sum(N>0) #number of labels above 0
    return L, N

def smoothImage(IM, cutOff):
    #smooths image in frequency domain
    m, n = IM.shape
    if cutOff > 0.99:
        cutOff = 0.99
    elif cutOff <= 0: #in matlab code is <. changed to <= because min value it is reset to is 0.1
        cutOff = 0.1
    m /= 2
    n /= 2
    x, y = np.mgrid[ -(m-0.5):(m-0.5)+1, -(n-0.5):(n-0.5)+1 ] #might need to use actual meshgrid and define index locations with linspace or arange
    x /= m
    y /= m #shouldnt this be divided by n? who knows
    x = np.sqrt(x**2 + y**2)
    x = (x.T > cutOff).astype(np.float64)
    struct = np.ones((50,50)) #50 by 50 seems very big, but whatever
    x = filters.rank.mean(x, struct)





#NOT FINISHED WITH THIS ONE YET.
#   organoidSegmentation.m
def organoidSegmentation():
    """
    this file contains a stand-alone script to wite [sic] out oncogene images
    calls the other functions in the folder. could probably be put in a "if __name__ == '__main__' " block.
    leave alone for now, will come back to it later.
    """
    return None




#to be defined:

def smoothImage(image, smoothin_param):
    return smoothed
















