"""
Teophile Lemay, 2021

functions from organoidCSApp folder ( https://github.com/DWALab/Phindr3D/tree/main/Phindr3D-OrganoidCSApp )
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology as morph
from skimage import segmentation as seg
from skimage import filters
from skimage import measure
from skimage.feature import peak_local_max
from scipy import ndimage
import phindr_functions as ph
import skimage.io as io
import re
import cv2 as cv


# builtin matlab functions to replicate
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

def imadjust(img, inrange=[0,1], outrange=[0,1], gamma=1):
    #convert image range from inrage to outrange # solution from ( https://stackoverflow.com/questions/39767612/what-is-the-equivalent-of-matlabs-imadjust-in-python )
    a, b = inrange
    c, d = outrange
    adjusted = (((img - a)/(b-a)) ** gamma) * (d-c) + c
    return adjusted

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
    inverted = flooded*(-1) + 1
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

def HMAX(f, h):
    """
    H-maxima transform (reconstruction by dilation of image, using (image - H) as the seed/marker)
    """
    img = f.copy().astype(np.float64)
    seed = img - h
    seed[seed<0] = 0
    return morph.reconstruction(seed, img, method='dilation')

def imextendedmax(img, H):
    """
    pick out regional maxima of H maxima transform

    analog to imextendedmaxima function from matlab
    """
    foot = np.ones((3,3))
    h_maxed = HMAX(img, H)
    return morph.local_maxima(h_maxed, footprint=foot, allow_borders=True)
    
def imimposemin(img, minima):
    """
    assume integer img values starting at 0
    """
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum((img + 1), marker)
    return morph.reconstruction(marker, mask, method='erosion')

def stdfilt(img, kernel_size=5):
    #called in getfsImage
    """
    apparently this is a faster way to compute std deviation filter ( https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html )
    
    result is similar to matlab stdfilt function but not perfect match.
    """
    c1 = ndimage.filters.uniform_filter(img, size=kernel_size, mode='reflect')
    c2 = ndimage.filters.uniform_filter((img*img), size=kernel_size, mode='reflect')
    res = c2 - c1*c1
    res[res < 0] = 0
    return np.sqrt(res)

def imcomplement(image):
    """Equivalent to matlabs imcomplement function"""
    if image.dtype == 'float64':
        max_type_val = 1
    else:
        max_type_val = np.finfo(image.dtype).max
    return max_type_val - image
    
def watershed(img):
    """
    img should be the distance transform already flipped to be "deep" where objects are
    """
    # coords = peak_local_max((-img), footprint=np.ones((3,3)))
    struct = np.ones((3,3)) #structure was 3,3
    dilated = cv.dilate(-img, struct)
    mask = ((-img) >= dilated).astype('uint8') #both used to be -img
    # # mask = np.zeros(img.shape, dtype=bool)
    # # mask[tuple(coords.T)] = True
    markers, ret = ndimage.label(mask)
    return seg.watershed(img, markers=markers, watershed_line=True)

def regionprops(watershed_img, final_im, IM11):
    """
    regionprops:
    area of each labelled region
    mean intensity of final_im in each labelled region
    mean intensity of entropy_filter on IM11 in each labelled region
    """
    labels = np.unique(watershed_img)
    if labels[0] == 0:
        labels = labels[1:]
    areas = np.zeros(labels.shape)
    final_im_intensities = np.zeros(labels.shape)
    entropy = np.zeros(labels.shape)
    ent = filters.rank.entropy(IM11, footprint=np.ones((5,5)))
    for i, label in enumerate(labels):
        areas[i] = np.sum((watershed_img == label))
        final_im_intensities[i] = np.mean(final_im[watershed_img == label])
        entropy[i] = np.mean(ent[watershed_img == label])
    return labels, areas, final_im_intensities, entropy

def bwperim(img, level=0.5):
    """
    trace contours of img along level line
    0.5 is good level for binary image
    """
    contours = measure.find_contours(img, level=level)
    trace = np.zeros(img.shape)
    for contour in contours:
        contour = contour.astype('int')
        trace[(contour[:, 0], contour[:, 1])] = 1
    return trace

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
        ll = np.zeros(2)
        ii = fIndex[labelImage]
        bins = np.append(np.linspace(0.5, numZ-0.5, num=int(numZ), endpoint=True), [numZ-0.5]) #this weird bin format gives a repeated final bin edge value, replicates histc function bin behaviour from matlab.
        n = np.histogram(ii, bins=bins)[0] #1d histogram array.
        n = n/np.sum(n)
        rndPoint = 1/numZ
        p = np.argwhere(n > rndPoint)+1 #find returns linear indices, so does argwhere, except starting at 0. for matlab , indices corresponds 1-1 to z-planes. need +1 to get proper z-planes values.
        if p.size == 0:
            minN = 1
            maxN = numZ
        else:
            minN = max([1, np.min(p)-2])
            maxN = min([numZ, np.max(p)+2])

        ll[0] = int(minN)
        ll[1] = int(maxN)
    except Exception as e:
        print('\nexception found')
        print(e)
        print()
    return ll #want ll to be zplane values, corresponding to fIndex values.

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

#   getSegmentedOverlayImage.m
def getSegmentedOverlayImage(final_im, min_area_spheroid, radius_spheroid, smoothin_param, entropy_thresh, intensity_threshold, scale_spheroid):
    # newfim = final_im.copy()
    SE = morph.disk(2*radius_spheroid)
    IM2 = cv.morphologyEx(final_im.astype('uint16'), cv.MORPH_TOPHAT, SE).astype('float64')
    IM4 = smoothImage(IM2, smoothin_param)
    minIM = np.min(IM4)
    maxIM = np.max(IM4)
    IM4 = (IM4 - minIM)/(maxIM - minIM) #rescale 0-1
    IM4 = imadjust(IM4, gamma=0.5)
    # #im4 is float image from 0 to 1
    IM6 = segmentImage(IM4, min_area_spheroid)
    IM6 = (IM6 > 0).astype('uint8')
    IM6 = cv.morphologyEx(IM6, cv.MORPH_CLOSE, np.ones((3,3))) #this operation is iterated 3 times in matlab script. No reason to do this because morphological closing is idempotent.
    IM6 = imfill(IM6)
    IM6 = bwareaopen(IM6, 20) #open sets of connected components with less than 20 members 
    #im6 is binary image
    IM7 = ndimage.distance_transform_edt(IM6)# matlab bwdist gives euclidean distance transform from non-zero elements. use on binary inverse of IM6 so distance transform from zero-elements. ndimage.distance_transform_edt is already distance from zero elements
    #im7 is float image
    if scale_spheroid > 1:
        scale_spheroid = 1
    elif scale_spheroid <= 0:
        scale_spheroid = 0.1
    splitFactor = scale_spheroid * radius_spheroid
    IM9 = imextendedmax(IM7, splitFactor)
    bw = np.logical_or(np.logical_not(IM6), IM9)
    IM10 = imimposemin(imcomplement(IM4*IM7),bw)
    L = watershed(IM10)
    L = np.maximum(L-1, 0) #bacground label is 1, so replace with 0.
    # L -= 1 #this replaces the line above, assumes background is labelled 1.
    L = removeBorderObjects(L, 30) #used to be 10.
    IM11 = (final_im - np.min(final_im)) / (np.max(final_im) - np.min(final_im))
    labels, areas, final_im_means, entropies = regionprops(L, final_im, IM11)
    # return L, labels, areas, final_im_means, entropies
    if np.sum(areas) != 0:
        i2 = areas >= min_area_spheroid
        i3 = final_im_means >= intensity_threshold
        i4 = entropies >= entropy_thresh
        ii = ((i2*i3*i4) == 0) #ii is True at the indices of all the labels that we want to discard
        for l in labels[np.nonzero(ii)]:
            L[L==l] = 0
    L = resetLabelImage(L)
    seg_image = bwperim((L >= 1))
    seg_image = morph.dilation(seg_image, footprint=np.ones((3,3)))
    return seg_image, L


#   getfsimage.m  
def getfsimage(imdata, channel):
    """
    % getfsimage - Outputs best focussed image from a set of 3D image slices
    % Input:
    % fnames - 3D image file names
    % Output:
    % final_image: Best focussed image
    """
    zVals = np.unique(imdata['Stack'])
    #want to end up with fnames == images from single channel being read in stack order.
    imInfo = ph.imfinfo( imdata.loc[imdata['Stack']==zVals[0], channel].values[0])
    prevImage = np.full((imInfo.Height, imInfo.Width), -1*np.inf)
    focusIndex = np.zeros((imInfo.Height, imInfo.Width))
    finalImage = np.zeros((imInfo.Height, imInfo.Width))
    for z in zVals:
        IM = io.imread(imdata.loc[imdata['Stack']==z, channel].values[0]).astype(np.float64)
        imtmp = stdfilt(IM, kernel_size=5)
        xgrad = ndimage.sobel(imtmp, axis=0) #directional gradients
        ygrad = ndimage.sobel(imtmp, axis=1)
        tmp = np.sqrt( (xgrad*xgrad) + (ygrad*ygrad) ) #gradient magnitude
        ii = (tmp >= prevImage)
        focusIndex[ii] = z
        finalImage[ii] = IM[ii]
        prevImage = np.maximum(tmp, prevImage)
    return finalImage, focusIndex

#in same file: #never called hough, not sure why we have it.
def binImage(I, bsize):
    m, n = I.shape
    B1 = ph.im2col(I, (bsize, bsize), type='distinct')
    B1 = np.mean(B1, axis=0) # I hope this is the right direction along which to take the mean, but I am not sure.
    im = np.reshape(B1, (int(m/bsize), int(n/bsize))) #if this doesnt work and throws an error, it is probably because the mean was taken along the wrong axis.
    return im

#   removeBorderObjects.m
def removeBorderObjects(L, dis):
    """
    remove objects touching border of image
    """
    borderimage = np.zeros(L.shape)
    #some very strange indexing here, need to reference matlab to numpy guide.
    borderimage[:, :dis] = 1
    borderimage[:, -dis:] = 1
    borderimage[:dis, :] = 1
    borderimage[-dis:, :] = 1

    L2 = borderimage * L
    uL = np.unique(L2)

    for borderL in uL:
        L[L == borderL] = 0
    
    L = resetLabelImage(L)
    return L

#   resetLabelImage.m
def resetLabelImage(L):
    """
    rename labels in a labelled image from 1 to # of labels
    """
    uL = np.unique(L)
    uL = uL[uL > 0] #remove background
    for i, label in enumerate(uL):
        ii = (L == label)
        L[ii] = i+1
    return L.astype(int)

#   segmentImage.m
def segmentImage(imfilename, minArea):
    if isinstance(imfilename, str): #can probably remove this part here.
        I = io.imread(imfilename)
    else:
        I = imfilename

    imthreshold = ph.getImageThreshold(I.astype(np.float64))
    bw = bwareaopen(I>imthreshold, minArea, conn=8) #conn used to be 4 here
    struct = np.ones((3,3))
    L, N = ndimage.label(bw, structure = struct)
    #set border to 1
    nI = np.zeros(I.shape)
    nI[:, np.r_[0, nI.shape[1]-1]] = 1 #hopefully this is correct indexing
    nI[np.r_[0, nI.shape[0]-1], :] = 1
    nL = L * nI
    uL = np.unique(nL)
    for i in range(0, uL.size): #this is a little redundant, does same as bwareaopen, as a double check after we remove the outside boundary.
        ii = L==uL[i]
        areaVal = np.sum(ii)
        if areaVal < minArea:
            L[ii] = 0
    ##stuff below is unused, so I commented it out.
    # N = np.unique(L)
    # N = N[N>0] #get rid of background label
    # for i in range(0, N.size):
    #     ii = L==N[i]
    #     L[ii] = i+1
    # N = np.unique(L)
    # N = np.sum(N>0) #number of labels above 0
    return L#, N

def smoothImage(IM, cutOff):
    #smooths image in frequency domain
    m = np.max(IM.shape)
    if cutOff > 0.99:
        cutOff = 0.99
    elif cutOff <= 0: #in matlab code is <. changed to <= because min value it is reset to is 0.1
        cutOff = 0.01
    m /= 2
    x, y = np.mgrid[ -(m-0.5):(m-0.5)+1, -(m-0.5):(m-0.5)+1 ] #might need to use actual meshgrid and define index locations with linspace or arange
    x /= m
    y /= m #shouldnt this be divided by n? who knows
    x = np.sqrt(x**2 + y**2)
    x = (x < cutOff).astype(np.float64)
    struct = np.ones((50,50))/(2500) #50 by 50 seems very big, but whatever
    x = cv.filter2D(x, -1, struct)
    x[x<0] = 0
    return np.abs(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(IM))*x))




#NOT FINISHED WITH THIS ONE YET.
#   organoidSegmentation.m
def organoidSegmentation():
    """
    this file contains a stand-alone script to wite [sic] out oncogene images
    calls the other functions in the folder. could probably be put in a "if __name__ == '__main__' " block.
    leave alone for now, will come back to it later.
    """
    return None


#another degenrate function: #probably don't need this really. metadata format should always be consistent and easy to hard-code parsing
#   ParseMetadataFile.m
# def parseMetaDataFile(metadatafilename):
#     """called in getPixelBinCenters"""
#     mData = []
#     chanInfo = {} #since we call attributes/cell labels: try to use dictionary
#     header = [] #all three of these are set up as cells
#     try:
#         with open(metadatafilename, mode='r+') as fid: #no truncation
#             header = fid.readline().strip() #first line with leading and trailing whitespaces removed.
#         header = regexpi(header, r'\t', fmt='split')
#         if header != 'ImageID':
#             print('\nERROR: please choose appropriate metadata file. \nThings will probably crash after this.\n\n')
#             return None
#         with open(metadatafilename) as fid:
#             tmp = np.loadtxt(fid, delimiter='\t', skiprows=1, dtype={'names': [f'col{i}' for i in range(0, header.size)], 'formats': ['S4' for i in range(0, header.size)]} )
#     except Exception as e:
#         print(e)
#         print('\nERROR reading metadata file did not work.\n\n')
#     for i in range(0, tmp.shape[1]):
#         mData.append(tmp[0, i]) #may or may not work properly.
#     chanInfo['channelColNumber'] = input('Select channels (type indices of choice separated by spaces) :').split() #list dialog box pops up. shows options of list of header entries, allows multiple selection mode, prompts to select channels. returns indices of selected channels in header.
#     chanInfo['channelColNumber'] = np.array([int(elem) for elem in chanInfo['channelColNumber']]) #array form of list of str numbers converted to int. (lets try this for now, seems close enough)
#     chanInfo['channelNames'] = header[0, chanInfo['channnelColNumber']]
#     chanInfo['channelColors'] = [f'color{i+1}' for i in range(0, chanInfo['channelColNumber'].size)]
#     ii = header == 'ImageID' #should be case insensitive though
#     imageID = mData[:, ii].astype(np.float64)
#     stackCol = np.logical_or((header == 'Stacks'), (header == 'Stacks')) #should also be case insensitive unfortunately
#     uImageID = np.unique(imageID)
#     for i in range(0, uImageID.size):
#         ii = imageID == uImageID[i]
#         kk = mData[ii, stackCol].astype(np.float64)
#         tmp = mData[ii, :]
#         kk = np.argsort(kk)
#         mData[ii, :] = tmp[kk, :]
#     ii = header == 'ImageID' #also needs to be case insensitive
#     imageID = mData[:, ii].astype(np.uint8) #hpe this works, might need to do as list comprehension instead
#     return mData, imageID, header, chanInfo
















