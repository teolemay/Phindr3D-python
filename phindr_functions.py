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
import glob
from sklearn import cluster

#my own functions:

Generator = np.random.default_rng()

#old version with lists
# def get_files(folder_path, chan_mark='ch', slice_mark='p'):
#     """
#     this function gets the tiff files in a folder and organizes them into image/slice/channels in a nested array, according to the file naming convention used in the sample data I have

#     numpy object array used for ease of indexing.

#     file names in folder MUST follow same pattern ### This will NOT work for many types of images desired.
#     """
#     tif_files = glob.glob(f'{folder_path}\*.tiff')
#     channel_pos = tif_files[0].rfind(chan_mark)
#     slice_pos = tif_files[0][:channel_pos].rfind(slice_mark)
#     path_frame = []
#     imageIDs = []
#     for file in tif_files:
#         if file[:slice_pos] not in imageIDs:
#             imageIDs.append(file[:slice_pos])
#             imgs_in_group = []
#             for fil in tif_files:
#                 if fil[:slice_pos] == file[:slice_pos]:
#                     imgs_in_group.append(fil)
#             slices = []
#             seen_slices = []
#             for slice in imgs_in_group:
#                 if slice[slice_pos:channel_pos] not in seen_slices:
#                     channels = []
#                     seen_slices.append(slice[slice_pos:channel_pos])
#                     for fi in imgs_in_group:
#                         if fi[:channel_pos] == slice[:channel_pos]:
#                             channels.append(fi)
#                     slices.append(channels)
#             path_frame.append(slices)
#     # path_frame = np.asarray(path_frame, dtype='object')
#     # imageIDs = np.asarray(imageIDs, dtype='object')
#     path_frame = np.array(path_frame, dtype='object')
#     imageIDs = np.array(imageIDs, dtype='object')
#     if path_frame.ndim != 3:
#         print('\nInconsistent image/channel dimensions. File loading failed.')
#     return path_frame, imageIDs

#new version with dictionaries
def get_files(folder_path, ID_pos=None, ID_mark='ImageID', ID_markextra=None, treat_mark='None', slice_mark='p', chan_mark='ch'):
    """
    lets see if I can modify this to work with uneven file name lengths
    I think it will be best to rebuild this system using nested dictionaries rather than lists (keep better track of progress)

    ### IT IS ASSUMED IN THIS FUNCTION THAT ALL MARKERS (ID, ZSLICE, CHANNEL, ETC.) END WITH NUMBERS THAT HELP MAKE THEM UNIQUE. ###

    :param folder_path: str, path to desired folder
    :param ID_mark: str or None, Put None if using other ID matching method string to match after which would be image ID. IF NO EXPLICIT IMAGE ID, THEN PUT 'start' -> will take all vals until the first other info mark.
    :param chan_mark: str, string to match after which would be channel number
    :param slic_mark: str, string to match after which would be 
    :param ID_pos: str or None, if str, "start" means find ids starting from beginning of filenames. "end" means find IDs at end of filenames
    :param ID_markextra: str or None, if str, then give secondary ID marker. (useful if image other image identifiers dont distinguish different objects)
    
    path_frame is nested dictionary: image id -> zslice -> channel -> image path
    also returns list of all image ids
    treatmentids is dictionary: treatment type -> all image ids with that treatment
    idstreatment is dictionary: image id -> which type of treatment the image has.
    """
    tif_files = glob.glob(f'{folder_path}\*.tiff')
    path_frame = {} #nested array/list framework to navigate to get image paths
    if treat_mark != None:
        treatmentids = {}
        idstreatment = {}
    else:
        treatmentids = []
        idstreatment = []
    for afile in tif_files:
        file = afile.removeprefix(f'{folder_path}\\')
        idlabelstart = ''
        idlabelend = ''
        idlabelmark = ''
        idlabelmarkextra= ''
        if ID_pos == 'start':
            tmpsli = re.search(slice_mark, file).start()
            tmpcha = re.search(chan_mark, file).start()
            idend = min(tmpsli, tmpcha)
            idlabelstart = file[:idend]
        if ID_pos == 'end':
            idstart = re.search(ID_mark, file).start()
            idend = re.search('.tiff', file).start()
            idlabelend = file[idstart: idend]
        if ID_mark != None:
            m1 = re.search(ID_mark, file)
            idstart = m1.start()
            m = re.search(r'\D', file[m1.end():])
            idend = m.start() + m1.end()
            idlabelmark = file[idstart:idend] #string
        if ID_markextra != None:
            m1 = re.search(ID_markextra, file)
            idstart = m1.start()
            m = re.search(r'\D', file[m1.end():])
            idend = m.start() + m1.end()
            idlabelmarkextra = file[idstart:idend]
        id = idlabelstart + idlabelend + idlabelmark + idlabelmarkextra
        m = re.search(slice_mark, file)
        slicestart=m.end()
        m = re.search(r'\D', file[slicestart:])
        sliceend = m.start() + slicestart
        slice = int(file[slicestart:sliceend]) #integer
        m = re.search(chan_mark, file)
        chanstart = m.end()
        m = re.search(r'\D', file[chanstart:])
        chanend = m.start() + chanstart
        chan = int(file[chanstart:chanend]) #integer
        if id not in path_frame: #new ID -> initiate new dictionary sequence
            path_frame[id] = {slice:{chan:afile}}
            if treat_mark != None:
                m1 = re.search(treat_mark, file)
                treatstart = m1.start()
                m = re.search('_', file[treatstart:]) #the end with underscore is hardcoded here for right now.
                treatend = m.start() + m1.start()
                treatval = file[treatstart:treatend]
                idstreatment[id] = treatval
                if treatval not in treatmentids:
                    treatmentids[treatval] = [id]
                else:
                    treatmentids[treatval].append(id)
        else: #id already exists, but new slice -> add slice entry to id dict, initiate new channel dict
            if slice not in path_frame[id]:
                path_frame[id][slice] = {chan:afile}
            else: #only take the first exemplar of each ## this is to deal with possibly having multiple Objectives/objects in a field. For these, take only the first for simplicity.
                if chan not in path_frame[id][slice]: #add if the channel is not already represented. if its already there, this is likely a different object.
                    path_frame[id][slice][chan] = afile
        
    return path_frame, list(path_frame.keys()), treatmentids, idstreatment


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
        self.trainingColforImageCategories = []
        self.superVoxelPerField = self.randTrainingSuperVoxel//self.randTrainingFields
        self.lowerbound = [0, 0, 0]
        self.upperbound = [1, 1, 1]
        self.numChannels = 3 #keep this here for now since it doesnt seem to be computed early enough in my implementation

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

#this one was taken directly from a matlab vectorized version
# better for large arrays 
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

        pad_row = int(row_ext != 0)*(nrows - row_ext)
        pad_col = int(col_ext != 0)*(ncols - col_ext)

        A1 = np.zeros((arr.shape[0]+pad_row, arr.shape[1]+pad_col))
        A1[:arr.shape[0], :arr.shape[1]] = arr

        t1 = np.reshape(A1, (nrows, int(A1.shape[0]/nrows), -1))
        t1tilde = np.transpose(t1.copy(), (0, 2, 1))
        t2 = np.reshape(t1tilde, (t1.shape[0]*t1.shape[2], -1))
        t3 = np.transpose(np.reshape(t2, (nele, int(t2.shape[0]/nele), -1)), (0, 2, 1))
        return np.reshape(t3, (nele, -1)).T

#better for small arrays
# def im2col(Arr, kernel_shape, type='distinct'):
#     if type =='distinct':
#         kernrows = kernel_shape[1]
#         kerncols = kernel_shape[0]

#         #try to not use padding, since images should already be cropped no?
#         kernsize = int(kernrows*kerncols)
#         out = np.zeros((int(Arr.size/kernsize), kernsize))
#         outrow = 0
#         i=0
#         j=0
#         while i < Arr.shape[1]:
#             while j < Arr.shape[0]:
#                 kernel_vals = Arr[i:i+kernrows, j:j+kerncols]
#                 new_row = kernel_vals.ravel()
#                 out[outrow, :] = new_row
#                 outrow += 1
#                 j += kernrows
#             i += kerncols
#             j=0
#     return out




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


#   equalTrainingSamplePartition.m
def equalTrainingSamplePartition(groups, samplesize):
    par = par_class()
    par.training = np.full((groups.size, 1), False)
    par.test = np.full((groups.size, 1), False)
    uniqueGroups = np.unique(groups[groups > 0])[0]
    for iGroups in range(0, uniqueGroups.size):
        ind = np.argwhere(groups == uniqueGroups[iGroups])
        p = Generator.permutation(ind.size)
        par.training[ind[p[:samplesize]], 0] = True #may or may not have to remove the ,0 at the end.
        par.test[ind[p[samplesize:]], 0] = True
    return par

#   extractImageLevelTextureFeatures.m
def extractImageLevelTextureFeatures(mData, allImageId, param, outputFileName='imagefeatures.csv', outputDir=''):
    param.correctshade = 0
    if param.countBackground:
        totalBins = param.numMegaVoxelBins + 1
    else:
        totalBins = param.numMegaVoxelBins
    uniqueImageID = allImageId
    resultIM = np.zeros((len(uniqueImageID), totalBins)) #for all images: put megavoxel frequencies
    resultRaw = np.zeros((len(uniqueImageID), totalBins))
    # metaIndexTmp = np.empty((uniqueImageID.size, mData.shape[1]), dtype='object') #still not really sure what to put here
    useTreatment=False
    if len(param.allTreatments) > 0:
        useTreatment = True
        Treatments = []
    for iImages in range(0, len(uniqueImageID)):
        # tImageAnal = time.time() #want to time this for some reason
        ii = uniqueImageID[iImages]
        slicekeys = list(mData[ii].keys())
        d, param.fmt = getImageInformation( mData[ii][slicekeys[0]] )
        d[2] = len(mData[ii])
        param = getTileInfo(d, param)
        tmpInfoTable = mData[ii] #this is all the slices and channels in one 3d image
        superVoxelProfile, fgSuperVoxel, Tass = getTileProfiles(tmpInfoTable, param.pixelBinCenters, param, ii)
        megaVoxelProfile, fgMegaVoxel = getMegaVoxelProfile(superVoxelProfile, fgSuperVoxel, param)
        imgProfile, rawProfile = getImageProfile(megaVoxelProfile, fgMegaVoxel, param)
        resultIM[iImages, :] = imgProfile
        resultRaw[iImages, :] = rawProfile
        if useTreatment:
            Treatments.append(param.imageTreatments[ii])
        # tmp = mData[ii] #this is never used.
        # metaIndexTmp[iImages, :] = tmp[0, :] # i dont really get what this is for, I have omitted it from the output file as of 2021-11-27
        # averageTime = averageTime + (time.time() - tImageAnal)
        # print('time remaining:', (uniqueImageID.size -iImages-1)*(averageTime/(iImages+1)), 's')  #time left update i guess
    numRawMV = np.sum(resultRaw, axis=1) #one value per image, gives number of megavoxels
    #dataheader is just megavoxel labels to use as a column header in the output file 

    #want to remove ImageID if existis in Metadata (i dont think this an issue that could come up here.)
    # ii = param.metaDataHeader === 'ImageID'
    # param.metaDataHeader = param.metaDataHeader
    # resultTmp = concatenate: unique ImageID | numRawMv | resultIM (might as well just leave these separate)
    # resultHeader = metadataheader (dont really have one at this point) | ImageID | numMV | dataHeaderIM ()
    #write outputs to file. pd.DataFrame would be VERY good for this part. have to figure out how to integrate properly
    #should probably try to put these together in a dataframe:
    #still dont have anything of type metadataheader, or anything to do with 
    dictResults = {
        'ImageID':uniqueImageID
    }
    if useTreatment:
            dictResults['treatment'] = Treatments
    dictResults['numMV'] = numRawMV
    for i in range(resultIM.shape[1]):
        mvlabel = f'MV{i+1}'
        dictResults[mvlabel] = resultIM[:, i] #e.g. mv cat 1: for each image, put here frequency of mvs of type 1.
    df = pd.DataFrame(dictResults)
    df.dropna(inplace=True)
    csv_name = outputFileName
    if len(outputDir) > 0:
        csv_name = outputDir + '\\' + csv_name
    if csv_name[-4:] != '.csv':
        csv_name = csv_name + '.csv'
    df.to_csv(csv_name) 
    print('\nAll done.')
    return param, resultIM, resultRaw, df #, metaIndexTmp

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
    # x = 
    # y =
    # z = 
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
    fileNames is a dict: {chan#:path, chan#:path, chan#:path}
    """
    fileNames = list(fileNames.values())
    if len(fileNames) == 0:
        print('File name empty')
    d = np.ones(3, dtype=int)
    fmt = 'tif'
    imFileName = fileNames[0] #just want the name of the first file.
    if isinstance(imFileName, dict): # then filenames is a list of dicts
        imFileName = list(imFileName.values()) # list of values of imFileNames if it was a dict (channel paths)
        imFileName = imFileName[0]
    d[2] = len(fileNames) #this is number of z stacks #will often be overwritten later.
    info = imfinfo(imFileName) #imfinfo is matlab built-in.
    d[0] = info.Height 
    d[1] = info.Width
    return d, fmt

#   getImageProfile.m
def getImageProfile(megaVoxelProfile, fgMegaVoxel, param):
    """called in extractImageLevelTextureFeatures"""
    """
    provides multi-parametric representation of image based on megavoxel categories
    """
    tmp1 = np.array([mat_dot(param.megaVoxelBincenters, param.megaVoxelBincenters, axis=1)]).T 
    tmp2 = mat_dot(megaVoxelProfile[fgMegaVoxel], megaVoxelProfile[fgMegaVoxel], axis=1) 
    a = np.add(tmp1, tmp2).T - (2*(megaVoxelProfile[fgMegaVoxel] @ param.megaVoxelBincenters.T))  #this might be wwrong shape. (maybe not tho)
    minDis = np.argmin(a, axis=1)
    x = np.zeros((megaVoxelProfile.shape[0], 1))
    x[fgMegaVoxel, 0] = minDis
    numbins = param.numMegaVoxelBins
    tmp = np.zeros(numbins+1)
    for i in range(0, numbins+1):
        tmp[i] = np.sum(x[fgMegaVoxel, 0] == (i-1))
    imageProfile = tmp
    if not param.countBackground:
        rawProfile = imageProfile[1:].copy()
        imageProfile = imageProfile[1:]
    else:
        rawProfile = imageProfile.copy()
    imageProfile = imageProfile / np.sum(imageProfile) #normalize the image profile
    return imageProfile, rawProfile

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
def getImageThresholdValues(mData, allImageId, param):
    """
    get image threshold values for dataset.  
    """
    intensityThresholdValues = np.full((5000, param.numChannels), np.nan) #not sure why we want 5000 rows
    startVal = 0
    endVal = 0
    for iImages in range(0, len(param.randFieldID)): #for each of the randomly selected images chosen earlier:
        ii = param.randFieldID[iImages] #image_ID #i
        xx = mData[ii] #dict of all the slices corresponding to image with imageID ii
        if len(xx) == 0:
            print('SSS') #dont know about this one, seems pointless
        d, param.fmt = getImageInformation(xx)
        param = getTileInfo(d, param)
        iTmp = getIndividualChannelThreshold(xx, param, ii=ii) #use mData[ii][0] because we obviously only want 1 image at a time, and mData contains all the 3d images so that leaves an extra layer of list to get through.
        intensityThresholdValues[startVal:endVal+iTmp.shape[0], :] = iTmp
        startVal += iTmp.shape[0]
        endVal += iTmp.shape[0]
    ii  = (intensityThresholdValues[:, 0] == np.nan) == False #ii is where not nan. Im not sure why they chose to write it like this tho.
    param.intensityThreshold = np.mean(intensityThresholdValues[np.isfinite(intensityThresholdValues).any(axis=1)], axis=0) # remember everythiing gets rescaled from 0 to 1 #drop rows containing nan, then take medians for each channel#intensityThresholdValues[ii]
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
    slicekeys = list(filenames.keys())
    channelkeys = list(filenames[slicekeys[0]].keys())
    numberChannels = len(channelkeys)
    thresh = np.zeros((len(slicekeys), numberChannels))
    if ii == None:
        ii = np.full(thresh.shape, True)
    if param.intensityNormPerTreatment:
        grpVal = np.argwhere(param.allTreatments == param.imageTreatments[ii])
    for iImages in range(0, len(slicekeys)):
        for iChannels in range(0, numberChannels):
            IM = io.imread(filenames[slicekeys[iImages]][channelkeys[iChannels]]) 
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
    for iImages in range(0, len(param.randFieldID)):
        ii = param.randFieldID[iImages]
        slicekeys = list(mData[ii].keys())
        d, param.fmt = getImageInformation(mData[ii][slicekeys[0]])
        d[2] = len(mData[ii])
        param = getTileInfo(d, param)
        tmpInfoTable = mData[ii]
        superVoxelProfile, fgSuperVoxel, Tass = getTileProfiles(tmpInfoTable, param.pixelBinCenters, param, ii)
        megaVoxelProfile, fgMegaVoxel = getMegaVoxelProfile(superVoxelProfile, fgSuperVoxel, param)
        if len(MegaVoxelsforTraining) == 0:
            MegaVoxelsforTraining = megaVoxelProfile[fgMegaVoxel]
        else:
            MegaVoxelsforTraining = np.concatenate((MegaVoxelsforTraining, megaVoxelProfile[fgMegaVoxel]))
    param.megaVoxelBincenters = getPixelBins(MegaVoxelsforTraining, param.numMegaVoxelBins)
    return param

#   getMegaVoxelProfile.m
def getMegaVoxelProfile(tileProfile, fgSuperVoxel, param):
    """called in extractImageLevelTextureFeatures"""
    """called in getMegaVoxelBinCenters"""
    temp1 = np.array([mat_dot(param.supervoxelBincenters, param.supervoxelBincenters, axis=1)]).T
    temp2 = mat_dot(tileProfile[fgSuperVoxel], tileProfile[fgSuperVoxel], axis=1)
    a = np.add(temp1, temp2).T - 2*(tileProfile[fgSuperVoxel] @ param.supervoxelBincenters.T)
    minDis = np.argmin(a, axis=1)
    x = np.zeros(tileProfile.shape[0])
    x[fgSuperVoxel] = minDis
    x = np.reshape(x, (int(param.croppedX/param.tileX), int(param.croppedY/param.tileY), int(param.croppedZ/param.tileZ)))
    if param.showImage:
        plt.figure()
        plt.imshow(x[:, :, 2], 'gray')
        plt.show()
    x = np.concatenate([ np.zeros((param.superVoxelXAddStart, x.shape[1], x.shape[2])), x, np.zeros((param.superVoxelXAddEnd, x.shape[1], x.shape[2])) ], axis=0)
    x = np.concatenate([ np.zeros((x.shape[0], param.superVoxelYAddStart, x.shape[2])), x, np.zeros((x.shape[0], param.superVoxelYAddEnd, x.shape[2])) ], axis=1)
    x = np.concatenate([ np.zeros((x.shape[0], x.shape[1], param.superVoxelZAddStart)), x, np.zeros((x.shape[0], x.shape[1], param.superVoxelZAddEnd)) ], axis=2)
    x = x.astype(np.uint8)
    param.numMegaVoxelsXY = int(x.shape[0] * x.shape[1] / (param.megaVoxelTileY * param.megaVoxelTileX))
    param.numMegaVoxels = int((param.numMegaVoxelsXY*x.shape[2])/param.megaVoxelTileZ)
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
    for iSuperVoxelImagesZ in range(0, x.shape[2]):
        sliceCounter += 1
        tmpData[:, startCol:endCol] = im2col(x[:, :, iSuperVoxelImagesZ], (param.megaVoxelTileX, param.megaVoxelTileY), type='distinct')
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
    megaVoxelProfile = np.divide(megaVoxelProfile, np.array([np.sum(megaVoxelProfile, axis=1)]).T) #hopefully this works, they ask for elementwise, but the arrays seem to have different shapes.
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

    #bincenters should be pixel categories

    allImageId is image ids of each image stack
    mData is metadata.
    param should be some parameter class object.
    """
    pixelsForTraining = np.zeros((300000, param.numChannels)) # long array [channel 1[very long zeros...], channel2[very long zeros...], channel3[very long zeros ...]] by 3 channels
    startVal = 0
    endVal = 0
    for iImages in range(len(param.randFieldID)): #for each i in range (length of training image set)
        ii = param.randFieldID[iImages]  #image name
        slicekeys = list(mData[ii].keys()) #each z slice corresponding to the image
        d, param.fmt = getImageInformation( mData[ii][slicekeys[0]] )
        d[2] = len(mData[ii])
        param = getTileInfo(d, param)
        param.randZForTraining = len(mData[ii])//2 #want to take half (floor) of the z slices
        iTmp = getTrainingPixels(mData[ii], param, ii) #load correct 3d 3 channel image. should be 3 channels with flatenned image in each.
        pixelsForTraining[startVal:endVal+iTmp.shape[0], :] = iTmp #add to list.
        startVal += iTmp.shape[0]
        endVal += iTmp.shape[0]
    pixelsForTraining = pixelsForTraining[np.sum(pixelsForTraining, axis=1) > 0, :] #this step gets rid of background pixels AND all the extra zeros left over at the end.
    param.pixelBinCenters = getPixelBins(pixelsForTraining, param.numVoxelBins)
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
    % Use kmeans clustering to get  (looks like it is using kmeans++ algorithm) # use sklearn kmeans (gives same results as matlab wit henough repeats!)
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
            kmeans = cluster.KMeans(n_clusters=numBins, n_init=100, max_iter=100).fit(randpermX)
            binCenters[:, :, iRandCycle] = kmeans.cluster_centers_
            temp1 = np.add(np.array([mat_dot(binCenters[:, :, numRandRpt-1], binCenters[:, :, numRandRpt-1], axis=1)]).T, mat_dot(x, x, axis=1)).T #still not sure which one of this or the next should be transposed
            temp2 = 2*(x @ binCenters[:, :, numRandRpt-1].T)
            a = (temp1 - temp2)
            sumD[iRandCycle] = np.sum(np.amin(a, axis=1))
        minDis = np.argmin(sumD)
        binCenters = binCenters[:, :, minDis]
    else: 
        kmeans = cluster.KMeans(n_clusters=numBins, n_init=100, max_iter=100).fit(x)
        binCenters = kmeans.cluster_centers_ 
    return np.abs(binCenters)

#   getScalingFactorforImages.m
def getScalingFactorforImages(metadata, allImageID, param):
    """
    compute lower and higher scaling values for each image
    param: structure of parameter value
    metadata: Metadata

    ### SHOULD MOSTLY WORK, THE INTENSITYNORMPERGTREATMENT AND GRP VAL THINGS ARE NOT YET PROPERLY EXAMINED.
    """
    if param.intensityNormPerTreatment:
        randFieldIDforNormalization, treatmentVals = getTrainingFields(metadata, param, allImageID, param.treatmentColNameForNormalization) #choose images for scaling
        grpVal = np.zeros(randFieldIDforNormalization.size)
    else:
        randFieldIDforNormalization, treatmentVals = getTrainingFields(metadata, param, allImageID, param.trainingColforImageCategories)
    minChannel = np.zeros((randFieldIDforNormalization.size, param.numChannels)) #min values of all selected images in all channels
    maxChannel = np.zeros((randFieldIDforNormalization.size, param.numChannels)) #max values of all selected images in all channels
    numImages = randFieldIDforNormalization.size
    for i in range(0, numImages):
        # which images 
        ii = randFieldIDforNormalization[i] #currently allImageID is a list of image IDs (grouping above 2d image slices)
        filenames = metadata[ii] #file dict of slices/channels { slice1:{channels}, slice2:{channels} }
        slices = list(filenames.keys()) #all the slice key values
        channels = list(filenames[slices[0]].keys()) #all the channel key values
        if len(filenames) == 0:
            print('SSS (no filenames)')
        # print(filenames[0, 0]) #in matlab this just returns the value at 0, 0 in the cell. for now, keep as print
        if i == 0:
            d, fmt = getImageInformation(filenames[slices[0]]) #filenames[0, 0] is selected group 0, stack 0, list of all channel images
        d[2] = len(filenames) #number of z stacks
        param = getTileInfo(d, param) 
        randZ = int(d[2]//2 )
        randZ = Generator.choice(int(d[2]), size=randZ, replace=False, shuffle=False) #choose half of the stack, randomly
        filenames = [filenames[slices[i]] for i in randZ] #list of dicts [{chan1:'path', ...}, {chan1:'path', ...}, ...] actual value of the slice position doesnt matter here.
        minVal = np.full((len(filenames), len(filenames[0])), np.inf)
        maxVal = np.full((len(filenames), len(filenames[0])), -1*np.inf)
        for j in range(0, len(filenames)): 
            for k in range(0, len(filenames[0])):
                IM = io.imread(filenames[j][channels[k]])
                minVal[j, k] = min(minVal[j, k], np.quantile(IM, 0.01))
                maxVal[j, k] = max(maxVal[j, k], np.quantile(IM, 0.99))
        minChannel[i, :] = np.amin(minVal, axis=0)
        maxChannel[i, :] = np.amax(maxVal, axis=0)
        if param.intensityNormPerTreatment:
            #index of the treatment for this image in the list of all treatments
            grpVal[i] = np.argwhere(param.allTreatments == param.imageTreatments[ii]) #param.imageTreatments[ii] is the treatment of the current image
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
        param.lowerbound = np.quantile(minChannel, 0.01, axis=0)
        param.upperbound = np.quantile(maxChannel, 0.99, axis=0)
    param.randFieldID = randFieldIDforNormalization #added this here because I dont know where else this would be determined.
    return param

#   getSuperVoxelBinCenters.m
def getSuperVoxelBinCenters(mData, allImageId, param):
    """
    compute bin centers for super voxels
    % mData  - Metadata
    % allImageID - Image ID's of each image stack
    % param - All parameters
    """
    param.pixelBinCenterDifferences = np.array([mat_dot(param.pixelBinCenters, param.pixelBinCenters, axis=1)]).T  # do the array of list of array trick to increase dimensionality of pixelBinCenterDiff  so that the transpose is actually different from the original array. 
    tilesForTraining = []   #(cont. above) this trick lets me broadcast it together with another array of different length in np.add()
    for iImages in range(0, param.randFieldID.size): # for each 3d 3 channel image in the training set
        ii = param.randFieldID[iImages]
        slicekeys = list(mData[ii].keys())
        d, param.fmt = getImageInformation(mData[ii][slicekeys[0]])
        d[2] = len(mData[ii])
        param = getTileInfo(d, param)
        tmpInfoTable = mData[ii]
        superVoxelProfile, fgSuperVoxel, TASS = getTileProfiles(tmpInfoTable, param.pixelBinCenters, param, ii)
        tmp = superVoxelProfile[fgSuperVoxel] #i think fg means foreground
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
    """called in getSuperVoxelBinCenters"""
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
    % should be higher than the respective threshold
    % TASScores: If TAS score is sleected
    """
    numTilesXY = int((param.croppedX*param.croppedY)/(param.tileX*param.tileY)) #why not just use param.numSuperVoxelsXY, I Have not idea. the calculation is the exact same.
    zEnd = -param.zOffsetEnd
    if zEnd == -0:
        zEnd = None
    filenames = list(filenames.values())
    channelkeys = list(filenames[0].keys())
    filenames = filenames[param.zOffsetStart:zEnd] #keep z stacks that are divisible by stack count
    sliceCounter = 0
    startVal = 0
    endVal=numTilesXY

    startCol= 0
    endCol = param.tileX*param.tileY
    if param.intensityNormPerTreatment:
        grpVal = np.argwhere(param.allTreatments == param.imageTreatments[ii])
    superVoxelProfile = np.zeros((param.numSuperVoxels, param.numVoxelBins+1))
    fgSuperVoxel = np.zeros(param.numSuperVoxels)
    if param.computeTAS:
        categoricalImage = np.zeros((param.croppedX, param.croppedY, param.croppedZ))
    #loop over file names and extract super voxels
    tmpData = np.zeros((numTilesXY, int(param.tileX*param.tileY*param.tileZ)))
    for iImages in range(0, len(filenames)):
        sliceCounter += 1
        croppedIM = np.zeros((param.origX, param.origY, param.numChannels)) #just one slice in all 3 channels
        for jChannels in range(0, param.numChannels):
            try:
                if param.intensityNormPerTreatment:
                    croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(filenames[iImages][channelkeys[jChannels]], param.fmt), low=param.lowerbound[grpVal, jChannels], high=param.upperbound[grpVal, jChannels])
                else:
                    croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(filenames[iImages][channelkeys[jChannels]], param.fmt), low=param.lowerbound[jChannels], high=param.upperbound[jChannels])
            except Exception as e:
                print(e)
                print('file ->', filenames[iImages][channelkeys[jChannels]])
        xEnd = -param.xOffsetEnd
        if xEnd == -0:
            xEnd = None   #if the end index is -0, you just index from 1 to behind 1 and get an empty array. change to 0 if the dimOffsetEnd value is 0.
        yEnd = -param.yOffsetEnd
        if yEnd == -0:
            yEnd = None
        croppedIM = croppedIM[param.xOffsetStart:xEnd, param.yOffsetStart:yEnd, :] #z portion of the offset has already been done by not loading the wrong slices
        x = np.reshape(croppedIM, (param.croppedX*param.croppedY, param.numChannels))
        fg = np.sum(x > param.intensityThreshold, axis=1) >= 1 #want to be greater than threshold in at least 1 channel
        pixelCategory = np.argmin(np.add(param.pixelBinCenterDifferences, mat_dot(x[fg,:], x[fg,:], axis=1)).T - 2*(x[fg,:] @ pixelBinCenters.T), axis=1) 
        x = np.zeros(param.croppedX*param.croppedY, dtype='uint8')
        x[fg] = pixelCategory
        if param.computeTAS:
            categoricalImage[:, :, iImages] = np.reshape(x, param.croppedX, param.croppedY)
        tmpData[:, startCol:endCol] = im2col(np.reshape(x, (param.croppedX, param.croppedY)), (param.tileX, param.tileY), type='distinct')
        startCol += (param.tileX*param.tileY)
        endCol += (param.tileX*param.tileY)
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
    if not param.countBackground:
        superVoxelProfile = superVoxelProfile[:, 1:]
    superVoxelProfile = np.divide(superVoxelProfile, np.array([np.sum(superVoxelProfile, axis=1)]).T) #dont worry about divide by zero errors, they are supposed to happen here!
    superVoxelProfile[superVoxelProfile == np.nan] = 0
    if param.computeTAS:
        TASScores = getCategoricalTASScores[categoricalImage, param.numVoxelBins]
    else:
        TASScores = np.zeros((1, 27*param.numVoxelBins))
    fgSuperVoxel = fgSuperVoxel.astype(bool)
    return superVoxelProfile, fgSuperVoxel.T, TASScores #TASScores is optional

#   getTrainingfields.m
def getTrainingFields(metaInfo, param, allImageId, treatmentColumn):
    """
    called in getScalingFactorforImages

    get smaller subset of images (usually 10) to define parameters for further analysis
        (nly used for scaling factors to scale down intensities from 0 to 1)
    """
    uniqueImageID = np.array(allImageId, dtype='object') #all image ID is currently a list of image ids (for 3d images)
    randomImage = False
    if len(allImageId) == 0:
        randomImage = True
    if len(treatmentColumn) == 0: #can use len instead of size as in original code because it should be a column vector
        randomImage = True
    if randomImage:
        randFieldID = np.array([uniqueImageID[i] for i in Generator.choice(uniqueImageID.size, size=param.randTrainingFields, replace=False, shuffle=False)])
    else:
        #have different treatments, want to choose training images from each treatment.
        numtreatments = len(param.allTreatments)
        param.randTrainingPerTreatment = -(-param.randTrainingFields//numtreatments) #ceiling division
        randFieldID = []
        for i in range(0, numtreatments):
            tmp = treatmentColumn[param.allTreatments[i]] #all image IDs corresponding to this treatment
            randFieldID = randFieldID + [tmp[j] for j in Generator.choice(len(tmp), size=param.randTrainingPerTreatment, replace=False, shuffle=False)]
        randFieldID = np.array(randFieldID)
    treatmentValues = [] #each randfield image -> which treatment does it have?
    for i in range(0, randFieldID.size):
        if not randomImage:
            treatmentValues.append(param.imageTreatments[randFieldID[i]])
        else:
            treatmentValues.append('RR')
    return randFieldID, np.array(treatmentValues) #randFieldID is a list of indices of images to choose

#   getTrainingPixels.m
def getTrainingPixels(filenames, param, ii):
    """called in getPixelBinCenters"""
    """
    filenames is dict of slices and channels {slice1:{chans}, slice2:{chans}, ...}
    param is param file
    ii is the image ID for the image that was passed in.
    """
    zslices = list(filenames.values()) #z slices
    channelkeys = list(zslices[0].keys()) #channels
    zslices = np.array([zslices[i] for i in Generator.choice(len(zslices), size=param.randZForTraining, replace=False, shuffle=False)]) #shuffle the slices
    trPixels = np.zeros((param.pixelsPerImage*param.randZForTraining, param.numChannels))
    startVal = 0
    if param.intensityNormPerTreatment:
        grpVal = np.argwhere(param.allTreatments == param.imageTreatments[ii])
    zslices = zslices[:(len(zslices)//2)]
    for iImages in range(0, len(zslices)):
        ####absolutely no point to crop any of these. background pixels get filtered out here anyway. worry about cropping later when super and megavoxels come into play.
            ####below used to be the cropped image setup. didnt work because different images/zslices might have different shapes
        # croppedIM = np.zeros((param.origX, param.origY, param.numChannels))
        if param.intensityNormPerTreatment:
            im0 = rescaleIntensity(io.imread(zslices[iImages][channelkeys[0]], param.fmt), low=param.lowerbound[grpVal, 0], high=param.upperbound[grpVal, 0])
        else:
            im0 = rescaleIntensity(io.imread(zslices[iImages][channelkeys[0]], param.fmt), low=param.lowerbound[0], high=param.upperbound[0])
        croppedIM = np.zeros((im0.size, param.numChannels))
        croppedIM[:, 0] = np.ravel(im0) #load things pre-flattened since there is no point to cropping here yet.
        for jChannels in range(1, param.numChannels):
            if param.intensityNormPerTreatment:
                croppedIM[:, jChannels] = np.ravel(rescaleIntensity(io.imread(zslices[iImages][channelkeys[jChannels]], param.fmt), low=param.lowerbound[grpVal, jChannels], high=param.upperbound[grpVal, jChannels]))
            else:
                croppedIM[:, jChannels] = np.ravel(rescaleIntensity(io.imread(zslices[iImages][channelkeys[jChannels]], param.fmt), low=param.lowerbound[jChannels], high=param.upperbound[jChannels]))
            ####below was nice convenient file loading into 3 channel image.
        #     if param.intensityNormPerTreatment:
        #         croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(zslices[iImages][channelkeys[jChannels]], param.fmt), low=param.lowerbound[grpVal, jChannels], high=param.upperbound[grpVal, jChannels])
        #     else:
        #         croppedIM[:,:, jChannels] = rescaleIntensity(io.imread(zslices[iImages][channelkeys[jChannels]], param.fmt), low=param.lowerbound[jChannels], high=param.upperbound[jChannels])
            ####below was image cropping.
        # xEnd = -param.xOffsetEnd
        # if xEnd == -0:
        #     xEnd = None   #if the end index is -0, you just index from 1 to behind 1 and get an empty array. change to 0 if the dimOffsetEnd value is 0.
        # yEnd = -param.yOffsetEnd
        # if yEnd == -0:
        #     yEnd = None
        # croppedIM = croppedIM[param.xOffsetStart:xEnd, param.yOffsetStart:yEnd, :] #crop
        # croppedIM = np.reshape(croppedIM, (param.croppedX*param.croppedY, param.numChannels)) #flatten
        croppedIM = croppedIM[np.sum(croppedIM > param.intensityThreshold, axis=1) >= param.numChannels/3, :]
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
    imageID = mData[:, ii].astype(np.uint8) #hope this works, might need to do as list comprehension instead
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
            elif wflag == 'Overwrite':
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























