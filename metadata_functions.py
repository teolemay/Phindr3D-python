"""
Teophile Lemay, 2022

This file contains functions to create and parse metadata as used by phindr3D
"""

import os
import re
import pandas as pd
import numpy as np

def createMetadata(folder_path, regex, mdatafilename='metadata_python.txt'):
    """
    This function creates a metadata txt file in the same format as used in the matlab Phindr implementation

    folder_path: path to iamge folder (full or relative)
    regex: regular expression matching image file names. must include named groups for all required image attributes (wellID, field, treatment, channel, stack, etc.)
    Matlab style regex can be adapted by adding P before group names. ex. : "(?P<WellID>\w+)__(?P<Treatment>\w+)__z(?P<Stack>\d+)__ch(?P<Channel>\d)__example.tiff"
    mdatafilename: filename for metadatafile that will be written.

    regex groups MUST INCLUDE Channel and Stack and at least one other image identification group
    regex groups CANNOT include ImageID or _file or MetadataFile.
    """
    f = os.listdir(folder_path)
    metadatafilename = f'{os.path.abspath(folder_path)}\\{mdatafilename}'
    #read images in folder
    rows = []
    for i, file in enumerate(f):
        m = re.fullmatch(regex, file)
        if m != None:
            d = m.groupdict()
            d['_file'] = os.path.abspath(f'{folder_path}\\{file}')
            rows.append(d)
    #make sure rows is not empty and that Channel and Stack are in the groups.
    if len(rows) == 0:
        print('\nFailed to create metadata. No regex matches found in folder.\n')
        return None
    if ('Channel' not in rows[0].keys()) or ('Stack' not in rows[0].keys()):
        print('\nFailed to create metadata. regex must contain "Channel" and "Stack" groups.')
        return None
    tmpdf = pd.DataFrame(rows)  
    #make new dataframe with desired colummns
    tags = tmpdf.columns
    channels = np.unique(tmpdf['Channel'])
    cols = []
    for chan in channels:
        cols.append(f'Channel_{chan}')
    for tag in tags:
        if tag not in ['Channel', 'Stack', '_file']:
            cols.append(tag)
    cols.append('Stack')
    cols.append('MetadataFile')
    df = pd.DataFrame(columns=cols)
    #add data to the new dataframe
    stacksubset = [tag for tag in tags if tag not in ['Channel', '_file']]
    idsubset = [tag for tag in tags if tag not in ['Channel', '_file', 'Stack']]
    df[stacksubset] = tmpdf.drop_duplicates(subset = stacksubset)[stacksubset]
    df.reset_index(inplace=True, drop=True)
    #add unique image ids based on the "other tags"
    idtmp = tmpdf.drop_duplicates(subset = idsubset)[idsubset].reset_index(drop=True)
    idtmp.reset_index(inplace=True)
    idtmp.rename(columns={'index':'ImageID'}, inplace=True)
    idtmp['ImageID'] = idtmp['ImageID'] + 1
    df = pd.merge(df, idtmp, left_on=idsubset, right_on=idsubset)
    #give metadatafilename
    df['MetadataFile'] = metadatafilename
    # fill in file paths for each channel
    for chan in channels:
        chandf = tmpdf[tmpdf['Channel']==chan].reset_index(drop=True)
        df[f'Channel_{chan}'] = chandf['_file']
    df.to_csv(metadatafilename, sep='\t', index=False)
    print(f'Metadata file created at \n{metadatafilename}')
    return metadatafilename


def parseMetadata(metadatapath):
    """
    probably don't need a parse metadata function. I think reading the metadata directly to dataframe and reading the dataframe would suffice.
    """
    return None