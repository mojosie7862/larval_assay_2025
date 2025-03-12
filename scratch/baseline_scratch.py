import numpy as np
import cv2
# from TrackedObject import TrackedObject as TO
import matplotlib.pyplot as plt
import os
import re  # regular expressions library
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects
from skimage import img_as_ubyte
import scipy.misc
import scipy.interpolate
import scipy.io
import pims
from tqdm import tqdm
import pandas as pd
import math
import imageio

import scipy.stats as stats
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import metrics

# set directory containing image files
track_data_dir = 'E:/jm_laCie_larval_behavior_tracked_data/'
inDir = "C:/Users/millard/larval_behavior_data/"

# names of videos to track
Name = '241210-nrxn1a_beta_db_hetinx-BL_.avi'

# Gene name
Gene = "nrxn1a_beta"

# deisgnate which fish are mut/het/wt
cntrl = [6, 10, 18, 25, 26, 33, 34, 39, 41, 47, 53, 55, 59, 60, 64, 70, 74, 75, 77, 82, 87, 88, 96, 97, 100]
het = [1, 4, 5, 7, 8, 11, 12, 17, 20, 21, 22, 23, 24, 27, 32, 36, 37, 38, 40, 44, 48, 50, 56, 57, 63, 66, 67, 68, 69,
       71, 72, 79, 81, 83, 84, 89, 90, 92, 93, 94, 95, 98]
mut = [9, 13, 14, 15, 16, 19, 28, 29, 30, 31, 35, 42, 43, 45, 46, 49, 51, 52, 54, 58, 61, 62, 65, 73, 76, 78, 80, 85,
       86, 91, 99]

wts = np.array(cntrl)
hets = np.array(het)
muts = np.array(mut)

# mut/mut():
# mut/het():
# het/mut():
# het/het(): 23,30,35,41,43,45,46,48,53,58,63,67,69,72
# mut/wt(): 26,27,62,70
# wt/mut(): 55
# wt/het():
# het/wt():
# wt/wt():

# set the directory in which to save the files
SaveTrackingIm = True;  # or False

desttracked = inDir + 'Results/' + Gene + '/' + Name + '/'

# Number of fish to track
nFish = 100

# Number of rows and columns of wells
nRow = 10
nCol = 10

# Number of pixels in x and y direction
nPixel = 1024

# frame rate in Hz
framerate = 20

# time between frames in milliseconds
framelength = 1 / framerate * 1000

# Define block time in seconds
blocksec = 60

# Define number of total blocks
nBlocks = 16

# this is to say if there is no folder, can make one
os.makedirs(os.path.dirname(desttracked), exist_ok=True)

data = np.load(track_data_dir + Name + '/AnalyzedData.npz')

medfiltcentY = data['medfiltcentY']
medfiltcentX = data['medfiltcentX']
disp = data['disp']
filtdisp = data['filtdisp']
thigmo = data['thigmo']
filtthigmo = data['filtthigmo']
centeryesno = data['centeryesno']
movementyesno = data['movementyesno']
movementyesnofill = data['movementyesnofill']
boutsperblock = data['boutsperblock']
totdistperblock = data['totdistperblock']
totdispperblock = data['totdispperblock']
tottimemovingperblock = data['tottimemovingperblock']
totavgspeedperblock = data['totavgspeedperblock']
avgthigmoperblock = data['avgthigmoperblock']
medthigmoperblock = data['medthigmoperblock']
fractionouterperblock = data['fractionouterperblock']
boutsperperiod = data['boutsperperiod']
totdistperperiod = data['totdistperperiod']
totdispperperiod = data['totdispperperiod']
tottimemovingperperiod = data['tottimemovingperperiod']
totavgspeedperperiod = data['totavgspeedperperiod']
avgthigmoperperiod = data['avgthigmoperperiod']
medthigmoperperiod = data['medthigmoperperiod']
fractionouterperperiod = data['fractionouterperperiod']
avgdistperboutperblock = data['avgdistperboutperblock']
avgdispperboutperblock = data['avgdispperboutperblock']
avgtimemovingperboutperblock = data['avgtimemovingperboutperblock']
avgspeedperboutperblock = data['avgspeedperboutperblock']
meddistperboutperblock = data['meddistperboutperblock']
meddispperboutperblock = data['meddispperboutperblock']
medtimemovingperboutperblock = data['medtimemovingperboutperblock']
medspeedperboutperblock = data['medspeedperboutperblock']
avgdistperboutperperiod = data['avgdistperboutperperiod']
avgdispperboutperperiod = data['avgdispperboutperperiod']
avgtimemovingperboutperperiod = data['avgtimemovingperboutperperiod']
avgspeedperboutperperiod = data['avgspeedperboutperperiod']
meddistperboutperperiod = data['meddistperboutperperiod']
meddispperboutperperiod = data['meddispperboutperperiod']
medtimemovingperboutperperiod = data['medtimemovingperboutperperiod']
medspeedperboutperperiod = data['medspeedperboutperperiod']
distbouts = data['distbouts']
dispbouts = data['dispbouts']
timebouts = data['timebouts']
speedbouts = data['speedbouts']

print("Setting up final arrays....")

print("Setting up final dictionaries....")
resultdict = {}

# setup dictionary with total results for all fish based on event
for number in range(nFish):
    geno = np.nan
    if (number + 1) in muts:
        geno = "mut"
    if (number + 1) in hets:
        geno = "het"
    if (number + 1) in wts:
        geno = "wt"
    for i in range(nBlocks + 2):
        if i < nBlocks:
            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'block': (i + 1),
                '#bouts': boutsperblock[number, i],
                'totdist': totdistperblock[number, i],
                'totdisp': totdispperblock[number, i],
                'tottimemvmt': tottimemovingperblock[number, i],
                'totavgspeed': totavgspeedperblock[number, i],
                'avgfromcenter': avgthigmoperblock[number, i],
                'medfromcenter': medthigmoperblock[number, i],
                'fractionouter': fractionouterperblock[number, i],
                'avgdistperbout': avgdistperboutperblock[number, i],
                'meddistperbout': meddistperboutperblock[number, i],
                'avgdispperbout': avgdispperboutperblock[number, i],
                'meddispperbout': meddispperboutperblock[number, i],
                'avgtimeperbout': avgtimemovingperboutperblock[number, i],
                'medtimeperbout': medtimemovingperboutperblock[number, i],
                'avgspeedperbout': avgspeedperboutperblock[number, i],
                'medspeedperbout': medspeedperboutperblock[number, i],
            }

        # last columns will be totals in light then dark (ie per period)
        if i == nBlocks:
            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'block': (i + 1),
                '#bouts': boutsperperiod[number, 0],
                'totdist': totdistperperiod[number, 0],
                'totdisp': totdispperperiod[number, 0],
                'tottimemvmt': tottimemovingperperiod[number, 0],
                'totavgspeed': totavgspeedperperiod[number, 0],
                'avgfromcenter': avgthigmoperperiod[number, 0],
                'medfromcenter': medthigmoperperiod[number, 0],
                'fractionouter': fractionouterperperiod[number, 0],
                'avgdistperbout': avgdistperboutperperiod[number, 0],
                'meddistperbout': meddistperboutperperiod[number, 0],
                'avgdispperbout': avgdispperboutperperiod[number, 0],
                'meddispperbout': meddispperboutperperiod[number, 0],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 0],
                'medtimeperbout': medtimemovingperboutperperiod[number, 0],
                'avgspeedperbout': avgspeedperboutperperiod[number, 0],
                'medspeedperbout': medspeedperboutperperiod[number, 0],
            }

        if i == nBlocks + 1:
            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'block': (i + 1),
                '#bouts': boutsperperiod[number, 1],
                'totdist': totdistperperiod[number, 1],
                'totdisp': totdispperperiod[number, 1],
                'tottimemvmt': tottimemovingperperiod[number, 1],
                'totavgspeed': totavgspeedperperiod[number, 1],
                'avgfromcenter': avgthigmoperperiod[number, 1],
                'medfromcenter': medthigmoperperiod[number, 1],
                'fractionouter': fractionouterperperiod[number, 1],
                'avgdistperbout': avgdistperboutperperiod[number, 1],
                'meddistperbout': meddistperboutperperiod[number, 1],
                'avgdispperbout': avgdispperboutperperiod[number, 1],
                'meddispperbout': meddispperboutperperiod[number, 1],
                'avgtimeperbout': avgtimemovingperboutperperiod[number, 1],
                'medtimeperbout': medtimemovingperboutperperiod[number, 1],
                'avgspeedperbout': avgspeedperboutperperiod[number, 1],
                'medspeedperbout': medspeedperboutperperiod[number, 1],
            }

dfresult = pd.DataFrame(data=resultdict)
dfresultT = dfresult.T

# dictionary for each individual fish
resultdictindiv = {}

# setup dictionary with total results for all fish into dictionary/dataframe
for number in range(nFish):
    geno = np.nan
    if (number + 1) in muts:
        geno = "mut"
    if (number + 1) in hets:
        geno = "het"
    if (number + 1) in wts:
        geno = "wt"
    for i in range(nBlocks + 2):
        if i == 0:
            resultdictindiv[str(number + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-#bouts': boutsperblock[number, i],
                str(i + 1) + '-totdist': totdistperblock[number, i],
                str(i + 1) + '-totdisp': totdispperblock[number, i],
                str(i + 1) + '-tottimemvmt': tottimemovingperblock[number, i],
                str(i + 1) + '-totavgspeed': totavgspeedperblock[number, i],
                str(i + 1) + '-avgfromcenter': avgthigmoperblock[number, i],
                str(i + 1) + '-medfromcenter': medthigmoperblock[number, i],
                str(i + 1) + '-fractionouter': fractionouterperblock[number, i],
                str(i + 1) + '-avgdistperbout': avgdistperboutperblock[number, i],
                str(i + 1) + '-meddistperbout': meddistperboutperblock[number, i],
                str(i + 1) + '-avgdispperbout': avgdispperboutperblock[number, i],
                str(i + 1) + '-meddispperbout': meddispperboutperblock[number, i],
                str(i + 1) + '-avgtimeperbout': avgtimemovingperboutperblock[number, i],
                str(i + 1) + '-medtimeperbout': medtimemovingperboutperblock[number, i],
                str(i + 1) + '-avgspeedperbout': avgspeedperboutperblock[number, i],
                str(i + 1) + '-medspeedperbout': medspeedperboutperblock[number, i],
            }

        if i > 0 and i < nBlocks:
            resultdictindiv[str(number + 1)].update({
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-#bouts': boutsperblock[number, i],
                str(i + 1) + '-totdist': totdistperblock[number, i],
                str(i + 1) + '-totdisp': totdispperblock[number, i],
                str(i + 1) + '-tottimemvmt': tottimemovingperblock[number, i],
                str(i + 1) + '-totavgspeed': totavgspeedperblock[number, i],
                str(i + 1) + '-avgfromcenter': avgthigmoperblock[number, i],
                str(i + 1) + '-medfromcenter': medthigmoperblock[number, i],
                str(i + 1) + '-fractionouter': fractionouterperblock[number, i],
                str(i + 1) + '-avgdistperbout': avgdistperboutperblock[number, i],
                str(i + 1) + '-meddistperbout': meddistperboutperblock[number, i],
                str(i + 1) + '-avgdispperbout': avgdispperboutperblock[number, i],
                str(i + 1) + '-meddispperbout': meddispperboutperblock[number, i],
                str(i + 1) + '-avgtimeperbout': avgtimemovingperboutperblock[number, i],
                str(i + 1) + '-medtimeperbout': medtimemovingperboutperblock[number, i],
                str(i + 1) + '-avgspeedperbout': avgspeedperboutperblock[number, i],
                str(i + 1) + '-medspeedperbout': medspeedperboutperblock[number, i],
            })

            # last columns will be totals in light then dark (ie per period)
        if i == nBlocks:
            resultdictindiv[str(number + 1)].update({
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-#bouts': boutsperperiod[number, 0],
                str(i + 1) + '-totdist': totdistperperiod[number, 0],
                str(i + 1) + '-totdisp': totdispperperiod[number, 0],
                str(i + 1) + '-tottimemvmt': tottimemovingperperiod[number, 0],
                str(i + 1) + '-totavgspeed': totavgspeedperperiod[number, 0],
                str(i + 1) + '-avgfromcenter': avgthigmoperperiod[number, 0],
                str(i + 1) + '-medfromcenter': medthigmoperperiod[number, 0],
                str(i + 1) + '-fractionouter': fractionouterperperiod[number, 0],
                str(i + 1) + '-avgdistperbout': avgdistperboutperperiod[number, 0],
                str(i + 1) + '-meddistperbout': meddistperboutperperiod[number, 0],
                str(i + 1) + '-avgdispperbout': avgdispperboutperperiod[number, 0],
                str(i + 1) + '-meddispperbout': meddispperboutperperiod[number, 0],
                str(i + 1) + '-avgtimeperbout': avgtimemovingperboutperperiod[number, 0],
                str(i + 1) + '-medtimeperbout': medtimemovingperboutperperiod[number, 0],
                str(i + 1) + '-avgspeedperbout': avgspeedperboutperperiod[number, 0],
                str(i + 1) + '-medspeedperbout': medspeedperboutperperiod[number, 0],
            })

        if i == nBlocks + 1:
            resultdictindiv[str(number + 1)].update({
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-#bouts': boutsperperiod[number, 1],
                str(i + 1) + '-totdist': totdistperperiod[number, 1],
                str(i + 1) + '-totdisp': totdispperperiod[number, 1],
                str(i + 1) + '-tottimemvmt': tottimemovingperperiod[number, 1],
                str(i + 1) + '-totavgspeed': totavgspeedperperiod[number, 1],
                str(i + 1) + '-avgfromcenter': avgthigmoperperiod[number, 1],
                str(i + 1) + '-medfromcenter': medthigmoperperiod[number, 1],
                str(i + 1) + '-fractionouter': fractionouterperperiod[number, 1],
                str(i + 1) + '-avgdistperbout': avgdistperboutperperiod[number, 1],
                str(i + 1) + '-meddistperbout': meddistperboutperperiod[number, 1],
                str(i + 1) + '-avgdispperbout': avgdispperboutperperiod[number, 1],
                str(i + 1) + '-meddispperbout': meddispperboutperperiod[number, 1],
                str(i + 1) + '-avgtimeperbout': avgtimemovingperboutperperiod[number, 1],
                str(i + 1) + '-medtimeperbout': medtimemovingperboutperperiod[number, 1],
                str(i + 1) + '-avgspeedperbout': avgspeedperboutperperiod[number, 1],
                str(i + 1) + '-medspeedperbout': medspeedperboutperperiod[number, 1],
            })

dfresultindiv = pd.DataFrame(data=resultdictindiv)
dfresultindivT = dfresultindiv.T

variables = ['#bouts', 'totdist', 'totdisp', 'tottimemvmt', 'totavgspeed', 'avgfromcenter', 'medfromcenter',
             'fractionouter', 'avgdistperbout',
             'meddistperbout', 'avgdispperbout', 'meddispperbout', 'avgtimeperbout', 'medtimeperbout',
             'avgspeedperbout', 'medspeedperbout'
             ]

genoresultdict = {}
statsdict = {}

# setup dictionary with results averaged over genotype into dictionary/dataframe
for number in range(nBlocks + 2):
    for i, element in enumerate(variables):
        if float(dfresultT.loc[
                     (dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [str(element)]].std(
                axis=0)) == 0:
            if float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                    dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)) == 0:
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "mut"] = {
                    'block': (number + 1),
                    'value': str(element),
                    'genotype': "mut",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                         str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                            str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': np.nan,
                    '(avg-sib)/sibstd': np.nan
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "het"] = {
                    'block': (number + 1),
                    'value': str(element),
                    'genotype': "het",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                         str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                            str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': np.nan
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
                    'block': (number + 1),
                    'value': str(element),
                    'genotype': "sib",
                    'count': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['block'] == (number + 1)), [str(element)]].count()),
                    'avg': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                    dfresultT['block'] == (number + 1)), [str(element)]].median(
                        axis=0)),
                    'stdev': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                }
            else:
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "mut"] = {
                    'block': (number + 1),
                    'value': str(element),
                    'genotype': "mut",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                         str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                            str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': np.nan,
                    '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) - float(
                        dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0))) / float(
                        dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0))
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "het"] = {
                    'block': (number + 1),
                    'value': str(element),
                    'genotype': "het",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                         str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                            str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                           str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': np.nan
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
                    'block': (number + 1),
                    'value': str(element),
                    'genotype': "sib",
                    'count': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['block'] == (number + 1)), [str(element)]].count()),
                    'avg': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                    dfresultT['block'] == (number + 1)), [str(element)]].median(
                        axis=0)),
                    'stdev': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                }
        else:
            genoresultdict[str(number + 1) + "-" + str(element) + "-" + "mut"] = {
                'block': (number + 1),
                'value': str(element),
                'genotype': "mut",
                'count': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].count()),
                'avg': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].mean(axis=0)),
                'median': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].median(axis=0)),
                'stdev': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].std(axis=0)),
                '(avg-wt)/wtstd': (float(dfresultT.loc[
                                             (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                                 str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT[
                                                                                                         'genotype'] == 'wt') & (
                                                                                                                dfresultT[
                                                                                                                    'block'] == (
                                                                                                                            number + 1)), [
                    str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                            dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                            dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) - float(dfresultT.loc[((
                                                                                                                                  dfresultT[
                                                                                                                                      'genotype'] == 'het') | (
                                                                                                                                  dfresultT[
                                                                                                                                      'genotype'] == 'wt')) & (
                                                                                                                                 dfresultT[
                                                                                                                                     'block'] == (
                                                                                                                                             number + 1)), [
                    str(element)]].mean(axis=0))) / float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                            dfresultT['genotype'] == 'wt')) & (dfresultT['block'] == (number + 1)), [str(element)]].std(
                    axis=0))
            }
            genoresultdict[str(number + 1) + "-" + str(element) + "-" + "het"] = {
                'block': (number + 1),
                'value': str(element),
                'genotype': "het",
                'count': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].count()),
                'avg': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].mean(axis=0)),
                'median': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].median(axis=0)),
                'stdev': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                    str(element)]].std(axis=0)),
                '(avg-wt)/wtstd': (float(dfresultT.loc[
                                             (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                                 str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT[
                                                                                                         'genotype'] == 'wt') & (
                                                                                                                dfresultT[
                                                                                                                    'block'] == (
                                                                                                                            number + 1)), [
                    str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                            dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0))
            }
        genoresultdict[str(number + 1) + "-" + str(element) + "-" + "wt"] = {
            'block': (number + 1),
            'value': str(element),
            'genotype': "wt",
            'count': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                str(element)]].count()),
            'avg': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                str(element)]].mean(axis=0)),
            'median': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                str(element)]].median(axis=0)),
            'stdev': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                str(element)]].std(axis=0))
        }
        genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
            'block': (number + 1),
            'value': str(element),
            'genotype': "sib",
            'count': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                        dfresultT['block'] == (number + 1)), [str(element)]].count()),
            'avg': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                        dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)),
            'median': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                        dfresultT['block'] == (number + 1)), [str(element)]].median(axis=0)),
            'stdev': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                        dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
        }

        # setup dictionary with statistics for each event and property, removing those with <5 tracked objects in block
        anova = np.array(stats.f_oneway(dfresultT.loc[
                                            (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                                str(element)]].dropna(),
                                        dfresultT.loc[
                                            (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                                str(element)]].dropna(),
                                        dfresultT.loc[
                                            (dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                                                str(element)]].dropna()))

        mut = np.array(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
            str(element)]].dropna())
        het = np.array(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
            str(element)]].dropna())
        wt = np.array(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
            str(element)]].dropna())
        sib = np.array(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                    dfresultT['block'] == (number + 1)), [str(element)]].dropna())

        if anova[1] < 0.05:
            sig = 'YES'
        else:
            sig = 'NO'

        ttestmutvhet = np.array(stats.ttest_ind(mut.astype(float), het.astype(float), equal_var=False))

        ttestmutvwt = np.array(stats.ttest_ind(mut.astype(float), wt.astype(float), equal_var=False))

        ttesthetvwt = np.array(stats.ttest_ind(het.astype(float), wt.astype(float), equal_var=False))

        ttestmutvsib = np.array(stats.ttest_ind(mut.astype(float), sib.astype(float), equal_var=False))

        if ttestmutvhet[1] < 0.05:
            sigmh = 'YES'
        else:
            sigmh = 'NO'

        if ttestmutvwt[1] < 0.05:
            sigmw = 'YES'
        else:
            sigmw = 'NO'

        if ttesthetvwt[1] < 0.05:
            sighw = 'YES'
        else:
            sighw = 'NO'

        if ttestmutvsib[1] < 0.05:
            sigms = 'YES'
        else:
            sigms = 'NO'

        if float(dfresultT.loc[
                     (dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [str(element)]].std(
                axis=0)) == 0:
            if float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                    dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)) == 0:
                statsdict[str("BL-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#wt': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                        str(element)]].count()),
                    'avgmut/avgwt': np.nan,
                    'avgmut/avgsib': np.nan,
                    '(avg-wt)/wtstd': np.nan,
                    '(avg-sib)/sibstd': np.nan,
                    'ANOVA F-value': float(anova[0]),
                    'ANOVA p-value': float(anova[1]),
                    'Sig ANOVA?': str(sig),
                    'mut v het p value': float(ttestmutvhet[1]),
                    'Sig MvH?': str(sigmh),
                    'mut v wt p value': float(ttestmutvwt[1]),
                    'Sig MvW?': str(sigmw),
                    'het v wt p value': float(ttesthetvwt[1]),
                    'Sig HvW?': str(sighw),
                    'mut v sib p value': float(ttestmutvsib[1]),
                    'Sig MvS?': str(sigms)
                }
            else:
                statsdict[str("BL-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#wt': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                        str(element)]].count()),
                    'avgmut/avgwt': np.nan,
                    'avgmut/avgsib': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) / float(
                        dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)),
                    '(avg-wt)/wtstd': np.nan,
                    '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) - float(
                        dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0))) / float(
                        dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                    'ANOVA F-value': float(anova[0]),
                    'ANOVA p-value': float(anova[1]),
                    'Sig ANOVA?': str(sig),
                    'mut v het p value': float(ttestmutvhet[1]),
                    'Sig MvH?': str(sigmh),
                    'mut v wt p value': float(ttestmutvwt[1]),
                    'Sig MvW?': str(sigmw),
                    'het v wt p value': float(ttesthetvwt[1]),
                    'Sig HvW?': str(sighw),
                    'mut v sib p value': float(ttestmutvsib[1]),
                    'Sig MvS?': str(sigms)
                }
        else:
            if float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                    dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)) == 0:
                statsdict[str("BL-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#wt': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                        str(element)]].count()),
                    'avgmut/avgwt': float(dfresultT.loc[
                                              (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                                  str(element)]].mean(axis=0)) / float(dfresultT.loc[(dfresultT[
                                                                                                          'genotype'] == 'wt') & (
                                                                                                                 dfresultT[
                                                                                                                     'block'] == (
                                                                                                                             number + 1)), [
                        str(element)]].mean(axis=0)),
                    'avgmut/avgsib': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) / float(
                        dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)),
                    '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) - float(
                        dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                            str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                    '(avg-sib)/sibstd': np.nan,
                    'ANOVA F-value': float(anova[0]),
                    'ANOVA p-value': float(anova[1]),
                    'Sig ANOVA?': str(sig),
                    'mut v het p value': float(ttestmutvhet[1]),
                    'Sig MvH?': str(sigmh),
                    'mut v wt p value': float(ttestmutvwt[1]),
                    'Sig MvW?': str(sigmw),
                    'het v wt p value': float(ttesthetvwt[1]),
                    'Sig HvW?': str(sighw),
                    'mut v sib p value': float(ttestmutvsib[1]),
                    'Sig MvS?': str(sigms)
                }

            else:
                statsdict[str("BL-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['block'] == (number + 1)), [
                                          str(element)]].count()),
                    '#wt': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                        str(element)]].count()),
                    'avgmut/avgwt': float(dfresultT.loc[
                                              (dfresultT['genotype'] == 'mut') & (dfresultT['block'] == (number + 1)), [
                                                  str(element)]].mean(axis=0)) / float(dfresultT.loc[(dfresultT[
                                                                                                          'genotype'] == 'wt') & (
                                                                                                                 dfresultT[
                                                                                                                     'block'] == (
                                                                                                                             number + 1)), [
                        str(element)]].mean(axis=0)),
                    'avgmut/avgsib': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) / float(
                        dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)),
                    '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) - float(
                        dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['block'] == (number + 1)), [
                            str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                    '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0)) - float(
                        dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].mean(axis=0))) / float(
                        dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                    dfresultT['block'] == (number + 1)), [str(element)]].std(axis=0)),
                    'ANOVA F-value': float(anova[0]),
                    'ANOVA p-value': float(anova[1]),
                    'Sig ANOVA?': str(sig),
                    'mut v het p value': float(ttestmutvhet[1]),
                    'Sig MvH?': str(sigmh),
                    'mut v wt p value': float(ttestmutvwt[1]),
                    'Sig MvW?': str(sigmw),
                    'het v wt p value': float(ttesthetvwt[1]),
                    'Sig HvW?': str(sighw),
                    'mut v sib p value': float(ttestmutvsib[1]),
                    'Sig MvS?': str(sigms)
                }

dfgenoresult = pd.DataFrame(data=genoresultdict)
dfgenoresultT = dfgenoresult.T

dfgenoresultT['block+value'] = dfgenoresultT['block'].apply(str) + "-" + dfgenoresultT['value'].apply(str)

dfstats = pd.DataFrame(data=statsdict)
dfstatsT = dfstats.T

dfstatsanova = dfstatsT.loc[(dfstatsT['Sig ANOVA?'] == 'YES')]
dfstatssigsibs = dfstatsT.loc[(dfstatsT['Sig MvS?'] == 'YES')]
dfstatssighets = dfstatsT.loc[(dfstatsT['Sig ANOVA?'] == 'YES') & (dfstatsT['Sig MvH?'] == 'YES')]
dfstatssigwts = dfstatsT.loc[(dfstatsT['Sig ANOVA?'] == 'YES') & (dfstatsT['Sig MvW?'] == 'YES')]
# if avg for mut would be in top or bottom 10th percentile
dfstatslargediff = dfstatsT.loc[(abs(dfstatsT['(avg-wt)/wtstd']) >= 1.28) | (abs(dfstatsT['(avg-sib)/sibstd']) >= 1.28)]

print("Saving results in spreadsheet....")
dfresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResultsByFishbyEvent.csv')
dfresultindivT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResultsByIndividualFish.csv')
dfgenoresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResultsByGenotype.csv')
dfstatsT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS.csv')
dfstatsanova.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS_Anova.csv')
dfstatssigsibs.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS_SigMutVSib.csv')
dfstatssighets.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS_SigMutVHet.csv')
dfstatssigwts.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS_SigMutVWT.csv')
dfstatslargediff.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/BaselineResults_STATS_LargeDiff.csv')

print("Plotting results....")
# All Blocks
plt.figure(figsize=(25, 50))
plt.subplot(6, 3, 1)
sns.boxplot(x='block', y='#bouts', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Number of bouts")

plt.subplot(6, 3, 2)
sns.boxplot(x='block', y='totdist', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Total distance moved")

plt.subplot(6, 3, 3)
sns.boxplot(x='block', y='totdisp', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Total displacement moved")

plt.subplot(6, 3, 4)
sns.boxplot(x='block', y='tottimemvmt', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Total time moved")

plt.subplot(6, 3, 5)
sns.boxplot(x='block', y='totavgspeed', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Total average speed")

plt.subplot(6, 3, 6)
sns.boxplot(x='block', y='avgfromcenter', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Average distance from center")

plt.subplot(6, 3, 7)
sns.boxplot(x='block', y='medfromcenter', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Median distance from center")

plt.subplot(6, 3, 8)
sns.boxplot(x='block', y='fractionouter', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Fraction of time in outer rim")

plt.subplot(6, 3, 9)
sns.boxplot(x='block', y='avgdistperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Average distance per bout")

plt.subplot(6, 3, 10)
sns.boxplot(x='block', y='meddistperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Median distance per bout")

plt.subplot(6, 3, 11)
sns.boxplot(x='block', y='avgdispperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Average displacement per bout")

plt.subplot(6, 3, 12)
sns.boxplot(x='block', y='meddispperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Median displacement per bout")

plt.subplot(6, 3, 13)
sns.boxplot(x='block', y='avgtimeperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Average time per bout")

plt.subplot(6, 3, 14)
sns.boxplot(x='block', y='medtimeperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Median time per bout")

plt.subplot(6, 3, 15)
sns.boxplot(x='block', y='avgspeedperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Average speed per bout")

plt.subplot(6, 3, 16)
sns.boxplot(x='block', y='medspeedperbout', data=dfresultT, hue='genotype', hue_order=['mut', 'het', 'wt'],
            palette="Reds_r").set_title("Median speed per bout")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_all.pdf', bbox_inches='tight')

plt.clf()

# Heatmaps
plt.figure(figsize=(25, 10))
plt.subplot(6, 3, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 4)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 4)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 5)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 5)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 6)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 6)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 7)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 8)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 8)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 9)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 9)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 10)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 10)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 11)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 11)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 12)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 12)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 13)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 13)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 14)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 14)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 15)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 15)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 16)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 16)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 17)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 17)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 18)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 18)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_all_Heatmaps.pdf', bbox_inches='tight')

plt.clf()

# Heatmaps for mut-sib
plt.figure(figsize=(25, 10))
plt.subplot(6, 3, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 4)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 4)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 5)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 5)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 6)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 6)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 7)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 8)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 8)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 9)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 9)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 10)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 10)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 11)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 11)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 12)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 12)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 13)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 13)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 14)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 14)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 15)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 15)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 16)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 16)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 17)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 17)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(6, 3, 18)
grid = dfgenoresultT.loc[(dfgenoresultT['block'] == 18)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='block+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_all_Heatmaps_MutvSib.pdf', bbox_inches='tight')

plt.clf()

# correlation heatmap for every value for each fish
del dfresultT['genotype']
plt.figure(figsize=(25, 10))
sns.heatmap(dfresultT.astype(float).corr(method='pearson'), vmin=-1, vmax=1, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_BaselineCorrelation matrix.pdf', bbox_inches='tight')


