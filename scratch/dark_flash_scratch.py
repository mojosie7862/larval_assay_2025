import numpy as np
import cv2
# from TrackedObject import TrackedObject as TO
import matplotlib.pyplot as plt
import os
import re  # regular expressions library
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import remove_small_objects
import scipy.misc
import scipy.interpolate
import scipy.io
import pims
from tqdm import tqdm
import pandas as pd
import math

import scipy.stats as stats
import seaborn as sns
from matplotlib.pyplot import figure
from sklearn import metrics

# import cv2

# image1 = cv2.imread('C:/Users/phili_000/Documents/PENN/Granato Lab/Programming/test images+videos/background.tif', 0)
# image2 = cv2.imread('F://Granato Lab/Test video/slice.tif', 0)
# bs = abs(image2 - image1)

# print(image1)


# set directory containing image files
track_data_dir = 'E:/jm_laCie_larval_behavior_tracked_data/'
inDir = "C:/Users/millard/larval_behavior_data/"

# names of videos to track
Name = '241210-nrxn1a_beta_db_hetinx-DF_.avi'

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

# set the directory in which to save the files

SaveTrackingIm = True;  # or False
destgene = inDir + 'Results/' + Gene + '/' + Name + '/'

# Number of fish to track
nFish = 100

# Number of rows and columns of wells
nRow = 10
nCol = 10

# Number of pixels in x and y direction
nPixel = 1024

# frames per tap event
nFrameseventtap = 500

# number of tap events
nTapevents = 42

# number of total blocks
nBlocks = 3

nBlockshab = 3

# frame rate in Hz
framerate = 500

# time between frames in milliseconds
framelength = 1 / framerate * 1000

# frame of tapstimulus; stimulus is after 30ms
tapstimframe = 0

# designate frame where angle about threshold would be deemed too early and removed
tooearly = 15

# absolute total curvature in radians before considered a candidate for slc or llc, or for react
obendthreshangle = 1.75
reactthreshangle = 0.5

# this is to say if there is no folder, can make one
os.makedirs(os.path.dirname(destgene), exist_ok=True)

data = np.load(track_data_dir + Name + '/AnalyzedData.npz')

latencytaps = data['latencytaps']
latencytapsms = data['latencytapsms']
slctaps = data['slctaps']
medfiltcentY = data['medfiltcentY']
medfiltcentX = data['medfiltcentX']
disp = data['disp']
filtdisp = data['filtdisp']
llctaps = data['llctaps']
reacttaps = data['reacttaps']
nomvmttaps = data['nomvmttaps']
tapbendmax = data['tapbendmax']
tapdeltaorient = data['tapdeltaorient']
tapdisplacement = data['tapdisplacement']
tapdistance = data['tapdistance']
movementyesno = data['movementyesno']
movementyesnofill = data['movementyesnofill']
slcnan = data['slcnan']
slctot = data['slctot']
latnan = data['latnan']
lattot = data['lattot']
tapbendmaxnan = data['tapbendmaxnan']
tapbendmaxtot = data['tapbendmaxtot']
angdispmaxnan = data['angdispmaxnan']
angdispmaxtot = data['angdispmaxtot']
avgangvelmax = data['avgangvelmax']
medangvelmax = data['medangvelmax']
tapdeltaorientnan = data['tapdeltaorientnan']
tapdeltaorienttot = data['tapdeltaorienttot']
tapdisplacementnan = data['tapdisplacementnan']
tapdisplacementtot = data['tapdisplacementtot']
tapdistancenan = data['tapdistancenan']
tapdistancetot = data['tapdistancetot']
right = data['right']
towards = data['towards']
towardshalf = data['towardshalf']
avglat = data['avglat']
medlat = data['medlat']
slc = data['slc']
llc = data['llc']
react = data['react']
nomvmt = data['nomvmt']
avgbend = data['avgbend']
medbend = data['medbend']
avgorient = data['avgorient']
medorient = data['medorient']
avgdisp = data['avgdisp']
meddisp = data['meddisp']
avgdist = data['avgdist']
meddist = data['meddist']
avgduration = data['avgduration']
medduration = data['medduration']

print("Setting up final dictionaries....")
resultdict = {}

# setup dictionary with total results for all fish into dictionary/dataframe
for number in range(nFish):
    geno = np.nan
    if (number + 1) in muts:
        geno = "mut"
    if (number + 1) in hets:
        geno = "het"
    if (number + 1) in wts:
        geno = "wt"
    for i in range(nBlocks):

        if i == 0:
            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'event': (i + 1),
                'totlatev': (lattot[number, i] - latnan[number, i]),
                'avglat': avglat[number, i],
                'medianlat': medlat[number, i],
                'totslcev': (slctot[number, i] - slcnan[number, i]),
                '%slc': slc[number, i],
                '%react': react[number, i],
                '%nomovmt': nomvmt[number, i],
                '%rightturn': right[number, i],
                'directionbias': abs(2 * right[number, i] - 1),
                '%towards': towards[number, i],
                '%towardshalf': towardshalf[number, i],
                'totbendev': tapbendmaxtot[number, i] - tapbendmaxnan[number, i],
                'avgbend': avgbend[number, i],
                'medianbend': medbend[number, i],
                'totangvelev': angdispmaxtot[number, i] - angdispmaxnan[number, i],
                'avgangvelmax': avgangvelmax[number, i],
                'medianangvelmax': medangvelmax[number, i],
                'totorientev': tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i],
                'avgorient': avgorient[number, i],
                'medianorient': medorient[number, i],
                'totdispev': tapdisplacementtot[number, i] - tapdisplacementnan[number, i],
                'avgdisp': avgdisp[number, i],
                'mediandisp': meddisp[number, i],
                'totdistev': tapdistancetot[number, i] - tapdistancenan[number, i],
                'avgdist': avgdist[number, i],
                'mediandist': meddist[number, i],
                'avgduration': avgduration[number, i],
                'medianduration': medduration[number, i]
            }

        if i > 0:
            habavglat = np.nan
            habmedlat = np.nan
            habavgbend = np.nan
            habmedbend = np.nan
            habavgangvel = np.nan
            habmedangvel = np.nan
            habavgorient = np.nan
            habmedorient = np.nan
            habavgdisp = np.nan
            habmeddisp = np.nan
            habavgdist = np.nan
            habmeddist = np.nan
            habavgduration = np.nan
            habmedduration = np.nan
            habslc = np.nan

            if (lattot[number, i] - latnan[number, i]) >= 5 and (lattot[number, 0] - latnan[number, 0]) >= 5:
                habavglat = (avglat[number, 0] - avglat[number, i]) / avglat[number, 0]
                habmedlat = (medlat[number, 0] - medlat[number, i]) / medlat[number, 0]
            if (tapbendmaxtot[number, i] - tapbendmaxnan[number, i]) >= 5 and (
                    tapbendmaxtot[number, 0] - tapbendmaxnan[number, 0]) >= 5:
                habavgbend = (avgbend[number, 0] - avgbend[number, i]) / avgbend[number, 0]
                habmedbend = (medbend[number, 0] - medbend[number, i]) / medbend[number, 0]
            if (angdispmaxtot[number, i] - angdispmaxnan[number, i]) >= 5 and (
                    angdispmaxtot[number, 0] - angdispmaxnan[number, 0]) >= 5:
                habavgangvel = (avgangvelmax[number, 0] - avgangvelmax[number, i]) / avgangvelmax[number, 0]
                habmedangvel = (medangvelmax[number, 0] - medangvelmax[number, i]) / medangvelmax[number, 0]
            if (tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i]) >= 5 and (
                    tapdeltaorienttot[number, 0] - tapdeltaorientnan[number, 0]) >= 5:
                habavgorient = (avgorient[number, 0] - avgorient[number, i]) / avgorient[number, 0]
                habmedorient = (medorient[number, 0] - medorient[number, i]) / medorient[number, 0]
            if (tapdisplacementtot[number, i] - tapdisplacementnan[number, i]) >= 5 and (
                    tapdisplacementtot[number, 0] - tapdisplacementnan[number, 0]) >= 5:
                habavgdisp = (avgdisp[number, 0] - avgdisp[number, i]) / avgdisp[number, 0]
                habmeddisp = (meddisp[number, 0] - meddisp[number, i]) / meddisp[number, 0]
            if (tapdistancetot[number, i] - tapdistancenan[number, i]) >= 5 and (
                    tapdistancetot[number, 0] - tapdistancenan[number, 0]) >= 5:
                habavgdist = (avgdist[number, 0] - avgdist[number, i]) / avgdist[number, 0]
                habmeddist = (meddist[number, 0] - meddist[number, i]) / meddist[number, 0]
                habavgduration = (avgduration[number, 0] - avgduration[number, i]) / avgduration[number, 0]
                habmedduration = (medduration[number, 0] - medduration[number, i]) / medduration[number, 0]

            # only calculate if SLC>0
            if slc[number, 0] > 0:
                if (slctot[number, i] - slcnan[number, i]) >= 5 and (slctot[number, 0] - slcnan[number, 0]) >= 5:
                    habslc = (slc[number, 0] - slc[number, i]) / slc[number, 0]
            else:
                habslc = np.nan

            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'event': (i + 1),
                'totlatev': (lattot[number, i] - latnan[number, i]),
                'avglat': avglat[number, i],
                'medianlat': medlat[number, i],
                'totslcev': (slctot[number, i] - slcnan[number, i]),
                '%slc': slc[number, i],
                '%react': react[number, i],
                '%nomovmt': nomvmt[number, i],
                '%rightturn': right[number, i],
                'directionbias': abs(2 * right[number, i] - 1),
                '%towards': towards[number, i],
                '%towardshalf': towardshalf[number, i],
                'totbendev': tapbendmaxtot[number, i] - tapbendmaxnan[number, i],
                'avgbend': avgbend[number, i],
                'medianbend': medbend[number, i],
                'totangvelev': angdispmaxtot[number, i] - angdispmaxnan[number, i],
                'avgangvelmax': avgangvelmax[number, i],
                'medianangvelmax': medangvelmax[number, i],
                'totorientev': tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i],
                'avgorient': avgorient[number, i],
                'medianorient': medorient[number, i],
                'totdispev': tapdisplacementtot[number, i] - tapdisplacementnan[number, i],
                'avgdisp': avgdisp[number, i],
                'mediandisp': meddisp[number, i],
                'totdistev': tapdistancetot[number, i] - tapdistancenan[number, i],
                'avgdist': avgdist[number, i],
                'mediandist': meddist[number, i],
                'avgduration': avgduration[number, i],
                'medianduration': medduration[number, i],
                'habslc': habslc,
                'habavglat': habavglat,
                'habmedlat': habmedlat,
                'habavgbend': habavgbend,
                'habmedbend': habmedbend,
                'habavgangvel': habavgangvel,
                'habmedangvel': habmedangvel,
                'habavgorient': habavgorient,
                'habmedorient': habmedorient,
                'habavgdisp': habavgdisp,
                'habmeddisp': habmeddisp,
                'habavgdist': habavgdist,
                'habmeddist': habmeddist,
                'habavgduration': habavgduration,
                'habmedduration': habmedduration
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
    for i in range(nBlocks):
        if i == 0:
            resultdictindiv[str(number + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-totlatev': (lattot[number, i] - latnan[number, i]),
                str(i + 1) + '-avglat': avglat[number, i],
                str(i + 1) + '-medianlat': medlat[number, i],
                str(i + 1) + '-totslcev': (slctot[number, i] - slcnan[number, i]),
                str(i + 1) + '-%slc': slc[number, i],
                str(i + 1) + '-%llc': llc[number, i],
                str(i + 1) + '-slc/llc ratio': (slc[number, i] - llc[number, i]) / (slc[number, i] + llc[number, i]),
                str(i + 1) + '-%react': react[number, i],
                str(i + 1) + '-%nomovmt': nomvmt[number, i],
                str(i + 1) + '-%rightturn': right[number, i],
                str(i + 1) + '-directionbias': abs(2 * right[number, i] - 1),
                str(i + 1) + '-%towards': towards[number, i],
                str(i + 1) + '-%towardshalf': towardshalf[number, i],
                str(i + 1) + '-totbendev': tapbendmaxtot[number, i] - tapbendmaxnan[number, i],
                str(i + 1) + '-avgbend': avgbend[number, i],
                str(i + 1) + '-medianbend': medbend[number, i],
                str(i + 1) + '-totangvelev': angdispmaxtot[number, i] - angdispmaxnan[number, i],
                str(i + 1) + '-avgangvelmax': avgangvelmax[number, i],
                str(i + 1) + '-medianangvelmax': medangvelmax[number, i],
                str(i + 1) + '-totorientev': tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i],
                str(i + 1) + '-avgorient': avgorient[number, i],
                str(i + 1) + '-medianorient': medorient[number, i],
                str(i + 1) + '-totdispev': tapdisplacementtot[number, i] - tapdisplacementnan[number, i],
                str(i + 1) + '-avgdisp': avgdisp[number, i],
                str(i + 1) + '-mediandisp': meddisp[number, i],
                str(i + 1) + '-totdistev': tapdistancetot[number, i] - tapdistancenan[number, i],
                str(i + 1) + '-avgdist': avgdist[number, i],
                str(i + 1) + '-mediandist': meddist[number, i],
                str(i + 1) + '-avgduration': avgduration[number, i],
                str(i + 1) + '-medianduration': medduration[number, i]
            }

        # calculate hab if correct blocks
        if i > 0:
            habavglat = np.nan
            habmedlat = np.nan
            habavgbend = np.nan
            habmedbend = np.nan
            habavgangvel = np.nan
            habmedangvel = np.nan
            habavgorient = np.nan
            habmedorient = np.nan
            habavgdisp = np.nan
            habmeddisp = np.nan
            habavgdist = np.nan
            habmeddist = np.nan
            habavgduration = np.nan
            habmedduration = np.nan
            habslc = np.nan

            if (lattot[number, i] - latnan[number, i]) >= 5 and (lattot[number, 0] - latnan[number, 0]) >= 5:
                habavglat = (avglat[number, 0] - avglat[number, i]) / avglat[number, 0]
                habmedlat = (medlat[number, 0] - medlat[number, i]) / medlat[number, 0]
            if (tapbendmaxtot[number, i] - tapbendmaxnan[number, i]) >= 5 and (
                    tapbendmaxtot[number, 0] - tapbendmaxnan[number, 0]) >= 5:
                habavgbend = (avgbend[number, 0] - avgbend[number, i]) / avgbend[number, 0]
                habmedbend = (medbend[number, 0] - medbend[number, i]) / medbend[number, 0]
            if (angdispmaxtot[number, i] - angdispmaxnan[number, i]) >= 5 and (
                    angdispmaxtot[number, 0] - angdispmaxnan[number, 0]) >= 5:
                habavgangvel = (avgangvelmax[number, 0] - avgangvelmax[number, i]) / avgangvelmax[number, 0]
                habmedangvel = (medangvelmax[number, 0] - medangvelmax[number, i]) / medangvelmax[number, 0]
            if (tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i]) >= 5 and (
                    tapdeltaorienttot[number, 0] - tapdeltaorientnan[number, 0]) >= 5:
                habavgorient = (avgorient[number, 0] - avgorient[number, i]) / avgorient[number, 0]
                habmedorient = (medorient[number, 0] - medorient[number, i]) / medorient[number, 0]
            if (tapdisplacementtot[number, i] - tapdisplacementnan[number, i]) >= 5 and (
                    tapdisplacementtot[number, 0] - tapdisplacementnan[number, 0]) >= 5:
                habavgdisp = (avgdisp[number, 0] - avgdisp[number, i]) / avgdisp[number, 0]
                habmeddisp = (meddisp[number, 0] - meddisp[number, i]) / meddisp[number, 0]
            if (tapdistancetot[number, i] - tapdistancenan[number, i]) >= 5 and (
                    tapdistancetot[number, 0] - tapdistancenan[number, 0]) >= 5:
                habavgdist = (avgdist[number, 0] - avgdist[number, i]) / avgdist[number, 0]
                habmeddist = (meddist[number, 0] - meddist[number, i]) / meddist[number, 0]
                habavgduration = (avgduration[number, 0] - avgduration[number, i]) / avgduration[number, 0]
                habmedduration = (medduration[number, 0] - medduration[number, i]) / medduration[number, 0]

            # only calculate if SLC>0
            if slc[number, 0] > 0:
                if (slctot[number, i] - slcnan[number, i]) >= 5 and (slctot[number, 0] - slcnan[number, 0]) >= 5:
                    habslc = (slc[number, 0] - slc[number, i]) / slc[number, 0]
            else:
                habslc = np.nan

            resultdictindiv[str(number + 1)].update({
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-totlatev': (lattot[number, i] - latnan[number, i]),
                str(i + 1) + '-avglat': avglat[number, i],
                str(i + 1) + '-medianlat': medlat[number, i],
                str(i + 1) + '-totslcev': (slctot[number, i] - slcnan[number, i]),
                str(i + 1) + '-%slc': slc[number, i],
                str(i + 1) + '-%llc': llc[number, i],
                str(i + 1) + '-slc/llc ratio': (slc[number, i] - llc[number, i]) / (slc[number, i] + llc[number, i]),
                str(i + 1) + '-%react': react[number, i],
                str(i + 1) + '-%nomovmt': nomvmt[number, i],
                str(i + 1) + '-%rightturn': right[number, i],
                str(i + 1) + '-directionbias': abs(2 * right[number, i] - 1),
                str(i + 1) + '-%towards': towards[number, i],
                str(i + 1) + '-%towardshalf': towardshalf[number, i],
                str(i + 1) + '-totbendev': tapbendmaxtot[number, i] - tapbendmaxnan[number, i],
                str(i + 1) + '-avgbend': avgbend[number, i],
                str(i + 1) + '-medianbend': medbend[number, i],
                str(i + 1) + '-totangvelev': angdispmaxtot[number, i] - angdispmaxnan[number, i],
                str(i + 1) + '-avgangvelmax': avgangvelmax[number, i],
                str(i + 1) + '-medianangvelmax': medangvelmax[number, i],
                str(i + 1) + '-totorientev': tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i],
                str(i + 1) + '-avgorient': avgorient[number, i],
                str(i + 1) + '-medianorient': medorient[number, i],
                str(i + 1) + '-totdispev': tapdisplacementtot[number, i] - tapdisplacementnan[number, i],
                str(i + 1) + '-avgdisp': avgdisp[number, i],
                str(i + 1) + '-mediandisp': meddisp[number, i],
                str(i + 1) + '-totdistev': tapdistancetot[number, i] - tapdistancenan[number, i],
                str(i + 1) + '-avgdist': avgdist[number, i],
                str(i + 1) + '-mediandist': meddist[number, i],
                str(i + 1) + '-avgduration': avgduration[number, i],
                str(i + 1) + '-medianduration': medduration[number, i],
                str(i + 1) + '-habslc': habslc,
                str(i + 1) + '-habavglat': habavglat,
                str(i + 1) + '-habmedlat': habmedlat,
                str(i + 1) + '-habavgbend': habavgbend,
                str(i + 1) + '-habmedbend': habmedbend,
                str(i + 1) + '-habavgangvel': habavgangvel,
                str(i + 1) + '-habmedangvel': habmedangvel,
                str(i + 1) + '-habavgorient': habavgorient,
                str(i + 1) + '-habmedorient': habmedorient,
                str(i + 1) + '-habavgdisp': habavgdisp,
                str(i + 1) + '-habmeddisp': habmeddisp,
                str(i + 1) + '-habavgdist': habavgdist,
                str(i + 1) + '-habmeddist': habmeddist,
                str(i + 1) + '-habavgduration': habavgduration,
                str(i + 1) + '-habmedduration': habmedduration
            })

dfresultindiv = pd.DataFrame(data=resultdictindiv)
dfresultindivT = dfresultindiv.T

variables = ['avglat', 'medianlat', '%slc', '%react', '%nomovmt', 'avgbend', 'medianbend', 'avgorient', 'medianorient',
             '%rightturn', 'directionbias', '%towards', '%towardshalf',
             'avgdisp', 'mediandisp', 'avgdist', 'mediandist', 'avgduration', 'medianduration', 'avgangvelmax',
             'medianangvelmax',
             'habslc', 'habavglat', 'habmedlat', 'habavgbend', 'habmedbend', 'habavgangvel', 'habmedangvel',
             'habavgorient', 'habmedorient', 'habavgdisp', 'habmeddisp', 'habavgdist', 'habmeddist', 'habavgduration',
             'habmedduration']

genoresultdict = {}
statsdict = {}

# setup dictionary with results averaged over genotype into dictionary/dataframe, removing those with <5 tracked objects in block
for number in range(nBlocks):
    for i, element in enumerate(variables):
        if str(element) == '%slc' or str(element) == '%react' or str(element) == '%nomovmt':
            events = 'totslcev'

        if str(element) == 'avglat' or str(element) == 'medianlat' or str(element) == 'habslc' or str(
                element) == 'habavglat' or str(element) == 'habmedlat' or str(element) == '%rightturn' or str(
                element) == 'directionbias' or str(element) == '%towards' or str(element) == '%towardshalf':
            events = 'totlatev'

        if str(element) == 'avgbend' or str(element) == 'medianbend' or str(element) == 'habavgbend' or str(
                element) == 'habmedbend':
            events = 'totbendev'

        if str(element) == 'avgangvelmax' or str(element) == 'medianangvelmax' or str(element) == 'habavgangvel' or str(
                element) == 'habmedangvel':
            events = 'totangvelev'

        if str(element) == 'avgorient' or str(element) == 'medianorient' or str(element) == 'habavgorient' or str(
                element) == 'habmedorient':
            events = 'totorientev'

        if str(element) == 'avgdisp' or str(element) == 'mediandisp' or str(element) == 'habavgdisp' or str(
                element) == 'habmeddisp':
            events = 'totdispev'

        if str(element) == 'avgdist' or str(element) == 'mediandist' or str(element) == 'habavgdist' or str(
                element) == 'habmeddist' or str(element) == 'avgduration' or str(element) == 'medianduration' or str(
                element) == 'habavgduration' or str(element) == 'habmedduration':
            events = 'totdistev'

        if float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)) == 0:
            genoresultdict[str(number + 1) + "-" + str(element) + "-" + "mut"] = {
                'event': (number + 1),
                'value': str(element),
                'genotype': "mut",
                'count': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].count()),
                'avg': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                'median': float(dfresultT.loc[
                                    (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                'stdev': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                '(avg-wt)/wtstd': np.nan,
                '(avg-sib)/sibstd': np.nan
            }
            genoresultdict[str(number + 1) + "-" + str(element) + "-" + "het"] = {
                'event': (number + 1),
                'value': str(element),
                'genotype': "het",
                'count': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].count()),
                'avg': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                'median': float(dfresultT.loc[
                                    (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                'stdev': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                '(avg-wt)/wtstd': np.nan
            }
            genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
                'event': (number + 1),
                'value': str(element),
                'genotype': "sib",
                'count': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                            dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                    str(element)]].count()),
                'avg': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                            dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0)),
                'median': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                            dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].median(
                    axis=0)),
                'stdev': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                            dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].std(
                    axis=0))
            }

        else:
            if float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                    dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)) == 0:
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "mut"] = {
                    'event': (number + 1),
                    'value': str(element),
                    'genotype': "mut",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                    dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': np.nan,
                    '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) - float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (
                                                                                       dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0))) / float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (
                                                                                        dfresultT[str(events)] >= 5), [
                        str(element)]].std(axis=0))
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "het"] = {
                    'event': (number + 1),
                    'value': str(element),
                    'genotype': "het",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                    dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': np.nan
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
                    'event': (number + 1),
                    'value': str(element),
                    'genotype': "sib",
                    'count': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avg': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                    dfresultT['event'] == (number + 1)) & (
                                                    dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].std(axis=0))
                }

            else:
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "mut"] = {
                    'event': (number + 1),
                    'value': str(element),
                    'genotype': "mut",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                    dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].std(axis=0)),
                    '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) - float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (
                                                                                       dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0))) / float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (
                                                                                        dfresultT[str(events)] >= 5), [
                        str(element)]].std(axis=0))
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "het"] = {
                    'event': (number + 1),
                    'value': str(element),
                    'genotype': "het",
                    'count': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avg': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                    dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                    '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].std(axis=0))
                }
                genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
                    'event': (number + 1),
                    'value': str(element),
                    'genotype': "sib",
                    'count': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avg': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)),
                    'median': float(dfresultT.loc[
                                        ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                    dfresultT['event'] == (number + 1)) & (
                                                    dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
                    'stdev': float(dfresultT.loc[
                                       ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                   dfresultT['event'] == (number + 1)) & (
                                                   dfresultT[str(events)] >= 5), [str(element)]].std(axis=0))
                }
        genoresultdict[str(number + 1) + "-" + str(element) + "-" + "wt"] = {
            'event': (number + 1),
            'value': str(element),
            'genotype': "wt",
            'count': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                        dfresultT[str(events)] >= 5), [str(element)]].count()),
            'avg': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                        dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
            'median': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                        dfresultT[str(events)] >= 5), [str(element)]].median(axis=0)),
            'stdev': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                        dfresultT[str(events)] >= 5), [str(element)]].std(axis=0))
        }
        # setup dictionary with statistics for each event and property, removing those with <5 tracked objects in block
        anova = np.array(stats.f_oneway(dfresultT.loc[
                                            (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                        dfresultT[str(events)] >= 5), [str(element)]].dropna(),
                                        dfresultT.loc[
                                            (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                        dfresultT[str(events)] >= 5), [str(element)]].dropna(),
                                        dfresultT.loc[
                                            (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                        dfresultT[str(events)] >= 5), [str(element)]].dropna()))

        mut = np.array(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                    dfresultT[str(events)] >= 5), [str(element)]].dropna())
        het = np.array(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                    dfresultT[str(events)] >= 5), [str(element)]].dropna())
        wt = np.array(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                    dfresultT[str(events)] >= 5), [str(element)]].dropna())
        sib = np.array(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].dropna())

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

        if float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)) == 0:
            if float(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0)) == 0:
                statsdict[str("DF-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                  dfresultT[str(events)] >= 5), [str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                  dfresultT[str(events)] >= 5), [str(element)]].count()),
                    '#wt': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].count()),
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
                statsdict[str("DF-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                  dfresultT[str(events)] >= 5), [str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                  dfresultT[str(events)] >= 5), [str(element)]].count()),
                    '#wt': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avgmut/avgwt': np.nan,
                    'avgmut/avgsib': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) / float(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (
                                dfresultT['genotype'] == 'het')) & (dfresultT['event'] == (number + 1)) & (
                                                                                       dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)),
                    '(avg-wt)/wtstd': np.nan,
                    '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) - float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (
                                                                                       dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0))) / float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (
                                                                                        dfresultT[str(events)] >= 5), [
                        str(element)]].std(axis=0)),
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
            statsdict[str("DF-") + str(number + 1) + "-" + str(element)] = {
                '#mut': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].count()),
                '#het': float(dfresultT.loc[(dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].count()),
                '#wt': float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                            dfresultT[str(events)] >= 5), [str(element)]].count()),
                'avgmut/avgwt': float(dfresultT.loc[
                                          (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                      dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0)) / float(dfresultT.loc[
                                         (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                     dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                'avgmut/avgsib': float(dfresultT.loc[
                                           (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                       dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0)) / float(dfresultT.loc[
                                         ((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                                                     dfresultT['event'] == (number + 1)) & (
                                                     dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)),
                '(avg-wt)/wtstd': (float(dfresultT.loc[
                                             (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                         dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0)) - float(dfresultT.loc[
                                         (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                     dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0))) / float(dfresultT.loc[
                                          (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                      dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
                '(avg-sib)/sibstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                            dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0)) - float(dfresultT.loc[
                                         ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                     dfresultT['event'] == (number + 1)) & (
                                                     dfresultT[str(events)] >= 5), [str(element)]].mean(
                    axis=0))) / float(dfresultT.loc[
                                          ((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                                                      dfresultT['event'] == (number + 1)) & (
                                                      dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)),
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

dfgenoresultT['event+value'] = dfgenoresultT['event'].apply(str) + "-" + dfgenoresultT['value'].apply(str)

dfstats = pd.DataFrame(data=statsdict)
dfstatsT = dfstats.T

dfstatsanova = dfstatsT.loc[(dfstatsT['Sig ANOVA?'] == 'YES')]
dfstatssigsibs = dfstatsT.loc[(dfstatsT['Sig MvS?'] == 'YES')]
dfstatssighets = dfstatsT.loc[(dfstatsT['Sig ANOVA?'] == 'YES') & (dfstatsT['Sig MvH?'] == 'YES')]
dfstatssigwts = dfstatsT.loc[(dfstatsT['Sig ANOVA?'] == 'YES') & (dfstatsT['Sig MvW?'] == 'YES')]
# if avg for mut would be in top or bottom 10th percentile
dfstatslargediff = dfstatsT.loc[(abs(dfstatsT['(avg-wt)/wtstd']) >= 1.28) | (abs(dfstatsT['(avg-sib)/sibstd']) >= 1.28)]

print("Saving results in spreadsheet....")
dfresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResultsByFishbyEvent.csv')
dfresultindivT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResultsByIndividualFish.csv')
dfgenoresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResultsByGenotype.csv')
dfstatsT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResults_STATS.csv')
dfstatsanova.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResults_STATS_Anova.csv')
dfstatssigsibs.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResults_STATS_SigMutVSib.csv')
dfstatssighets.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResults_STATS_SigMutVHet.csv')
dfstatssigwts.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResults_STATS_SigMutVWT.csv')
dfstatslargediff.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/DFlashResults_STATS_LargeDiff.csv')

# create heatmap for mut and het comparios to wildtype


print("Plotting results....")
# Hab Blocks
plt.figure(figsize=(25, 50))

plt.subplot(11, 3, 1)
sns.boxplot(x='event', y='totslcev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Obend Events")

plt.subplot(11, 3, 2)
sns.boxplot(x='event', y='totlatev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Latency Events")

plt.subplot(11, 3, 4)
sns.boxplot(x='event', y='%slc', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - %Obend")

plt.subplot(11, 3, 5)
sns.boxplot(x='event', y='%react', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - %React")

plt.subplot(11, 3, 6)
sns.boxplot(x='event', y='%nomovmt', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - %No Movement")

plt.subplot(11, 3, 7)
sns.boxplot(x='event', y='%rightturn', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - %Right turn")

plt.subplot(11, 3, 8)
sns.boxplot(x='event', y='directionbias', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - R/L Bias")

plt.subplot(11, 3, 9)
sns.boxplot(x='event', y='%towards', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - %Towards stimulus")

plt.subplot(11, 3, 10)
sns.boxplot(x='event', y='%towardshalf', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - %Towards stimulus (midle half initial)")

plt.subplot(11, 3, 11)
sns.boxplot(x='event', y='avglat', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Average Latency")

plt.subplot(11, 3, 12)
sns.boxplot(x='event', y='medianlat', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Median Latency")

plt.subplot(11, 3, 13)
sns.boxplot(x='event', y='totbendev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Bend Events")

plt.subplot(11, 3, 14)
sns.boxplot(x='event', y='avgbend', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Average Max Bend")

plt.subplot(11, 3, 15)
sns.boxplot(x='event', y='medianbend', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Median Max Bend")

plt.subplot(11, 3, 16)
sns.boxplot(x='event', y='totangvelev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Ang Vel Max Events")

plt.subplot(11, 3, 17)
sns.boxplot(x='event', y='avgangvelmax',
            data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totangvelev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Average Max Ang Velocity")

plt.subplot(11, 3, 18)
sns.boxplot(x='event', y='medianangvelmax',
            data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totangvelev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Median Max Ang Velocity")

plt.subplot(11, 3, 19)
sns.boxplot(x='event', y='totorientev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Orient Events")

plt.subplot(11, 3, 20)
sns.boxplot(x='event', y='avgorient', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totorientev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Average Change in Orientation")

plt.subplot(11, 3, 21)
sns.boxplot(x='event', y='medianorient',
            data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totorientev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Median Change in Orientation")

plt.subplot(11, 3, 22)
sns.boxplot(x='event', y='totdispev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Displacement Events")

plt.subplot(11, 3, 23)
sns.boxplot(x='event', y='avgdisp', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Average Displacement")

plt.subplot(11, 3, 24)
sns.boxplot(x='event', y='mediandisp', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Median Displacement")

plt.subplot(11, 3, 25)
sns.boxplot(x='event', y='totdistev', data=dfresultT.loc[(dfresultT['event'] >= 1)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Total Displacement Events")

plt.subplot(11, 3, 26)
sns.boxplot(x='event', y='avgdist', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Average Distance")

plt.subplot(11, 3, 27)
sns.boxplot(x='event', y='mediandist', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Median Distance")

plt.subplot(11, 3, 28)
sns.boxplot(x='event', y='avgduration', data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "DF Habituation - Average Duration")

plt.subplot(11, 3, 29)
sns.boxplot(x='event', y='medianduration',
            data=dfresultT.loc[(dfresultT['event'] >= 1) & (dfresultT['totdistev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("DF Habituation - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_DF.pdf', bbox_inches='tight')

plt.clf()

# Hab Values
plt.figure(figsize=(25, 25))

plt.subplot(5, 3, 1)
sns.boxplot(x='event', y='habslc', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Obend")

plt.subplot(5, 3, 2)
sns.boxplot(x='event', y='habavglat', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Latency")

plt.subplot(5, 3, 3)
sns.boxplot(x='event', y='habmedlat', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Latency")

plt.subplot(5, 3, 4)
sns.boxplot(x='event', y='habavgbend', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Bend Max")

plt.subplot(5, 3, 5)
sns.boxplot(x='event', y='habmedbend', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Bend Max")

plt.subplot(5, 3, 6)
sns.boxplot(x='event', y='habavgangvel', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Ang vel Max")

plt.subplot(5, 3, 7)
sns.boxplot(x='event', y='habmedangvel', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Ang vel Max")

plt.subplot(5, 3, 8)
sns.boxplot(x='event', y='habavgorient', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Reorientation")

plt.subplot(5, 3, 9)
sns.boxplot(x='event', y='habmedorient', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Reorientation")

plt.subplot(5, 3, 10)
sns.boxplot(x='event', y='habavgdisp', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Displacement")

plt.subplot(5, 3, 11)
sns.boxplot(x='event', y='habmeddisp', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Displacement")

plt.subplot(5, 3, 12)
sns.boxplot(x='event', y='habavgdist', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Distance")

plt.subplot(5, 3, 13)
sns.boxplot(x='event', y='habmeddist', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Distance")

plt.subplot(5, 3, 14)
sns.boxplot(x='event', y='habavgduration', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Duration")

plt.subplot(5, 3, 15)
sns.boxplot(x='event', y='habmedduration', data=dfresultT.loc[(dfresultT['event'] >= 2)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_DF_Hab_values.pdf', bbox_inches='tight')

plt.clf()

# Hab heatmaps
plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_DF_Heatmaps.pdf', bbox_inches='tight')

plt.clf()

# Hab heatmaps Mut v Sib
plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_DF_Heatmaps_MutvSib.pdf', bbox_inches='tight')

plt.clf()

# correlation heatmap for every value for each fish
del dfresultT['genotype']
plt.figure(figsize=(25, 10))
sns.heatmap(dfresultT.astype(float).corr(method='pearson'), vmin=-1, vmax=1, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_DFCorrelation matrix.pdf', bbox_inches='tight')
