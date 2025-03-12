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
inDir = "C:/Users/millard/larval_behavior_data/"
track_data_dir = 'E:/jm_laCie_larval_behavior_tracked_data/'

# names of videos to track
Name = '241210-nrxn1a_beta_db_hetinx-Tap_.avi'

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

# 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50
# 51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100

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

# frames per tap event
nFrameseventtap = 100

# number of tap events
nTapevents = 100

# number of total blocks
nBlocks = 10

# frame rate in Hz
framerate = 500

# time between frames in milliseconds
framelength = 1 / framerate * 1000

# frame of tapstimulus; stimulus is after 30ms
tapstimframe = 15

# absolute total curvature in radians before considered a candidate for slc or llc, or for react
slcthreshangle = 0.8
reactthreshangle = 0.5

# time threshold for slc after stimulus, in ms
slcthreshms = 20

# number of event blocks for sens
nBlockssens = 2

# number of ppi types
nBlocksppi = 5

# number of event blocks for sens
nBlockshab = 3

# this is to say if there is no folder, can make one
os.makedirs(os.path.dirname(desttracked), exist_ok=True)

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

# setup dictionary with total results for all fish based on event
for number in range(nFish):
    geno = np.nan
    if (number + 1) in muts:
        geno = "mut"
    if (number + 1) in hets:
        geno = "het"
    if (number + 1) in wts:
        geno = "wt"
    for i in range(nBlocks):

        if i < nBlockssens:
            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'event': (i + 1),
                'totlatev': (lattot[number, i] - latnan[number, i]),
                'avglat': avglat[number, i],
                'medianlat': medlat[number, i],
                'totslcev': (slctot[number, i] - slcnan[number, i]),
                '%slc': slc[number, i],
                '%llc': llc[number, i],
                'slc/llc ratio': (slc[number, i] - llc[number, i]) / (slc[number, i] + llc[number, i]),
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

        if i == ((nBlockssens + nBlocksppi) - 1):
            if (slctot[number, 0] - slcnan[number, 0]) >= 5 and (slctot[number, 1] - slcnan[number, 1]) >= 5 and (
                    slctot[number, 6] - slcnan[number, 6]) >= 5:
                x = [0.6, 2, 10]
                y = [slc[number, 0], slc[number, 1], slc[number, 6]]
                auc = metrics.auc(x, y)
            else:
                auc = np.nan

            resultdict[str(number + 1) + "-" + str(i + 1)] = {
                'fish#': (number + 1),
                'genotype': geno,
                'event': (i + 1),
                'totlatev': (lattot[number, i] - latnan[number, i]),
                'avglat': avglat[number, i],
                'medianlat': medlat[number, i],
                'totslcev': (slctot[number, i] - slcnan[number, i]),
                '%slc': slc[number, i],
                'AUC': auc,
                '%llc': llc[number, i],
                'slc/llc ratio': (slc[number, i] - llc[number, i]) / (slc[number, i] + llc[number, i]),
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

        # calculate ppi if correct blocks
        if i >= nBlockssens and i < (nBlockssens + nBlocksppi - 1):
            ppiavglat = np.nan
            ppimedlat = np.nan
            ppiavgbend = np.nan
            ppimedbend = np.nan
            ppiavgangvel = np.nan
            ppimedangvel = np.nan
            ppiavgorient = np.nan
            ppimedorient = np.nan
            ppiavgdisp = np.nan
            ppimeddisp = np.nan
            ppiavgdist = np.nan
            ppimeddist = np.nan
            ppiavgduration = np.nan
            ppimedduration = np.nan
            ppislc = np.nan

            if (lattot[number, i] - latnan[number, i]) >= 5 and (
                    lattot[number, (nBlockssens + nBlocksppi - 1)] - latnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavglat = (avglat[number, (nBlockssens + nBlocksppi - 1)] - avglat[number, i]) / avglat[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimedlat = (medlat[number, (nBlockssens + nBlocksppi - 1)] - medlat[number, i]) / medlat[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapbendmaxtot[number, i] - tapbendmaxnan[number, i]) >= 5 and (
                    tapbendmaxtot[number, (nBlockssens + nBlocksppi - 1)] - tapbendmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgbend = (avgbend[number, (nBlockssens + nBlocksppi - 1)] - avgbend[number, i]) / avgbend[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimedbend = (medbend[number, (nBlockssens + nBlocksppi - 1)] - medbend[number, i]) / medbend[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (angdispmaxtot[number, i] - angdispmaxnan[number, i]) >= 5 and (
                    angdispmaxtot[number, (nBlockssens + nBlocksppi - 1)] - angdispmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgangvel = (avgangvelmax[number, (nBlockssens + nBlocksppi - 1)] - avgangvelmax[number, i]) / \
                               avgangvelmax[number, (nBlockssens + nBlocksppi - 1)]
                ppimedangvel = (medangvelmax[number, (nBlockssens + nBlocksppi - 1)] - medangvelmax[number, i]) / \
                               medangvelmax[number, (nBlockssens + nBlocksppi - 1)]
            if (tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i]) >= 5 and (
                    tapdeltaorienttot[number, (nBlockssens + nBlocksppi - 1)] - tapdeltaorientnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgorient = (avgorient[number, (nBlockssens + nBlocksppi - 1)] - avgorient[number, i]) / avgorient[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimedorient = (medorient[number, (nBlockssens + nBlocksppi - 1)] - medorient[number, i]) / medorient[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdisplacementtot[number, i] - tapdisplacementnan[number, i]) >= 5 and (
                    tapdisplacementtot[number, (nBlockssens + nBlocksppi - 1)] - tapdisplacementnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgdisp = (avgdisp[number, (nBlockssens + nBlocksppi - 1)] - avgdisp[number, i]) / avgdisp[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimeddisp = (meddisp[number, (nBlockssens + nBlocksppi - 1)] - meddisp[number, i]) / meddisp[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdistancetot[number, i] - tapdistancenan[number, i]) >= 5 and (
                    tapdistancetot[number, (nBlockssens + nBlocksppi - 1)] - tapdistancenan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgdist = (avgdist[number, (nBlockssens + nBlocksppi - 1)] - avgdist[number, i]) / avgdist[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimeddist = (meddist[number, (nBlockssens + nBlocksppi - 1)] - meddist[number, i]) / meddist[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppiavgduration = (avgduration[number, (nBlockssens + nBlocksppi - 1)] - avgduration[number, i]) / \
                                 avgduration[number, (nBlockssens + nBlocksppi - 1)]
                ppimedduration = (medduration[number, (nBlockssens + nBlocksppi - 1)] - medduration[number, i]) / \
                                 medduration[number, (nBlockssens + nBlocksppi - 1)]

            # only calculate if SLC>0
            if slc[number, (nBlockssens + nBlocksppi - 1)] > 0:
                if (slctot[number, i] - slcnan[number, i]) >= 5 and (
                        slctot[number, (nBlockssens + nBlocksppi - 1)] - slcnan[
                    number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                    ppislc = (slc[number, (nBlockssens + nBlocksppi - 1)] - slc[number, i]) / slc[
                        number, (nBlockssens + nBlocksppi - 1)]
            else:
                ppislc = np.nan

            resultdict[(str(number + 1) + "-" + str(i + 1))] = {
                'fish#': (number + 1),
                'genotype': geno,
                'event': (i + 1),
                'totlatev': (lattot[number, i] - latnan[number, i]),
                'avglat': avglat[number, i],
                'medianlat': medlat[number, i],
                'totslcev': (slctot[number, i] - slcnan[number, i]),
                '%slc': slc[number, i],
                '%llc': llc[number, i],
                'slc/llc ratio': (slc[number, i] - llc[number, i]) / (slc[number, i] + llc[number, i]),
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
                'ppislc': ppislc,
                'ppiavglat': ppiavglat,
                'ppimedlat': ppimedlat,
                'ppiavgbend': ppiavgbend,
                'ppimedbend': ppimedbend,
                'ppiavgangvel': ppiavgangvel,
                'ppimedangvel': ppimedangvel,
                'ppiavgorient': ppiavgorient,
                'ppimedorient': ppimedorient,
                'ppiavgdisp': ppiavgdisp,
                'ppimeddisp': ppimeddisp,
                'ppiavgdist': ppiavgdist,
                'ppimeddist': ppimeddist,
                'ppiavgduration': ppiavgduration,
                'ppimedduration': ppimedduration
            }

        # calculate hab if correct blocks
        if i >= (nBlockssens + nBlocksppi):
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

            if (lattot[number, i] - latnan[number, i]) >= 5 and (
                    lattot[number, (nBlockssens + nBlocksppi - 1)] - latnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavglat = (avglat[number, (nBlockssens + nBlocksppi - 1)] - avglat[number, i]) / avglat[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmedlat = (medlat[number, (nBlockssens + nBlocksppi - 1)] - medlat[number, i]) / medlat[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapbendmaxtot[number, i] - tapbendmaxnan[number, i]) >= 5 and (
                    tapbendmaxtot[number, (nBlockssens + nBlocksppi - 1)] - tapbendmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgbend = (avgbend[number, (nBlockssens + nBlocksppi - 1)] - avgbend[number, i]) / avgbend[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmedbend = (medbend[number, (nBlockssens + nBlocksppi - 1)] - medbend[number, i]) / medbend[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (angdispmaxtot[number, i] - angdispmaxnan[number, i]) >= 5 and (
                    angdispmaxtot[number, (nBlockssens + nBlocksppi - 1)] - angdispmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgangvel = (avgangvelmax[number, (nBlockssens + nBlocksppi - 1)] - avgangvelmax[number, i]) / \
                               avgangvelmax[number, (nBlockssens + nBlocksppi - 1)]
                habmedangvel = (medangvelmax[number, (nBlockssens + nBlocksppi - 1)] - medangvelmax[number, i]) / \
                               medangvelmax[number, (nBlockssens + nBlocksppi - 1)]
            if (tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i]) >= 5 and (
                    tapdeltaorienttot[number, (nBlockssens + nBlocksppi - 1)] - tapdeltaorientnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgorient = (avgorient[number, (nBlockssens + nBlocksppi - 1)] - avgorient[number, i]) / avgorient[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmedorient = (medorient[number, (nBlockssens + nBlocksppi - 1)] - medorient[number, i]) / medorient[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdisplacementtot[number, i] - tapdisplacementnan[number, i]) >= 5 and (
                    tapdisplacementtot[number, (nBlockssens + nBlocksppi - 1)] - tapdisplacementnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgdisp = (avgdisp[number, (nBlockssens + nBlocksppi - 1)] - avgdisp[number, i]) / avgdisp[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmeddisp = (meddisp[number, (nBlockssens + nBlocksppi - 1)] - meddisp[number, i]) / meddisp[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdistancetot[number, i] - tapdistancenan[number, i]) >= 5 and (
                    tapdistancetot[number, (nBlockssens + nBlocksppi - 1)] - tapdistancenan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgdist = (avgdist[number, (nBlockssens + nBlocksppi - 1)] - avgdist[number, i]) / avgdist[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmeddist = (meddist[number, (nBlockssens + nBlocksppi - 1)] - meddist[number, i]) / meddist[
                    number, (nBlockssens + nBlocksppi - 1)]
                habavgduration = (avgduration[number, (nBlockssens + nBlocksppi - 1)] - avgduration[number, i]) / \
                                 avgduration[number, (nBlockssens + nBlocksppi - 1)]
                habmedduration = (medduration[number, (nBlockssens + nBlocksppi - 1)] - medduration[number, i]) / \
                                 medduration[number, (nBlockssens + nBlocksppi - 1)]

            # only calculate if SLC>0
            if slc[number, (nBlockssens + nBlocksppi - 1)] > 0:
                if (slctot[number, i] - slcnan[number, i]) >= 5 and (
                        slctot[number, (nBlockssens + nBlocksppi - 1)] - slcnan[
                    number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                    habslc = (slc[number, (nBlockssens + nBlocksppi - 1)] - slc[number, i]) / slc[
                        number, (nBlockssens + nBlocksppi - 1)]
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
                '%llc': llc[number, i],
                'slc/llc ratio': (slc[number, i] - llc[number, i]) / (slc[number, i] + llc[number, i]),
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

        if i < nBlockssens:
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
                str(i + 1) + '-medianduration': medduration[number, i]
            })

        if i == ((nBlockssens + nBlocksppi) - 1):
            if (slctot[number, 0] - slcnan[number, 0]) >= 5 and (slctot[number, 1] - slcnan[number, 1]) >= 5 and (
                    slctot[number, 6] - slcnan[number, 6]) >= 5:
                x = [0.6, 2, 10]
                y = [slc[number, 0], slc[number, 1], slc[number, 6]]
                auc = metrics.auc(x, y)
            else:
                auc = np.nan

            resultdictindiv[str(number + 1)].update({
                'fish#': (number + 1),
                'genotype': geno,
                str(i + 1) + '-totlatev': (lattot[number, i] - latnan[number, i]),
                str(i + 1) + '-avglat': avglat[number, i],
                str(i + 1) + '-medianlat': medlat[number, i],
                str(i + 1) + '-totslcev': (slctot[number, i] - slcnan[number, i]),
                str(i + 1) + '-%slc': slc[number, i],
                str(i + 1) + '-AUC': auc,
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
            })

        # calculate ppi if correct blocks
        if i >= nBlockssens and i < (nBlockssens + nBlocksppi - 1):
            ppiavglat = np.nan
            ppimedlat = np.nan
            ppiavgbend = np.nan
            ppimedbend = np.nan
            ppiavgangvel = np.nan
            ppimedangvel = np.nan
            ppiavgorient = np.nan
            ppimedorient = np.nan
            ppiavgdisp = np.nan
            ppimeddisp = np.nan
            ppiavgdist = np.nan
            ppimeddist = np.nan
            ppiavgduration = np.nan
            ppimedduration = np.nan
            ppislc = np.nan

            if (lattot[number, i] - latnan[number, i]) >= 5 and (
                    lattot[number, (nBlockssens + nBlocksppi - 1)] - latnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavglat = (avglat[number, (nBlockssens + nBlocksppi - 1)] - avglat[number, i]) / avglat[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimedlat = (medlat[number, (nBlockssens + nBlocksppi - 1)] - medlat[number, i]) / medlat[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapbendmaxtot[number, i] - tapbendmaxnan[number, i]) >= 5 and (
                    tapbendmaxtot[number, (nBlockssens + nBlocksppi - 1)] - tapbendmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgbend = (avgbend[number, (nBlockssens + nBlocksppi - 1)] - avgbend[number, i]) / avgbend[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimedbend = (medbend[number, (nBlockssens + nBlocksppi - 1)] - medbend[number, i]) / medbend[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (angdispmaxtot[number, i] - angdispmaxnan[number, i]) >= 5 and (
                    angdispmaxtot[number, (nBlockssens + nBlocksppi - 1)] - angdispmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgangvel = (avgangvelmax[number, (nBlockssens + nBlocksppi - 1)] - avgangvelmax[number, i]) / \
                               avgangvelmax[number, (nBlockssens + nBlocksppi - 1)]
                ppimedangvel = (medangvelmax[number, (nBlockssens + nBlocksppi - 1)] - medangvelmax[number, i]) / \
                               medangvelmax[number, (nBlockssens + nBlocksppi - 1)]
            if (tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i]) >= 5 and (
                    tapdeltaorienttot[number, (nBlockssens + nBlocksppi - 1)] - tapdeltaorientnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgorient = (avgorient[number, (nBlockssens + nBlocksppi - 1)] - avgorient[number, i]) / avgorient[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimedorient = (medorient[number, (nBlockssens + nBlocksppi - 1)] - medorient[number, i]) / medorient[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdisplacementtot[number, i] - tapdisplacementnan[number, i]) >= 5 and (
                    tapdisplacementtot[number, (nBlockssens + nBlocksppi - 1)] - tapdisplacementnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgdisp = (avgdisp[number, (nBlockssens + nBlocksppi - 1)] - avgdisp[number, i]) / avgdisp[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimeddisp = (meddisp[number, (nBlockssens + nBlocksppi - 1)] - meddisp[number, i]) / meddisp[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdistancetot[number, i] - tapdistancenan[number, i]) >= 5 and (
                    tapdistancetot[number, (nBlockssens + nBlocksppi - 1)] - tapdistancenan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                ppiavgdist = (avgdist[number, (nBlockssens + nBlocksppi - 1)] - avgdist[number, i]) / avgdist[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppimeddist = (meddist[number, (nBlockssens + nBlocksppi - 1)] - meddist[number, i]) / meddist[
                    number, (nBlockssens + nBlocksppi - 1)]
                ppiavgduration = (avgduration[number, (nBlockssens + nBlocksppi - 1)] - avgduration[number, i]) / \
                                 avgduration[number, (nBlockssens + nBlocksppi - 1)]
                ppimedduration = (medduration[number, (nBlockssens + nBlocksppi - 1)] - medduration[number, i]) / \
                                 medduration[number, (nBlockssens + nBlocksppi - 1)]

            # only calculate if SLC>0
            if slc[number, (nBlockssens + nBlocksppi - 1)] > 0:
                if (slctot[number, i] - slcnan[number, i]) >= 5 and (
                        slctot[number, (nBlockssens + nBlocksppi - 1)] - slcnan[
                    number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                    ppislc = (slc[number, (nBlockssens + nBlocksppi - 1)] - slc[number, i]) / slc[
                        number, (nBlockssens + nBlocksppi - 1)]
            else:
                ppislc = np.nan

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
                str(i + 1) + '-ppislc': ppislc,
                str(i + 1) + '-ppiavglat': ppiavglat,
                str(i + 1) + '-ppimedlat': ppimedlat,
                str(i + 1) + '-ppiavgbend': ppiavgbend,
                str(i + 1) + '-ppimedbend': ppimedbend,
                str(i + 1) + '-ppiavgangvel': ppiavgangvel,
                str(i + 1) + '-ppimedangvel': ppimedangvel,
                str(i + 1) + '-ppiavgorient': ppiavgorient,
                str(i + 1) + '-ppimedorient': ppimedorient,
                str(i + 1) + '-ppiavgdisp': ppiavgdisp,
                str(i + 1) + '-ppimeddisp': ppimeddisp,
                str(i + 1) + '-ppiavgdist': ppiavgdist,
                str(i + 1) + '-ppimeddist': ppimeddist,
                str(i + 1) + '-ppiavgduration': ppiavgduration,
                str(i + 1) + '-ppimedduration': ppimedduration
            })

        # calculate hab if correct blocks
        if i >= (nBlockssens + nBlocksppi):
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

            if (lattot[number, i] - latnan[number, i]) >= 5 and (
                    lattot[number, (nBlockssens + nBlocksppi - 1)] - latnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavglat = (avglat[number, (nBlockssens + nBlocksppi - 1)] - avglat[number, i]) / avglat[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmedlat = (medlat[number, (nBlockssens + nBlocksppi - 1)] - medlat[number, i]) / medlat[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapbendmaxtot[number, i] - tapbendmaxnan[number, i]) >= 5 and (
                    tapbendmaxtot[number, (nBlockssens + nBlocksppi - 1)] - tapbendmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgbend = (avgbend[number, (nBlockssens + nBlocksppi - 1)] - avgbend[number, i]) / avgbend[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmedbend = (medbend[number, (nBlockssens + nBlocksppi - 1)] - medbend[number, i]) / medbend[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (angdispmaxtot[number, i] - angdispmaxnan[number, i]) >= 5 and (
                    angdispmaxtot[number, (nBlockssens + nBlocksppi - 1)] - angdispmaxnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgangvel = (avgangvelmax[number, (nBlockssens + nBlocksppi - 1)] - avgangvelmax[number, i]) / \
                               avgangvelmax[number, (nBlockssens + nBlocksppi - 1)]
                habmedangvel = (medangvelmax[number, (nBlockssens + nBlocksppi - 1)] - medangvelmax[number, i]) / \
                               medangvelmax[number, (nBlockssens + nBlocksppi - 1)]
            if (tapdeltaorienttot[number, i] - tapdeltaorientnan[number, i]) >= 5 and (
                    tapdeltaorienttot[number, (nBlockssens + nBlocksppi - 1)] - tapdeltaorientnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgorient = (avgorient[number, (nBlockssens + nBlocksppi - 1)] - avgorient[number, i]) / avgorient[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmedorient = (medorient[number, (nBlockssens + nBlocksppi - 1)] - medorient[number, i]) / medorient[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdisplacementtot[number, i] - tapdisplacementnan[number, i]) >= 5 and (
                    tapdisplacementtot[number, (nBlockssens + nBlocksppi - 1)] - tapdisplacementnan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgdisp = (avgdisp[number, (nBlockssens + nBlocksppi - 1)] - avgdisp[number, i]) / avgdisp[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmeddisp = (meddisp[number, (nBlockssens + nBlocksppi - 1)] - meddisp[number, i]) / meddisp[
                    number, (nBlockssens + nBlocksppi - 1)]
            if (tapdistancetot[number, i] - tapdistancenan[number, i]) >= 5 and (
                    tapdistancetot[number, (nBlockssens + nBlocksppi - 1)] - tapdistancenan[
                number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                habavgdist = (avgdist[number, (nBlockssens + nBlocksppi - 1)] - avgdist[number, i]) / avgdist[
                    number, (nBlockssens + nBlocksppi - 1)]
                habmeddist = (meddist[number, (nBlockssens + nBlocksppi - 1)] - meddist[number, i]) / meddist[
                    number, (nBlockssens + nBlocksppi - 1)]
                habavgduration = (avgduration[number, (nBlockssens + nBlocksppi - 1)] - avgduration[number, i]) / \
                                 avgduration[number, (nBlockssens + nBlocksppi - 1)]
                habmedduration = (medduration[number, (nBlockssens + nBlocksppi - 1)] - medduration[number, i]) / \
                                 medduration[number, (nBlockssens + nBlocksppi - 1)]

            # only calculate if SLC>0
            if slc[number, (nBlockssens + nBlocksppi - 1)] > 0:
                if (slctot[number, i] - slcnan[number, i]) >= 5 and (
                        slctot[number, (nBlockssens + nBlocksppi - 1)] - slcnan[
                    number, (nBlockssens + nBlocksppi - 1)]) >= 5:
                    habslc = (slc[number, (nBlockssens + nBlocksppi - 1)] - slc[number, i]) / slc[
                        number, (nBlockssens + nBlocksppi - 1)]
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

variables = ['avglat', 'medianlat', '%slc', '%llc', '%react', '%nomovmt', 'avgbend', 'medianbend', 'avgorient',
             'medianorient', '%rightturn', 'directionbias', '%towards', '%towardshalf',
             'avgdisp', 'mediandisp', 'avgdist', 'mediandist', 'avgduration', 'medianduration', 'avgangvelmax',
             'medianangvelmax',
             'slc/llc ratio', 'AUC',
             'ppislc', 'ppiavglat', 'ppimedlat', 'ppiavgbend', 'ppimedbend', 'ppiavgangvel', 'ppimedangvel',
             'ppiavgorient', 'ppimedorient', 'ppiavgdisp', 'ppimeddisp', 'ppiavgdist', 'ppimeddist', 'ppiavgduration',
             'ppimedduration',
             'habslc', 'habavglat', 'habmedlat', 'habavgbend', 'habmedbend', 'habavgangvel', 'habmedangvel',
             'habavgorient', 'habmedorient', 'habavgdisp', 'habmeddisp', 'habavgdist', 'habmeddist', 'habavgduration',
             'habmedduration']

genoresultdict = {}
statsdict = {}

# setup dictionary with results averaged over genotype into dictionary/dataframe, removing those with <5 tracked objects in block
for number in range(nBlocks):
    for i, element in enumerate(variables):
        if str(element) == '%slc' or str(element) == '%llc' or str(element) == '%react' or str(
                element) == '%nomovmt' or str(element) == 'slc/llc ratio' or str(element) == 'ppislc' or str(
                element) == 'habslc' or str(element) == 'AUC':
            events = 'totslcev'

        if str(element) == 'avglat' or str(element) == 'medianlat' or str(element) == 'ppiavglat' or str(
                element) == 'ppimedlat' or str(element) == 'habavglat' or str(element) == 'habmedlat' or str(
                element) == '%rightturn' or str(element) == 'directionbias' or str(element) == '%towards' or str(
                element) == '%towardshalf':
            events = 'totlatev'

        if str(element) == 'avgbend' or str(element) == 'medianbend' or str(element) == 'ppiavgbend' or str(
                element) == 'ppimedbend' or str(element) == 'habavgbend' or str(element) == 'habmedbend':
            events = 'totbendev'

        if str(element) == 'avgangvelmax' or str(element) == 'medianangvelmax' or str(element) == 'ppiavgangvel' or str(
                element) == 'ppimedangvel' or str(element) == 'habavgangvel' or str(element) == 'habmedangvel':
            events = 'totangvelev'

        if str(element) == 'avgorient' or str(element) == 'medianorient' or str(element) == 'ppiavgorient' or str(
                element) == 'ppimedorient' or str(element) == 'habavgorient' or str(element) == 'habmedorient':
            events = 'totorientev'

        if str(element) == 'avgdisp' or str(element) == 'mediandisp' or str(element) == 'ppiavgdisp' or str(
                element) == 'ppimeddisp' or str(element) == 'habavgdisp' or str(element) == 'habmeddisp':
            events = 'totdispev'

        if str(element) == 'avgdist' or str(element) == 'mediandist' or str(element) == 'ppiavgdist' or str(
                element) == 'ppimeddist' or str(element) == 'habavgdist' or str(element) == 'habmeddist' or str(
                element) == 'avgduration' or str(element) == 'medianduration' or str(
                element) == 'habavgduration' or str(element) == 'habmedduration' or str(
                element) == 'ppiavgduration' or str(element) == 'ppimedduration':
            events = 'totdistev'

        if float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)) == 0:
            if float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].std(
                    axis=0)) == 0:
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
                    '(avg-sib)/sibstd': np.nan
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
            if float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].std(
                    axis=0)) == 0:
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
                    '(avg-sib)/sibstd': np.nan
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
        genoresultdict[str(number + 1) + "-" + str(element) + "-" + "sib"] = {
            'event': (number + 1),
            'value': str(element),
            'genotype': "sib",
            'count': float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (dfresultT['genotype'] == 'wt')) & (
                        dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].count()),
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
                dfresultT[str(events)] >= 5), [str(element)]].std(axis=0)) == 0:
            if float(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].std(
                    axis=0)) == 0:
                statsdict[str("Tap-") + str(number + 1) + "-" + str(element)] = {
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
                statsdict[str("Tap-") + str(number + 1) + "-" + str(element)] = {
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
            if float(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (dfresultT['genotype'] == 'het')) & (
                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [str(element)]].std(
                    axis=0)) == 0:
                statsdict[str("Tap-") + str(number + 1) + "-" + str(element)] = {
                    '#mut': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                  dfresultT[str(events)] >= 5), [str(element)]].count()),
                    '#het': float(dfresultT.loc[
                                      (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                  dfresultT[str(events)] >= 5), [str(element)]].count()),
                    '#wt': float(dfresultT.loc[
                                     (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                 dfresultT[str(events)] >= 5), [str(element)]].count()),
                    'avgmut/avgwt': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)),
                    'avgmut/avgsib': np.nan,
                    '(avg-wt)/wtstd': (float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0)) - float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].mean(axis=0))) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                        str(element)]].std(axis=0)),
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
                if float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                        dfresultT[str(events)] >= 5), [str(element)]].mean(axis=0)) == 0:
                    statsdict[str("Tap-") + str(number + 1) + "-" + str(element)] = {
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
                                                                                           dfresultT[
                                                                                               str(events)] >= 5), [
                            str(element)]].mean(axis=0)),
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
                                    dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (dfresultT[
                                                                                                                  str(events)] >= 5), [
                            str(element)]].mean(axis=0))) / float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                    dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (dfresultT[
                                                                                                                  str(events)] >= 5), [
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
                    statsdict[str("Tap-") + str(number + 1) + "-" + str(element)] = {
                        '#mut': float(dfresultT.loc[
                                          (dfresultT['genotype'] == 'mut') & (dfresultT['event'] == (number + 1)) & (
                                                      dfresultT[str(events)] >= 5), [str(element)]].count()),
                        '#het': float(dfresultT.loc[
                                          (dfresultT['genotype'] == 'het') & (dfresultT['event'] == (number + 1)) & (
                                                      dfresultT[str(events)] >= 5), [str(element)]].count()),
                        '#wt': float(dfresultT.loc[
                                         (dfresultT['genotype'] == 'wt') & (dfresultT['event'] == (number + 1)) & (
                                                     dfresultT[str(events)] >= 5), [str(element)]].count()),
                        'avgmut/avgwt': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                            str(element)]].mean(axis=0)) / float(dfresultT.loc[(dfresultT['genotype'] == 'wt') & (
                                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                            str(element)]].mean(axis=0)),
                        'avgmut/avgsib': float(dfresultT.loc[(dfresultT['genotype'] == 'mut') & (
                                    dfresultT['event'] == (number + 1)) & (dfresultT[str(events)] >= 5), [
                            str(element)]].mean(axis=0)) / float(dfresultT.loc[((dfresultT['genotype'] == 'wt') | (
                                    dfresultT['genotype'] == 'het')) & (dfresultT['event'] == (number + 1)) & (
                                                                                           dfresultT[
                                                                                               str(events)] >= 5), [
                            str(element)]].mean(axis=0)),
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
                                    dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (dfresultT[
                                                                                                                  str(events)] >= 5), [
                            str(element)]].mean(axis=0))) / float(dfresultT.loc[((dfresultT['genotype'] == 'het') | (
                                    dfresultT['genotype'] == 'wt')) & (dfresultT['event'] == (number + 1)) & (dfresultT[
                                                                                                                  str(events)] >= 5), [
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
dfresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResultsByFishbyEvent.csv')
dfresultindivT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResultsByIndividualFish.csv')
dfgenoresultT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResultsByGenotype.csv')
dfstatsT.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResults_STATS.csv')
dfstatsanova.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResults_STATS_Anova.csv')
dfstatssigsibs.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResults_STATS_SigMutVSib.csv')
dfstatssighets.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResults_STATS_SigMutVHet.csv')
dfstatssigwts.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResults_STATS_SigMutVWT.csv')
dfstatslargediff.to_csv(inDir + 'Results/' + Gene + '/' + Name + '/TapResults_STATS_LargeDiff.csv')

# create heatmap for mut and het comparios to wildtype


print("Plotting results....")
# Sensitivity Blocks
plt.figure(figsize=(25, 50))
plt.subplot(11, 3, 1)
sns.boxplot(x='event', y='totslcev', data=dfresultT.loc[(dfresultT['event'] <= 2) | (dfresultT['event'] == 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total SLC Events")

plt.subplot(11, 3, 2)
sns.boxplot(x='event', y='totlatev', data=dfresultT.loc[(dfresultT['event'] <= 2) | (dfresultT['event'] == 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total Latency Events")

plt.subplot(11, 3, 4)
sns.boxplot(x='event', y='%slc',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - %SLC")

plt.subplot(11, 3, 5)
sns.boxplot(x='event', y='%llc',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - %LLC")

plt.subplot(11, 3, 6)
sns.boxplot(x='event', y='%react',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - %React")

plt.subplot(11, 3, 7)
sns.boxplot(x='event', y='%nomovmt',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - %No Movement")

plt.subplot(11, 3, 8)
sns.boxplot(x='event', y='%rightturn',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - %Right turn")

plt.subplot(11, 3, 9)
sns.boxplot(x='event', y='directionbias',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - R/L Bias")

plt.subplot(11, 3, 10)
sns.boxplot(x='event', y='%towards',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - %Towards stimulus")

plt.subplot(11, 3, 11)
sns.boxplot(x='event', y='%towardshalf',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - %Towards stimulus (midle half initial)")

plt.subplot(11, 3, 12)
sns.boxplot(x='event', y='avglat',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Average Latency")

plt.subplot(11, 3, 13)
sns.boxplot(x='event', y='medianlat',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Median Latency")

plt.subplot(11, 3, 14)
sns.boxplot(x='event', y='totbendev', data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7))],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total Bend Events")

plt.subplot(11, 3, 15)
sns.boxplot(x='event', y='avgbend',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Average Max Bend")

plt.subplot(11, 3, 16)
sns.boxplot(x='event', y='medianbend',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Median Max Bend")

plt.subplot(11, 3, 17)
sns.boxplot(x='event', y='totangvelev', data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7))],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total Ang Vel Max Events")

plt.subplot(11, 3, 18)
sns.boxplot(x='event', y='avgangvelmax', data=dfresultT.loc[
    ((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totangvelev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Average Max Ang Velocity")

plt.subplot(11, 3, 19)
sns.boxplot(x='event', y='medianangvelmax', data=dfresultT.loc[
    ((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totangvelev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Median Max Ang Velocity")

plt.subplot(11, 3, 20)
sns.boxplot(x='event', y='totorientev', data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7))],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total Orient Events")

plt.subplot(11, 3, 21)
sns.boxplot(x='event', y='avgorient', data=dfresultT.loc[
    ((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totorientev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Average Change in Orientation")

plt.subplot(11, 3, 22)
sns.boxplot(x='event', y='medianorient', data=dfresultT.loc[
    ((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totorientev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Median Change in Orientation")

plt.subplot(11, 3, 23)
sns.boxplot(x='event', y='totdispev', data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7))],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total Displacement Events")

plt.subplot(11, 3, 24)
sns.boxplot(x='event', y='avgdisp',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Average Displacement")

plt.subplot(11, 3, 25)
sns.boxplot(x='event', y='mediandisp',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Median Displacement")

plt.subplot(11, 3, 26)
sns.boxplot(x='event', y='totdistev', data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7))],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Total Displacement Events")

plt.subplot(11, 3, 27)
sns.boxplot(x='event', y='avgdist',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Average Distance")

plt.subplot(11, 3, 28)
sns.boxplot(x='event', y='mediandist',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Median Distance")

plt.subplot(11, 3, 29)
sns.boxplot(x='event', y='avgduration',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Sensitivity - Average Duration")

plt.subplot(11, 3, 30)
sns.boxplot(x='event', y='medianduration',
            data=dfresultT.loc[((dfresultT['event'] <= 2) | (dfresultT['event'] == 7)) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Sensitivity - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Sensitivity.pdf', bbox_inches='tight')

plt.clf()

# Sensitivity Values
plt.figure(figsize=(10, 5))
sns.boxplot(x='event', y='AUC', data=dfresultT.loc[(dfresultT['event'] == 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("AUC")
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Sensitivity - AUC.pdf', bbox_inches='tight')
plt.clf()

# Decision Making Values
plt.figure(figsize=(15, 5))
sns.boxplot(x='event', y='slc/llc ratio', data=dfresultT.loc[(dfresultT['totslcev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Decision Making")
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Decision Making.pdf', bbox_inches='tight')
plt.clf()

# PPI Blocks
plt.figure(figsize=(25, 50))
plt.subplot(11, 3, 1)
sns.boxplot(x='event', y='totslcev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Total SLC Events")

plt.subplot(11, 3, 2)
sns.boxplot(x='event', y='totlatev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Total Latency Events")

plt.subplot(11, 3, 4)
sns.boxplot(x='event', y='%slc',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - %SLC")

plt.subplot(11, 3, 5)
sns.boxplot(x='event', y='%llc',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - %LLC")

plt.subplot(11, 3, 6)
sns.boxplot(x='event', y='%react',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - %React")

plt.subplot(11, 3, 7)
sns.boxplot(x='event', y='%nomovmt',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - %No Movement")

plt.subplot(11, 3, 8)
sns.boxplot(x='event', y='%rightturn',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - %Right turn")

plt.subplot(11, 3, 9)
sns.boxplot(x='event', y='directionbias',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - R/L Bias")

plt.subplot(11, 3, 10)
sns.boxplot(x='event', y='%towards',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - %Towards stimulus")

plt.subplot(11, 3, 11)
sns.boxplot(x='event', y='%towardshalf',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - %Towards stimulus (midle half initial)")

plt.subplot(11, 3, 12)
sns.boxplot(x='event', y='avglat',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Average Latency")

plt.subplot(11, 3, 13)
sns.boxplot(x='event', y='medianlat',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Median Latency")

plt.subplot(11, 3, 14)
sns.boxplot(x='event', y='totbendev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Total Bend Events")

plt.subplot(11, 3, 15)
sns.boxplot(x='event', y='avgbend',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Average Max Bend")

plt.subplot(11, 3, 16)
sns.boxplot(x='event', y='medianbend',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Median Max Bend")

plt.subplot(11, 3, 17)
sns.boxplot(x='event', y='totangvelev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - Total Ang Vel Max Events")

plt.subplot(11, 3, 18)
sns.boxplot(x='event', y='avgangvelmax',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totangvelev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - Average Max Ang Velocity")

plt.subplot(11, 3, 19)
sns.boxplot(x='event', y='medianangvelmax',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totangvelev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Median Max Ang Velocity")

plt.subplot(11, 3, 20)
sns.boxplot(x='event', y='totorientev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Total Orient Events")

plt.subplot(11, 3, 21)
sns.boxplot(x='event', y='avgorient',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totorientev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - Average Change in Orientation")

plt.subplot(11, 3, 22)
sns.boxplot(x='event', y='medianorient',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totorientev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - Median Change in Orientation")

plt.subplot(11, 3, 23)
sns.boxplot(x='event', y='totdispev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - Total Displacement Events")

plt.subplot(11, 3, 24)
sns.boxplot(x='event', y='avgdisp',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Average Displacement")

plt.subplot(11, 3, 25)
sns.boxplot(x='event', y='mediandisp',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Median Displacement")

plt.subplot(11, 3, 26)
sns.boxplot(x='event', y='totdistev', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "PPI - Total Displacement Events")

plt.subplot(11, 3, 27)
sns.boxplot(x='event', y='avgdist',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Average Distance")

plt.subplot(11, 3, 28)
sns.boxplot(x='event', y='mediandist',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Median Distance")

plt.subplot(11, 3, 29)
sns.boxplot(x='event', y='avgduration',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Average Duration")

plt.subplot(11, 3, 30)
sns.boxplot(x='event', y='medianduration',
            data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("PPI - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PPI.pdf', bbox_inches='tight')

plt.clf()

# PPI Values
plt.figure(figsize=(25, 25))

plt.subplot(5, 3, 1)
sns.boxplot(x='event', y='ppislc', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - SLC")

plt.subplot(5, 3, 2)
sns.boxplot(x='event', y='ppiavglat', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Latency")

plt.subplot(5, 3, 3)
sns.boxplot(x='event', y='ppimedlat', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Latency")

plt.subplot(5, 3, 4)
sns.boxplot(x='event', y='ppiavgbend', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Bend Max")

plt.subplot(5, 3, 5)
sns.boxplot(x='event', y='ppimedbend', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Bend Max")

plt.subplot(5, 3, 6)
sns.boxplot(x='event', y='ppiavgangvel', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Ang vel Max")

plt.subplot(5, 3, 7)
sns.boxplot(x='event', y='ppimedangvel', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Ang vel Max")

plt.subplot(5, 3, 8)
sns.boxplot(x='event', y='ppiavgorient', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Reorientation")

plt.subplot(5, 3, 9)
sns.boxplot(x='event', y='ppimedorient', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Reorientation")

plt.subplot(5, 3, 10)
sns.boxplot(x='event', y='ppiavgdisp', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Displacement")

plt.subplot(5, 3, 11)
sns.boxplot(x='event', y='ppimeddisp', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Displacement")

plt.subplot(5, 3, 12)
sns.boxplot(x='event', y='ppiavgdist', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Distance")

plt.subplot(5, 3, 13)
sns.boxplot(x='event', y='ppimeddist', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Distance")

plt.subplot(5, 3, 14)
sns.boxplot(x='event', y='ppiavgduration', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Average Duration")

plt.subplot(5, 3, 15)
sns.boxplot(x='event', y='ppimedduration', data=dfresultT.loc[(dfresultT['event'] >= 3) & (dfresultT['event'] <= 6)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% PPI - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PPI_values.pdf', bbox_inches='tight')

plt.clf()

# Hab Blocks
plt.figure(figsize=(25, 50))
plt.subplot(11, 3, 1)
sns.boxplot(x='event', y='totslcev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total SLC Events")

plt.subplot(11, 3, 2)
sns.boxplot(x='event', y='totlatev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total Latency Events")

plt.subplot(11, 3, 4)
sns.boxplot(x='event', y='%slc', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - %SLC")

plt.subplot(11, 3, 5)
sns.boxplot(x='event', y='%llc', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - %LLC")

plt.subplot(11, 3, 6)
sns.boxplot(x='event', y='%react', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - %React")

plt.subplot(11, 3, 7)
sns.boxplot(x='event', y='%nomovmt', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totslcev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - %No Movement")

plt.subplot(11, 3, 8)
sns.boxplot(x='event', y='%rightturn', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - %Right turn")

plt.subplot(11, 3, 9)
sns.boxplot(x='event', y='directionbias', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - R/L Bias")

plt.subplot(11, 3, 10)
sns.boxplot(x='event', y='%towards', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - %Towards stimulus")

plt.subplot(11, 3, 11)
sns.boxplot(x='event', y='%towardshalf', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - %Towards stimulus (midle half initial)")

plt.subplot(11, 3, 12)
sns.boxplot(x='event', y='avglat', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Average Latency")

plt.subplot(11, 3, 13)
sns.boxplot(x='event', y='medianlat', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totlatev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Median Latency")

plt.subplot(11, 3, 14)
sns.boxplot(x='event', y='totbendev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total Bend Events")

plt.subplot(11, 3, 15)
sns.boxplot(x='event', y='avgbend', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - Average Max Bend")

plt.subplot(11, 3, 16)
sns.boxplot(x='event', y='medianbend', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totbendev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Median Max Bend")

plt.subplot(11, 3, 17)
sns.boxplot(x='event', y='totangvelev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total Ang Vel Max Events")

plt.subplot(11, 3, 18)
sns.boxplot(x='event', y='avgangvelmax',
            data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totangvelev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Average Max Ang Velocity")

plt.subplot(11, 3, 19)
sns.boxplot(x='event', y='medianangvelmax',
            data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totangvelev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Median Max Ang Velocity")

plt.subplot(11, 3, 20)
sns.boxplot(x='event', y='totorientev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total Orient Events")

plt.subplot(11, 3, 21)
sns.boxplot(x='event', y='avgorient', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totorientev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - Average Change in Orientation")

plt.subplot(11, 3, 22)
sns.boxplot(x='event', y='medianorient',
            data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totorientev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Median Change in Orientation")

plt.subplot(11, 3, 23)
sns.boxplot(x='event', y='totdispev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total Displacement Events")

plt.subplot(11, 3, 24)
sns.boxplot(x='event', y='avgdisp', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - Average Displacement")

plt.subplot(11, 3, 25)
sns.boxplot(x='event', y='mediandisp', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totdispev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - Median Displacement")

plt.subplot(11, 3, 26)
sns.boxplot(x='event', y='totdistev', data=dfresultT.loc[(dfresultT['event'] >= 7)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Total Displacement Events")

plt.subplot(11, 3, 27)
sns.boxplot(x='event', y='avgdist', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - Average Distance")

plt.subplot(11, 3, 28)
sns.boxplot(x='event', y='mediandist', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Median Distance")

plt.subplot(11, 3, 29)
sns.boxplot(x='event', y='avgduration', data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totdistev'] >= 5)],
            hue='genotype', hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title(
    "Habituation - Average Duration")

plt.subplot(11, 3, 30)
sns.boxplot(x='event', y='medianduration',
            data=dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['totdistev'] >= 5)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("Habituation - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Habituation.pdf', bbox_inches='tight')

plt.clf()

# Hab Values
plt.figure(figsize=(25, 25))

plt.subplot(5, 3, 1)
sns.boxplot(x='event', y='habslc', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - SLC")

plt.subplot(5, 3, 2)
sns.boxplot(x='event', y='habavglat', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Latency")

plt.subplot(5, 3, 3)
sns.boxplot(x='event', y='habmedlat', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Latency")

plt.subplot(5, 3, 4)
sns.boxplot(x='event', y='habavgbend', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Bend Max")

plt.subplot(5, 3, 5)
sns.boxplot(x='event', y='habmedbend', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Bend Max")

plt.subplot(5, 3, 6)
sns.boxplot(x='event', y='habavgangvel', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Ang vel Max")

plt.subplot(5, 3, 7)
sns.boxplot(x='event', y='habmedangvel', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Ang vel Max")

plt.subplot(5, 3, 8)
sns.boxplot(x='event', y='habavgorient', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Reorientation")

plt.subplot(5, 3, 9)
sns.boxplot(x='event', y='habmedorient', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Reorientation")

plt.subplot(5, 3, 10)
sns.boxplot(x='event', y='habavgdisp', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Displacement")

plt.subplot(5, 3, 11)
sns.boxplot(x='event', y='habmeddisp', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Displacement")

plt.subplot(5, 3, 12)
sns.boxplot(x='event', y='habavgdist', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Distance")

plt.subplot(5, 3, 13)
sns.boxplot(x='event', y='habmeddist', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Distance")

plt.subplot(5, 3, 14)
sns.boxplot(x='event', y='habavgduration', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Average Duration")

plt.subplot(5, 3, 15)
sns.boxplot(x='event', y='habmedduration', data=dfresultT.loc[(dfresultT['event'] >= 8)], hue='genotype',
            hue_order=['mut', 'het', 'wt'], palette="Reds_r").set_title("% Hab - Median Duration")

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Hab_values.pdf', bbox_inches='tight')

plt.clf()

# sensitivity heatmaps
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 3, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 3, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Sensitivity_Heatmaps.pdf', bbox_inches='tight')

plt.clf()

# PPI heatmaps
plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 4)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 5)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 4)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 6)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PPI_Heatmaps.pdf', bbox_inches='tight')

plt.clf()

# Hab heatmaps
plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 8)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 9)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 4)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 10)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-wt)/wtstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Habituation_Heatmaps.pdf', bbox_inches='tight')

plt.clf()

# sensitivity heatmaps mut v sib
plt.figure(figsize=(25, 10))
plt.subplot(2, 3, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 1)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 3, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 2)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 3, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Sensitivity_Heatmaps_MutvSib.pdf', bbox_inches='tight')

plt.clf()

# PPI heatmaps mut v sib
plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 3)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 4)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 5)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 4)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 6)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PPI_Heatmaps_MutvSib.pdf', bbox_inches='tight')

plt.clf()

# Hab heatmaps mut v sib
plt.figure(figsize=(25, 10))
plt.subplot(2, 2, 1)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 7)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 2)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 8)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 3)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 9)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.subplot(2, 2, 4)
grid = dfgenoresultT.loc[(dfgenoresultT['event'] == 10)].dropna()
sns.heatmap(grid.pivot(index='genotype', columns='event+value', values='(avg-sib)/sibstd').astype(float), vmin=-2.5,
            vmax=2.5, cmap='RdBu')

plt.tight_layout()

plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_Habituation_Heatmaps_MutvSib.pdf', bbox_inches='tight')

plt.clf()

# correlation heatmap for every value for each fish
totcorrheatmap = dfresultT.copy()

del totcorrheatmap['genotype']
del totcorrheatmap['event']
del totcorrheatmap['fish#']
del totcorrheatmap['totbendev']
del totcorrheatmap['totdispev']
del totcorrheatmap['totdistev']
del totcorrheatmap['totangvelev']

del totcorrheatmap['totlatev']
del totcorrheatmap['totorientev']
del totcorrheatmap['totslcev']

plt.figure(figsize=(25, 10))
sns.heatmap(totcorrheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_TotalCorrelation matrix.pdf', bbox_inches='tight')

propcorrheatmap = totcorrheatmap.copy()

del propcorrheatmap['habavgbend']
del propcorrheatmap['habavgdisp']
del propcorrheatmap['habavgdist']
del propcorrheatmap['habavglat']
del propcorrheatmap['habavgorient']
del propcorrheatmap['habavgduration']
del propcorrheatmap['habmedbend']
del propcorrheatmap['habavgangvel']
del propcorrheatmap['habmedangvel']
del propcorrheatmap['habmeddisp']
del propcorrheatmap['habmeddist']
del propcorrheatmap['habmedlat']
del propcorrheatmap['habmedorient']
del propcorrheatmap['habmedduration']
del propcorrheatmap['habslc']

del propcorrheatmap['ppiavgbend']
del propcorrheatmap['ppiavgdisp']
del propcorrheatmap['ppiavgdist']
del propcorrheatmap['ppiavglat']
del propcorrheatmap['ppiavgorient']
del propcorrheatmap['ppimedbend']
del propcorrheatmap['ppimeddisp']
del propcorrheatmap['ppimeddist']
del propcorrheatmap['ppimedlat']
del propcorrheatmap['ppimedorient']
del propcorrheatmap['ppislc']
del propcorrheatmap['ppiavgduration']
del propcorrheatmap['ppimedduration']
del propcorrheatmap['ppiavgangvel']
del propcorrheatmap['ppimedangvel']

plt.figure(figsize=(15, 5))
sns.heatmap(propcorrheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PropsCorrelation matrix.pdf', bbox_inches='tight')

# correlation heat map for ppi components
corrppiheatmap = dfresultT.loc[(dfresultT['event'] >= 7) & (dfresultT['event'] <= 10)].copy()

del corrppiheatmap['genotype']
del corrppiheatmap['%llc']
del corrppiheatmap['%nomovmt']
del corrppiheatmap['%react']
del corrppiheatmap['%slc']
del corrppiheatmap['AUC']
del corrppiheatmap['avgbend']
del corrppiheatmap['avgdisp']
del corrppiheatmap['avgdist']
del corrppiheatmap['avgduration']

del corrppiheatmap['avglat']
del corrppiheatmap['avgorient']
del corrppiheatmap['event']
del corrppiheatmap['fish#']
del corrppiheatmap['habavgbend']
del corrppiheatmap['habavgdisp']

del corrppiheatmap['habavgduration']
del corrppiheatmap['habmedduration']
del corrppiheatmap['habavgdist']
del corrppiheatmap['habavglat']
del corrppiheatmap['habavgorient']
del corrppiheatmap['habmedbend']
del corrppiheatmap['habmeddisp']
del corrppiheatmap['habmeddist']

del corrppiheatmap['habmedlat']
del corrppiheatmap['habmedorient']
del corrppiheatmap['habslc']
del corrppiheatmap['medianbend']
del corrppiheatmap['mediandisp']
del corrppiheatmap['mediandist']
del corrppiheatmap['medianduration']

del corrppiheatmap['medianlat']
del corrppiheatmap['medianorient']
del corrppiheatmap['slc/llc ratio']
del corrppiheatmap['totbendev']
del corrppiheatmap['totdispev']
del corrppiheatmap['totdistev']

del corrppiheatmap['totlatev']
del corrppiheatmap['totorientev']
del corrppiheatmap['totslcev']

plt.figure(figsize=(15, 5))
sns.heatmap(corrppiheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_PPICorrelation matrix.pdf', bbox_inches='tight')

# correlation heat map for hab components
corrhabheatmap = dfresultT.loc[(dfresultT['event'] >= 12)].copy()

del corrhabheatmap['genotype']
del corrhabheatmap['%llc']
del corrhabheatmap['%nomovmt']
del corrhabheatmap['%react']
del corrhabheatmap['%slc']
del corrhabheatmap['AUC']
del corrhabheatmap['avgbend']
del corrhabheatmap['avgdisp']
del corrhabheatmap['avgdist']

del corrhabheatmap['avglat']
del corrhabheatmap['avgorient']
del corrhabheatmap['avgduration']
del corrhabheatmap['medianduration']
del corrhabheatmap['event']
del corrhabheatmap['fish#']
del corrhabheatmap['ppiavgbend']
del corrhabheatmap['ppiavgdisp']

del corrhabheatmap['ppiavgdist']
del corrhabheatmap['ppiavglat']
del corrhabheatmap['ppiavgorient']
del corrhabheatmap['ppiavgduration']
del corrhabheatmap['ppimedbend']
del corrhabheatmap['ppimeddisp']
del corrhabheatmap['ppimeddist']
del corrhabheatmap['ppimedduration']

del corrhabheatmap['ppimedlat']
del corrhabheatmap['ppimedorient']
del corrhabheatmap['ppislc']
del corrhabheatmap['medianbend']
del corrhabheatmap['mediandisp']
del corrhabheatmap['mediandist']

del corrhabheatmap['medianlat']
del corrhabheatmap['medianorient']
del corrhabheatmap['slc/llc ratio']
del corrhabheatmap['totbendev']
del corrhabheatmap['totdispev']
del corrhabheatmap['totdistev']

del corrhabheatmap['totlatev']
del corrhabheatmap['totorientev']
del corrhabheatmap['totslcev']

plt.figure(figsize=(15, 5))
sns.heatmap(corrhabheatmap.astype(float).corr(method='pearson'), vmin=-0.6, vmax=0.6, cmap='RdBu')
plt.savefig(inDir + 'Results/' + Gene + '/' + Name + '/Plots_HabCorrelation matrix.pdf', bbox_inches='tight')
