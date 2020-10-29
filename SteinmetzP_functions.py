#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:56:43 2020

@author: bryzgalovdm
"""
# %% Import libs
import numpy as np
import csv
import glob, os, sys
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore, spearmanr, mannwhitneyu, pearsonr
import time
import imp
from scipy.sparse import csr_matrix

# %% Import Steinmetz dependency
os.chdir('/Users/bryzgalovdm/Documents/Steinmetz_dataset/')
import SteinmetzHelpers

# %% Load data
def LoadSteinmetzData(datadir, tlockevent='stim',
                      binsize=1/100, dT=3.5, T0=1.5,
                      sessionstoload='all'):
    '''
    LoadSteinmetzData(datadir, tlockevent='stim',
                      binsize=1/100, dT=3.5, T0=1.5,
                      sessionstoload='all')

    Loads from folder with full Steinmetz dataset to alldat

    Args:
    datadir:        directory with full Steinmetz dataset
    tlockevent:     event to lock onto ('stim' OR 'react' OR 'resp')
                    'react' works only if you specify RTdir - directory with reaction_times.npy
    binsize:        bin size to bin spiketrains (in s)
    dT:             length of trial to retrieve (in sec)
    T0:             length of pre-event period
    sessionstoload: sessions number to load into alldat
                    (you may put 'all' and it will load everything from datadir)


    Returns:
      alldat     : List with dictionaries: each dict is one recording:
                  alldat[0]['spks']  - binned time-locked spiketrains with binsize
                  alldat[0]['brainarea'] - location of each cluster in the brain
                  alldat[0]['response'] - response type on each trial
                  alldat[0]['response_time'] - response time on each trial
                  alldat[0]['feedback_type'] - feedback type on each trial
                  alldat[0]['feedback_time'] - feedback time type on each trial
                  alldat[0]['contrast_right'] - proportion of contrast on each trial (right screen)
                  alldat[0]['contrast_left'] - proportion of contrast on each trial (left screen)
                  alldat[0]['bin_size'] - binsize
                  alldat[0]['onset'] - times of the event to locked onto


    Example:
      alldat = LoadSteinmetzData(datadir, tlockevent='react', sessionstoload=[0,2,3])

'''

    alldat = []
    # Check the arguments
    fdir = glob.glob(os.path.join(datadir, "*", "")) # List the folders in datadir

    if type(sessionstoload) is str:
        if sessionstoload == 'all':
            numsessions = np.arange(len(fdir))
            print(f'We will load all {len(fdir)} sessions')
        else:
            raise ValueError('If sessionstoload is string, it could take only "all" value')
    elif type(sessionstoload) is list or type(sessionstoload) is np.ndarray:
        numsessions = sessionstoload
        print(f'We will load {len(sessionstoload)} sessions out of {len(fdir)}')
    elif type(sessionstoload) is int:
        numsessions = [sessionstoload]
        print(f'We will load the session number {sessionstoload+1}')
    else:
        raise TypeError("sessionstoload is either 'all' or list or numpy array or int")

    for idir in range(len(numsessions)):
        # good cells and brain regions
        good_cells, brain_region, br = SteinmetzHelpers.get_good_cells(fdir[idir])
        # spikes
        stimes, sclust = SteinmetzHelpers.get_spikes(fdir[idir])
        # event types
        response, vis_right, vis_left, feedback_type = SteinmetzHelpers.get_event_types(fdir[idir])
        # event timing
        response_times, visual_times, rsp, gocue, feedback_time = SteinmetzHelpers.get_event_times(fdir[idir])
        # trials loader
        if tlockevent == 'stim':
            S = SteinmetzHelpers.psth(stimes, sclust, visual_times-T0, dT, binsize)
        elif tlockevent == 'resp':
            S = SteinmetzHelpers.psth(stimes, sclust, response_times-T0, dT, binsize)
        elif tlockevent == 'react':
            reaction_time = np.load(fdir[idir] + 'trials.reaction_times.npy')
            real_react_time = reaction_time[:,0].reshape(len(reaction_time),1)/1000
            tolocktimes_react = visual_times+real_react_time-T0
            tolocktimes_react[tolocktimes_react==np.inf] = response_times[tolocktimes_react==np.inf]
            S = SteinmetzHelpers.psth(stimes, sclust, tolocktimes_react, dT, binsize)


        # % Do the data
        good_cells = good_cells * (np.mean(S, axis=(1,2))>0)
        S  = S[good_cells].astype('int8')

        alldat.append({})
        alldat[idir]['response'] = response
        ntrials = len(alldat[idir]['response'])

        alldat[idir]['brain_area'] = brain_region[good_cells]
        alldat[idir]['spks'] = S
        alldat[idir]['response_time'] = rsp
        alldat[idir]['feedback_time'] = feedback_time
        alldat[idir]['feedback_type'] = feedback_type
        alldat[idir]['contrast_right'] = vis_right[:ntrials]
        alldat[idir]['contrast_left'] = vis_left[:ntrials]
        alldat[idir]['bin_size'] = binsize
        alldat[idir]['onset'] = visual_times
        if tlockevent == 'react':
            alldat[idir]['reaction_times'] = tolocktimes_react

    print('Loaded successfully')

    return alldat

# %% Taken from tutorials
def plot_weights(models, sharey=True):
  """Draw a stem plot of weights for each model in models dict."""
  n = len(models)
  f = plt.figure(figsize=(10, 2.5 * n))
  axs = f.subplots(n, sharex=True, sharey=sharey)
  axs = np.atleast_1d(axs)

  for ax, (title, model) in zip(axs, models.items()):

    ax.margins(x=.02)
    stem = ax.stem(model.coef_.squeeze(), use_line_collection=True)
    stem[0].set_marker(".")
    stem[0].set_color(".2")
    stem[1].set_linewidths(.5)
    stem[1].set_color(".2")
    stem[2].set_visible(False)
    ax.axhline(0, color="C3", lw=3)
    ax.set(ylabel="Weight", title=title)
  ax.set(xlabel="Neuron (a.k.a. feature)")
  f.tight_layout()

# %% It is legacy function
def LoadReactionTime(RTdir, inalldat=False, alldat=[], sessionstoload='all'):

    # Sessions to load handling
    if type(sessionstoload) is str:
        if sessionstoload == 'all':
            numsessions = np.arrange(len(alldat))
        else:
            raise TypeError('If sessionstoload is string, it could take only "all" value')
    elif type(sessionstoload) is list or type(sessionstoload) is np.array:
        numsessions = sessionstoload
        print(f'We will load {len(sessionstoload)} sessions out of {len(len(fdir))}')
    elif type(sessionstoload) is int:
        numsessions = [sessionstoload]
        print(f'We will load the session number {sessionstoload+1}')
    else:
        raise TypeError("sessionstoload is either 'all' or list or numpy array or int")

    # Load the file
    reaction_times = np.load(RTdir + '/reaction_times.npy', allow_pickle=True)
    reaction_times = reaction_times[numsessions]

    # If you need to incorporate this in alldat
    if inalldat:

        for idir in numsessions:
            alldat[idir]['reaction_time'] = reaction_times[idir]

        return alldat
    else:
        return reaction_times