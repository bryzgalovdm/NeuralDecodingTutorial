#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 21:56:43 2020
Modified: 14/12/2020 - Dataset wrapped as a class

@author: bryzgalovdm
"""
# %% Import libs
import numpy as np
import csv
import glob, os, sys

# %% Import Steinmetz dependency
os.chdir('/Users/bryzgalovdm/Documents/Steinmetz_dataset/')
import SteinmetzHelpers

# %% Provisory dict for areas
areas = {
    'motor_ctx': ['MOp', 'MOs'],
    'visual_ctx': ['VISa', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'],
    'hippocampus': ['CA1', 'CA3', 'DG', 'SUB', 'POST'],
    'thalamus': ['LD', 'LGd', 'LP', 'MD', 'MG', 'PO', 'POL', 'RT', 'VPL', 'VPM']
}

# %% Dataset class
class SteinmetzDataset:
    def __init__(self, path, eventtype, dt=0.05, dT=3.5, T0=1.5, brain='whole'):
        self.path = path
        self.eventtype = eventtype
        self.dt = dt # binsize
        self.dT = dT # overall length of epoch
        self.T0 = T0 # time before locking event
        self.brain = brain
        # Attributes to be loaded
        self.response = []
        self.brain_area = []
        self.spikes = []
        self.response_time = []
        self.feedback_time = []
        self.feedback_type = []
        self.contrast_right = []
        self.contrast_left = []
        self.onset = []

    def countNeuronsinArea(self, area):
        # Load data
        good_cells, brain_region, _ = SteinmetzHelpers.get_good_cells(self.path)
        # Count cells
        good_region = brain_region[good_cells]
        NN = len(good_region) # number of regions
        barea = np.zeros(NN, )
        barea[np.isin(good_region, area)] = 1
        return np.sum(barea)

    def load(self):
        # good cells and brain regions
        good_cells, brain_region, _ = SteinmetzHelpers.get_good_cells(self.path)
        # event timing
        response_times, visual_times, rsp, _, feedback_time = SteinmetzHelpers.get_event_times(self.path)
        # event types
        stimes, sclust = SteinmetzHelpers.get_spikes(self.path)
        response, vis_right, vis_left, feedback_type = SteinmetzHelpers.get_event_types(self.path)
        # trials loader
        if self.eventtype == 'stim':
            S = SteinmetzHelpers.psth(stimes, sclust, visual_times-self.T0, self.dT, self.dt)
        elif self.eventtype == 'resp':
            S = SteinmetzHelpers.psth(stimes, sclust, response_times-self.T0, self.dT, self.dt)
        elif self.eventtype == 'react':
            reaction_time = np.load(self.path + 'trials.reaction_times.npy')
            real_react_time = reaction_time[:,0].reshape(len(reaction_time),1)/1000
            tolocktimes_react = visual_times+real_react_time-self.T0
            tolocktimes_react[tolocktimes_react==np.inf] = response_times[tolocktimes_react==np.inf]
            S = SteinmetzHelpers.psth(stimes, sclust, tolocktimes_react, self.dT, self.dt)

        # % Do the data
        good_cells = good_cells * (np.mean(S, axis=(1,2))>0)
        S = S[good_cells].astype('int8')

        self.response = response
        self.brain_area = brain_region[good_cells]
        self.spikes = S
        self.response_time = rsp
        self.feedback_time = feedback_time
        self.feedback_type = feedback_type
        self.contrast_right = vis_right[:len(response)]
        self.contrast_left = vis_left[:len(response)]
        if self.eventtype == 'stim':
            self.onset = visual_times
        elif self.eventtype == 'resp':
            self.onset = response_times
        elif self.eventtype == 'react':
            self.onset = tolocktimes_react

        print(self.path + ' loaded successfully')


# %% Load data
def LoadSteinmetzData(datadir, eventtype='stim',
                      binsize=1/100, dT=3.5, T0=1.5,
                      sessionstoload='all'):
    '''
    LoadSteinmetzData(datadir, eventtype='stim',
                      binsize=1/100, dT=3.5, T0=1.5,
                      sessionstoload='all')

    Loads from folder with full Steinmetz dataset to alldat

    Args:
    datadir:        directory with full Steinmetz dataset
    eventtype:      event to lock onto ('stim' OR 'react' OR 'resp')
                    'react' works only if you specify RTdir - directory with reaction_times.npy
    binsize:        bin size to bin spiketrains (in s)
    dT:             length of trial to retrieve (in sec)
    T0:             length of pre-event period
    sessionstoload: sessions number to load into alldat
                    (you may put 'all' and it will load everything from datadir)


    Returns:
      alldat     : List with dictionaries: each dict is one recording:
                  alldat[0].spikes  - binned time-locked spiketrains with binsize
                  alldat[0].brain_area - location of each cluster in the brain
                  alldat[0].response - response type on each trial
                  alldat[0].response_time - response time on each trial
                  alldat[0].feedback_type - feedback type on each trial
                  alldat[0].feedback_time - feedback time type on each trial
                  alldat[0].contrast_right - proportion of contrast on each trial (right screen)
                  alldat[0].contrast_left - proportion of contrast on each trial (left screen)
                  alldat[0].onset - times of the event to locked onto


    Example:
      alldat = LoadSteinmetzData(datadir, eventype='react', sessionstoload=[0,2,3])

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
        alldat.append(SteinmetzDataset(fdir[idir], eventtype))
        alldat[idir].load()

    return alldat