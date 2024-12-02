#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on Mon Dec  2 08:13:06 2024
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'emotion_valence_word_recall'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1470, 956]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/aaryasaraswat/Documents/PSY310/SOLO PROJECT/emotion_valence_word_recall_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='event'
        )
    if deviceManager.getDevice('proceed_key') is None:
        # initialise proceed_key
        proceed_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='proceed_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='Pyglet',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='Pyglet'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    welcome = visual.TextStim(win=win, name='welcome',
        text='Welcome to the experiment!\n\nIn this task, you will see words presented one by one. Your goal is to remember as many words as possible.\n\nSome words will be positive, some negative, and some neutral. Try to remember all words, regardless of their type.\n\nAfter the presentation, you will be asked to recall as many words as you can.\n\nPress the space bar to begin.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    proceed_key = keyboard.Keyboard(deviceName='proceed_key')
    
    # --- Initialize components for Routine "word_presentation" ---
    a = visual.ShapeStim(
        win=win, name='a', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "recall_phase" ---
    recall_ins = visual.TextStim(win=win, name='recall_ins',
        text='Please type all the words you remember in the box below. Press "Space" when done.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    recall_box = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='recall_box',
         depth=-2, autoLog=True,
    )
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[welcome, proceed_key],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for proceed_key
    proceed_key.keys = []
    proceed_key.rt = []
    _proceed_key_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome* updates
        
        # if welcome is starting this frame...
        if welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome.frameNStart = frameN  # exact frame index
            welcome.tStart = t  # local t and not account for scr refresh
            welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome.started')
            # update status
            welcome.status = STARTED
            welcome.setAutoDraw(True)
        
        # if welcome is active this frame...
        if welcome.status == STARTED:
            # update params
            pass
        
        # *proceed_key* updates
        waitOnFlip = False
        
        # if proceed_key is starting this frame...
        if proceed_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            proceed_key.frameNStart = frameN  # exact frame index
            proceed_key.tStart = t  # local t and not account for scr refresh
            proceed_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(proceed_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'proceed_key.started')
            # update status
            proceed_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(proceed_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(proceed_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if proceed_key.status == STARTED and not waitOnFlip:
            theseKeys = proceed_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _proceed_key_allKeys.extend(theseKeys)
            if len(_proceed_key_allKeys):
                proceed_key.keys = _proceed_key_allKeys[-1].name  # just the last key pressed
                proceed_key.rt = _proceed_key_allKeys[-1].rt
                proceed_key.duration = _proceed_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if proceed_key.keys in ['', [], None]:  # No response was made
        proceed_key.keys = None
    thisExp.addData('proceed_key.keys',proceed_key.keys)
    if proceed_key.keys != None:  # we had a response
        thisExp.addData('proceed_key.rt', proceed_key.rt)
        thisExp.addData('proceed_key.duration', proceed_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('word_stim.csv'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "word_presentation" ---
        # create an object to store info about Routine word_presentation
        word_presentation = data.Routine(
            name='word_presentation',
            components=[a, text],
        )
        word_presentation.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text.setText(Word)
        # store start times for word_presentation
        word_presentation.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        word_presentation.tStart = globalClock.getTime(format='float')
        word_presentation.status = STARTED
        thisExp.addData('word_presentation.started', word_presentation.tStart)
        word_presentation.maxDuration = None
        # keep track of which components have finished
        word_presentationComponents = word_presentation.components
        for thisComponent in word_presentation.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "word_presentation" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        word_presentation.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *a* updates
            
            # if a is starting this frame...
            if a.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                a.frameNStart = frameN  # exact frame index
                a.tStart = t  # local t and not account for scr refresh
                a.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(a, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'a.started')
                # update status
                a.status = STARTED
                a.setAutoDraw(True)
            
            # if a is active this frame...
            if a.status == STARTED:
                # update params
                pass
            
            # if a is stopping this frame...
            if a.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > a.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    a.tStop = t  # not accounting for scr refresh
                    a.tStopRefresh = tThisFlipGlobal  # on global time
                    a.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'a.stopped')
                    # update status
                    a.status = FINISHED
                    a.setAutoDraw(False)
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                word_presentation.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in word_presentation.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "word_presentation" ---
        for thisComponent in word_presentation.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for word_presentation
        word_presentation.tStop = globalClock.getTime(format='float')
        word_presentation.tStopRefresh = tThisFlipGlobal
        thisExp.addData('word_presentation.stopped', word_presentation.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if word_presentation.maxDurationReached:
            routineTimer.addTime(-word_presentation.maxDuration)
        elif word_presentation.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "recall_phase" ---
    # create an object to store info about Routine recall_phase
    recall_phase = data.Routine(
        name='recall_phase',
        components=[recall_ins, recall_box],
    )
    recall_phase.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    # Create a Clock to track reaction time
    response_clock = core.Clock()  # Create a clock for reaction time
    
    
    recall_box.reset()
    # store start times for recall_phase
    recall_phase.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    recall_phase.tStart = globalClock.getTime(format='float')
    recall_phase.status = STARTED
    thisExp.addData('recall_phase.started', recall_phase.tStart)
    recall_phase.maxDuration = None
    # keep track of which components have finished
    recall_phaseComponents = recall_phase.components
    for thisComponent in recall_phase.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "recall_phase" ---
    recall_phase.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from code
        from psychopy import event
        
        # Check if the space key was pressed
        if 'space' in event.getKeys():
            continueRoutine = False  # End the routine
            
        # Record reaction time when the participant finishes their response
        if 'space' in event.getKeys():
            reaction_time = response_clock.getTime()  # Capture the time at the space key press
            continueRoutine = False  # End the routine when space is pressed
        
        
        
        # *recall_ins* updates
        
        # if recall_ins is starting this frame...
        if recall_ins.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            recall_ins.frameNStart = frameN  # exact frame index
            recall_ins.tStart = t  # local t and not account for scr refresh
            recall_ins.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_ins, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_ins.started')
            # update status
            recall_ins.status = STARTED
            recall_ins.setAutoDraw(True)
        
        # if recall_ins is active this frame...
        if recall_ins.status == STARTED:
            # update params
            pass
        
        # if recall_ins is stopping this frame...
        if recall_ins.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > recall_ins.tStartRefresh + 3.0-frameTolerance:
                # keep track of stop time/frame for later
                recall_ins.tStop = t  # not accounting for scr refresh
                recall_ins.tStopRefresh = tThisFlipGlobal  # on global time
                recall_ins.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'recall_ins.stopped')
                # update status
                recall_ins.status = FINISHED
                recall_ins.setAutoDraw(False)
        
        # *recall_box* updates
        
        # if recall_box is starting this frame...
        if recall_box.status == NOT_STARTED and tThisFlip >= 3.0-frameTolerance:
            # keep track of start time/frame for later
            recall_box.frameNStart = frameN  # exact frame index
            recall_box.tStart = t  # local t and not account for scr refresh
            recall_box.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(recall_box, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'recall_box.started')
            # update status
            recall_box.status = STARTED
            recall_box.setAutoDraw(True)
        
        # if recall_box is active this frame...
        if recall_box.status == STARTED:
            # update params
            pass
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            recall_phase.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in recall_phase.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "recall_phase" ---
    for thisComponent in recall_phase.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for recall_phase
    recall_phase.tStop = globalClock.getTime(format='float')
    recall_phase.tStopRefresh = tThisFlipGlobal
    thisExp.addData('recall_phase.stopped', recall_phase.tStop)
    thisExp.addData('recall_box.text',recall_box.text)
    thisExp.nextEntry()
    # the Routine "recall_phase" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
