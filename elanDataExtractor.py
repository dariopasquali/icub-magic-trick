import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'

def timeToJonas(time):
    h, m, sms = time.split(':')
    s, ms = sms.split('.')
    
    sm = str(m)
    ss = str(s)
    ssms = str(ms)
    
    if(len(sm) == 1):
        sm = "0" + sm
        
    if(len(sm) == 1):
        sm = "0" + sm
        
    if(len(ssms) == 1):
        ssms = "00" + ssms
        
    if(len(ssms) == 2):
        ssms = "0" + ssms
    
    return sm + "m" + ss + "s" + ssms

def jonasClass(cl):
    if("-L" in cl):
        return "L"
    else:
        return "T"
    
def darioClass(cl):
    if("-L" in cl):
        return cl.split("-")[0]
    else:
        return cl
    
def darioTarget(cl):
    if("-L" in cl):
        return 1
    else:
        return 0


def readDFWindows(fn):
    teen = pd.read_csv(fn, sep=',', names=['tire', 'remove', 'start', 'stop', 'class'])
    teen['start_jonas'] = teen['start'].apply(timeToJonas)
    teen['stop_jonas'] = teen['stop'].apply(timeToJonas)
    teen["shift_start"] = teen["start_jonas"].shift(-1)
    teen["shift_stop"] = teen["stop_jonas"].shift(-1)
    teen["shift_class"] = teen["class"].shift(-1)

    teen = teen.drop(['tire', 'start', 'stop', 'remove', 'class'], axis=1)   
    teen = teen.dropna()

    return teen


def readDFLinux(fn):
    teen = pd.read_csv(fn, sep=',', names=['tire', 'r0', 'start', 'r1', 'stop','r2', 'duration', 'r3', 'class'])
    teen['start_jonas'] = teen['start'].apply(timeToJonas)
    teen['stop_jonas'] = teen['stop'].apply(timeToJonas)
    teen["shift_start"] = teen["start_jonas"].shift(-1)
    teen["shift_stop"] = teen["stop_jonas"].shift(-1)
    teen["shift_class"] = teen["class"].shift(-1)

    teen = teen.drop(['tire', 'start', 'stop', 'r0', 'r1', 'r2', 'r3', 'class', 'duration'], axis=1)   
    teen = teen.dropna()

    return teen


def elanToLieDetection(fn, fout, mode="jonas", source="win", outmode=None):

    print("Convert {} to {}".format(fn, fout))
    
    teen = None
    if(source == "win"):
        teen = readDFWindows(fn)
    else:
        teen = readDFLinux(fn)

    colrename = []

    if(mode == "jonas"):
        colrename = ["start_question_0", "stop_question_0", "start_answer_0", "stop_answer_0", "class_0"]
        teen["shift_class"] = teen["shift_class"].apply(jonasClass)

        for i in range(1,6):    
            l1 = "start_question_{}".format(i)
            l2 = "stop_question_{}".format(i)
            l3 = "start_answer_{}".format(i)
            l4 = "stop_answer_{}".format(i)
            l5 = "class_{}".format(i)

            teen[l1] = teen['start_jonas'].shift(-1*i)
            teen[l2] = teen['stop_jonas'].shift(-1*i)
            teen[l3] = teen["shift_start"].shift(-1*i)
            teen[l4] = teen["shift_stop"].shift(-1*i)
            teen[l5] = teen["shift_class"].shift(-1*i)

            colrename.append(l1)
            colrename.append(l2)
            colrename.append(l3)
            colrename.append(l4)
            colrename.append(l5)

        teen = teen.dropna()  

    elif(mode == "dario"):
        teen["label"] = teen["shift_class"].apply(darioTarget)
        teen["shift_class"] = teen["shift_class"].apply(darioClass) 
        colrename = ["start_question", "stop_question", "start_answer", "stop_answer", "class", "label"]

    id = None
    if("\\" in fn):
        id = fn.split("\\")[-1]
    else:
        id = fn.split("/")[-1]    
    id = id.split(".")[0]
    teen["name"] = id
    teen["RIIR"] = "IR"
    teen["TL"] = ""

    colorder = ["name", "RIIR", "TL"]
    colrename.extend(colorder)
    colorder.extend(colrename[:-3])

    teen.columns = colrename
    teen = teen[colorder]

    if(outmode == None):
        return teen

    if(outmode=="append"):
        with open(fout, 'a') as f:
            teen.to_csv(f, header=False, sep=",", index=False)
    else:
        teen.to_csv(fout, header=False, sep=",", index=False)

    return teen




elanToLieDetection("data/annotations_lie_raw/s6.csv","data/annotations_lie/s6.csv",  source="linux", mode="dario", outmode="write")
elanToLieDetection("data/annotations_lie_raw/s11.csv","data/annotations_lie/s11.csv",  source="linux", mode="dario", outmode="write")
elanToLieDetection("data/annotations_lie_raw/s14.csv","data/annotations_lie/s14.csv",  source="linux", mode="dario", outmode="write")
elanToLieDetection("data/annotations_lie_raw/s24.csv","data/annotations_lie/s24.csv",  source="linux", mode="dario", outmode="write")
