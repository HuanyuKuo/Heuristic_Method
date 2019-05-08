import numpy as np
import matplotlib
import math
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#from scipy.optimize import curve_fit
from lmfit import Model, Parameters
from scipy import special
import os
from os.path import expanduser
import sys
sys.float_info.max

matplotlib.rc('xtick', labelsize=15)     
matplotlib.rc('ytick', labelsize=15)

###########################################################################
## Global variables
###########################################################################
Pop = 8
TotalTimePoint = 9
OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/0307/"
if not os.path.exists(OutputFilePath):
    os.makedirs(OutputFilePath)
BarcodeCountPath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/SourceFile/"

bluish_green = "#009E73"


def read_MeanFitness(Pop):
    InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/Pop"+str(Pop)+"/"
    FilePathName = InputFilePath+"P"+str(Pop)+"_It5_MeanFitness.txt"
    file = open(FilePathName,"r")
    triplets=file.read().split()
    file.close()
    y=np.array(triplets, dtype=np.float32)
    return y

def read_Fitness(Pop):
    InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/Pop"+str(Pop)+"/"
    FilePathName = InputFilePath+"P"+str(Pop)+"_It5_RelativeFitness.txt"
    file = open(FilePathName,"r")
    triplets=file.read().split()
    file.close()
    y=np.array(triplets, dtype=np.float32)
    #print(y.shape)
    return np.transpose(np.reshape(y,(-1,TotalTimePoint-1)))

def read_Adpative_LNs(Pop):
    InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/Pop"+str(Pop)+"/"
    FilePathName = InputFilePath+"P"+str(Pop)+"_It5_ListAdaptiveBarcodeIndex.txt"
    file = open(FilePathName,"r")
    triplets=file.read().split()
    file.close()
    y=np.array(triplets, dtype=np.int32)
    return y
	
def readBarcodeCount(Pop):
    global BarcodeCountPath, TotalTimePoint 
    FileName = "Pop"+str(Pop)+"BowtieReadCounts"
    FilePathName = BarcodeCountPath +FileName+".txt"
    file = open(FilePathName,"r")
    triplets=file.read().split()
    file.close()
    A=np.array(triplets, dtype=np.int32)
    return  np.transpose(np.reshape(A,(-1,TotalTimePoint)))

def get_Barcode_Freq(BarcodeCountsArray):  # BarcodeCountsArray[t][bc] Add 0.5 to zero count to aviod divergent of log10(freqeuncy)
    AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(0,len(BarcodeCountsArray))]
    #BarcodeFreqArray =  np.asarray([[(count+ int(count==0)*0.5)/AllCountsArray[t] for count in BarcodeCountsArray[t]] for t in range(0,len(BarcodeCountsArray))])
    BarcodeFreqArray =  np.asarray([[count/AllCountsArray[t]+ int(count==0)*10**-8 for count in BarcodeCountsArray[t]] for t in range(0,len(BarcodeCountsArray))])
    return BarcodeFreqArray, AllCountsArray


def plot_CleanAdaptive_BCFeq(Pop,time_end,ListAdaptive,ListCleanAdaptive):
    global OutputFilePath, TotalTimePoint 
    # read barcode count
    BarcodeCountsArray = readBarcodeCount(Pop)
    # get Barcode Freq
    AllCountsArray = np.sum(BarcodeCountsArray,axis=1)
    BarcodeFreqArray =  [[(count/AllCountsArray[t]+ int(count==0)*10**-8) for count in BarcodeCountsArray[t]] for t in range(0,TotalTimePoint)]
    # plot Frequency for Adaptive Lineages
    plt.figure()
    x = np.asarray([2*i for i in range(TotalTimePoint)])
    plt.ylabel("log10 Barcode Freqeuncy")
    plt.xlabel("Time (cycle)")
    plt.title("Adaptive Lineages, Red:" +str(np.size(ListCleanAdaptive))+" Grey:"  +str(np.size(ListAdaptive)-np.size(ListCleanAdaptive)) )
    plt.xlim(0,np.max(x)+1)
    for bc in ListAdaptive:
        y = [np.log10(BarcodeFreqArray[t][bc]) for t in range(0,len(x))]
        if bc in ListCleanAdaptive:
            plt.plot(x,y,color='red',alpha=0.3, linewidth = 0.3)
        else:
            plt.plot(x,y,color='black',alpha=1, linewidth = 0.3)
    plt.savefig(OutputFilePath+"Adaptive_Lineages_Clean_Pop"+str(Pop)+".png")
'''
def plot_DEF_shift(Pop,ListCleanAdaptive,t,t_mid,freq,Med):
    finess = read_Fitness(Pop)
    binwidth = 0.1
    NLs = np.size(ListCleanAdaptive)
    #data = finess[t][ListCleanAdaptive]
    data = finess[t_mid][ListCleanAdaptive]
    data2 = finess[t+1][ListCleanAdaptive]
    #BarcodeCountsArray = readBarcodeCount(Pop)
    #freq = BarcodeFreqArray[t][ListCleanAdaptive]
    ds = data2 - data
    Mean = np.mean(ds)
    Med = np.median(ds)
    Var = np.var(ds)
    # plot without shift
    plt.figure()
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    #plt.hist(data,bins,label="cycle"+str(2*t+1),color= 'blue', alpha=0.3)
    plt.hist(data,bins,label="cycle"+str(2*t_mid+1),color= 'blue', alpha=0.3)
    bins = np.arange(min(data2), max(data2) + binwidth, binwidth)
    plt.hist(data2,bins,label="cycle"+str(2*t+3),color= 'green', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel("fitness (1/cycle)")
    plt.title("Adaptive Lineages, NLs="+str(NLs))
    plt.savefig(OutputFilePath+"CleanAdaptive_DFE_Pop"+str(Pop)+"_T"+str(t)+".png")
    # plot scatter
    plt.figure()
    line_max = max(np.max(data),np.max(data2))
    line_min = min(np.min(data),np.min(data2))
    line = np.arange(line_min, line_max, 0.01)
    plt.plot(line,line, 'k--')
    plt.plot(line,line+Med, 'r--')
    colormap = plt.cm.jet(np.linspace(0,1,101)) 
    colmax = -1
    colmin = -4
    colbin = (colmax-colmin)/100
    for bc in range(np.size(ListCleanAdaptive)):
        s = np.log10(freq[bc])
        if s>=colmax:
            s= colmax
        elif s< colmin:
            s = colmin
        colorIndx = (int)((s-colmin)/colbin)
        plt.scatter(data[bc],data2[bc], s=50, facecolors='none', edgecolors=colormap[colorIndx])
    plt.title("Pop "+str(Pop) + ",  NLs="+str(np.size(ListCleanAdaptive)))
    plt.xlabel("s  cycle " + str(2*t_mid+1))
    plt.ylabel("s  cycle " + str(2*t+3))
    plt.savefig(OutputFilePath+"CleanAdaptive_Pop"+str(Pop)+"_s"+str(t_mid)+"_s"+str(t+1)+".png")
    # plot with shift
    data2 = data2 - Med
    plt.figure()
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    plt.hist(data,bins,label="cycle"+str(2*t_mid+1),color= 'blue', alpha=0.3)
    bins = np.arange(min(data2), max(data2) + binwidth, binwidth)
    plt.hist(data2,bins,label="cycle"+str(2*t+3),color= 'green', alpha=0.3)
    plt.legend(loc='best')
    plt.xlabel("fitness (1/cycle)")
    plt.title("Adaptive Lineages, NLs="+str(NLs))
    plt.savefig(OutputFilePath+"CleanAdaptive_DFE_Shift_Pop"+str(Pop)+"_T"+str(t)+".png")
    # plot ds
    plt.figure()
    bins = np.arange(min(ds), max(ds) + binwidth, binwidth)
    plt.hist(ds,bins,label="NLs="+str(NLs),color= 'black', alpha=0.4)
    plt.xlabel("s(cycle "+str(2*t+3)+") - s(cycle "+str(2*t_mid+1)+")")
    plt.title("Mean "+str(Mean)+" Median "+str(Med)+" Var "+str(Var))
    plt.legend(loc='best')
    plt.savefig(OutputFilePath+"CleanAdaptive_ds_Dist_Pop"+str(Pop)+"_T"+str(t)+".png")
    plt.close('all')

def DFE_shift(Pop,t_mid):
    global TotalTimePoint
    Adaptive_NLs = read_Adpative_LNs(Pop)
    BarcodeCountsArray = readBarcodeCount(Pop)
    BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)
    finess = read_Fitness(Pop)
    mean_fitness = read_MeanFitness(Pop)
    Median_ds = []
    Var_ds = []
    for t in range(t_mid,TotalTimePoint-2):
        #listNonzeros = np.nonzero(BarcodeCountsArray[t]>10)[0]
        listNonzeros = np.nonzero(BarcodeCountsArray[t_mid]>100)[0]
        listNonzeros2 = np.nonzero(BarcodeCountsArray[t+1]>100)[0]
        ListCleanAdaptive = list(set(listNonzeros)&set(listNonzeros2)&set(Adaptive_NLs))
        #ds = finess[t+1][ListCleanAdaptive] - finess[t][ListCleanAdaptive]
        ds = finess[t+1][ListCleanAdaptive] - finess[t_mid][ListCleanAdaptive]
        Median_ds.append(np.median(ds))
        Var_ds.append(np.var(ds))
        freq = BarcodeFreqArray[t_mid][ListCleanAdaptive]
        plot_DEF_shift(Pop,ListCleanAdaptive,t,t_mid,freq,np.median(ds))
    #t0=3
    #sum_Ds = [np.sum(Median_ds[t0:t]) for t in range(t0,np.size(Median_ds)+1)]
    Alignment_mean_fitness = [m for m in mean_fitness[0:t_mid+1]]
    for t in range(t_mid+1,TotalTimePoint-1):
        #Alignment_mean_fitness[t] = Alignment_mean_fitness[t]-sum_Ds[t-t0]
        Alignment_mean_fitness.append(mean_fitness[t]-Median_ds[t-t_mid-1])
    plt.figure()
    x = np.asarray([1+2*i for i in range(np.size(mean_fitness))])
    plt.plot(x,Alignment_mean_fitness,color='blue',marker='o',ls='solid',label='new mean fitness')
    plt.plot(x,mean_fitness,color='black',marker='o',ls='solid',label='old mean fitness')
    plt.legend(loc='best')
    plt.xlabel('Time (cycle)')
    plt.ylabel('Mean Fitness (1/cycle)')
    plt.savefig(OutputFilePath+"AlignmingMeanFitness_Pop"+str(Pop)+".png")

'''

'''
def find_Adaptives(Pop,t_mid,D_fitness,freq):
    CV_CUT = 0.25
    Freq_CUT = -2.5
    S_CUT = 0
    BarcodeFreqArray = freq
    #listlarge = np.nonzero(np.min(BarcodeCountsArray[t_mid-2:t_mid+1],axis=0)>Count_CUT)[0]
    #listlarge = np.nonzero(np.log10(np.min(BarcodeFreqArray[t_mid-2:t_mid+1],axis=0))>Freq_CUT)[0]
    listlarge = np.nonzero(np.log10(BarcodeFreqArray[t_mid+1])>Freq_CUT)[0]
    fitness = [read_Fitness(Pop)[t] - D_fitness[t] for t in range(t_mid-2,t_mid+1)]
    #fitness = 
    s_Mean = np.mean(fitness, axis=0)
    s_STD = np.std(fitness, axis=0)
    s_CV= np.abs(np.asarray([std/mean for std,mean in zip(s_STD,s_Mean)]))
    CV_freq = np.asarray([cv/f*10**-3 for cv,f in zip(s_CV,freq[t_mid])])
    #print(CV_freq.shape)
    listCV = np.nonzero(CV_freq < CV_CUT)[0]  #np.nonzero(s_CV < CV_CUT)[0]
    listMean = np.nonzero(s_Mean > S_CUT )[0]
    listADP = list(set(listlarge))#list(set(listCV) & set(listMean) & set(listlarge))
    print(np.size(listADP))
    
    plt.figure()
    #plt.plot(s_Mean,s_CV,'k.',markersize=2)
    #plt.plot(s_Mean[listlarge],s_CV[listlarge],'b.',markersize=2)
    plt.plot(s_Mean[listlarge],CV_freq[listlarge],'b.',markersize=2)
    #plt.plot(s_Mean[listlarge],np.exp(-1*s_CV[listlarge]),'b.',markersize=2)
    #plt.plot(np.log10(freq[t_mid][listlarge]),s_CV[listlarge],'b.',markersize=2)
    #plt.plot(s_Mean[listADP],s_CV[listADP],'r.',markersize=3)
    plt.plot(s_Mean[listADP],CV_freq[listADP],'r.',markersize=3)
    #plt.plot(np.log10(freq[t_mid][listADP]),s_CV[listADP],'r.',markersize=3)
    #plt.plot(s_Mean[listADP],np.exp(-1*s_CV[listADP]),'r.',markersize=3)
    plt.ylim(0,2)
    plt.xlim(0,1.5)
    plt.xlabel("Mean of fitness")
    #plt.xlabel("log10 Barcode Freq")
    plt.ylabel("abs CV of fitness")
    #plt.ylabel("Exp -| CV of fitness|")
    plt.title("P"+str(Pop)+" Cycle"+str(2*t_mid-3)+"-"+str(2*t_mid+1))
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_s2CV_Cycle"+str(2*t_mid+1)+".png")
    #plt.savefig(OutputFilePath+"P"+str(Pop)+"_s2ExpCV_Cycle"+str(2*t_mid+1)+".png")
    plt.close('all')
    
    return listADP
'''
def find_largeLNs(Pop,t_mid,freq):
    Freq_CUT = -3.0
    listlarge = np.nonzero(np.log10(freq[t_mid+1])>Freq_CUT)[0]
    listADP = list(set(listlarge))
    return listADP

def plot_S1S2(Pop,list,t1,t2,D_fitness,Freq):
    global OutputFilePath
    s1 = read_Fitness(Pop)[t1][list] - D_fitness[t1]
    s2 = read_Fitness(Pop)[t2][list] - D_fitness[t2]
    freq = Freq[t1][list]
    ds = np.asarray([i-j for i,j in zip(s2,s1)])
    Med = np.median(ds)
    Mean = np.mean(ds)
    WeightMean = np.average(ds,weights=freq)
    line_max = max(np.max(s1),np.max(s2))
    line_min = min(np.min(s1),np.min(s2))
    line = np.arange(line_min, line_max, 0.01)
    plt.figure()
    plt.plot(line,line, 'k--')
    plt.plot(line,line+WeightMean, 'r--')
    #plt.plot(line,line+Med, 'r--')
    colormap = plt.cm.jet(np.linspace(0,1,101)) 
    colmax = -1
    colmin = -4
    colbin = (colmax-colmin)/100
    for bc in range(np.size(list)):
        s = np.log10(freq[bc])
        if s>=colmax:
            s= colmax
        elif s< colmin:
            s = colmin
        colorIndx = (int)((s-colmin)/colbin)
        plt.scatter(s1[bc],s2[bc],s=20, facecolors='none', edgecolors=colormap[colorIndx])
    plt.xlabel('Fitness cycle '+str(2*t1+1))
    plt.ylabel('Fitness cycle '+str(2*t2+1))
    plt.title('P'+str(Pop)+"  NLs="+str(np.size(list))+" WM "+str(WeightMean))#+ ", weighted mean "+str(WeightMean))
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_Cycle_"+str(2*t1+1)+"-"+str(2*t2+1)+".png")
    return WeightMean
    #return Med
def plot_new_mean_fitness(Pop,D_fitness,t_mid): 
    global OutputFilePath
    mean_fitness = read_MeanFitness(Pop)
    x = [2*t+1 for t in range(np.size(mean_fitness))]
    new_mean_fitness = mean_fitness - D_fitness
    print(new_mean_fitness)
    plt.figure()
    plt.plot(x,mean_fitness,color='black',marker='o',ls='solid',label='old mean fitness')
    plt.plot(x[0:t_mid+1],new_mean_fitness[0:t_mid+1],color='red',marker='o',ls='solid',label='new mean fitness')
    plt.xlabel("Time (cycle)")
    plt.title("P"+str(Pop)+"  Mean Fitness (1/cycle)")
    plt.legend(loc='best')
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_MeanFitness_Cycle_"+str(2*t_mid+1)+".png")
    
    
##### Plot Mean Fitness


T_ini = 3

D_mean_fitness = [0 for i in range(TotalTimePoint-1)]

BarcodeCountsArray = readBarcodeCount(Pop)



#Adaptive_NLs = read_Adpative_LNs(Pop)
#listsameADP = list(set(listADP)& set(Adaptive_NLs))
#BarcodeCountsArray = readBarcodeCount(Pop)
#AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(0,len(BarcodeCountsArray))]

BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)

#listADP = [0 for i in range(T_ini,TotalTimePoint-1)]
#listADP[0] = find_Adaptives(Pop,T_ini,D_mean_fitness,BarcodeFreqArray)
listlargeLNs = find_largeLNs(Pop,T_ini,BarcodeFreqArray)
#listADP_all = listADP[0]
for t in range(T_ini+1,TotalTimePoint-1):
    #D_mean_fitness[t] = plot_S1S2(Pop,listADP[t-1-T_ini],t-1,t,D_mean_fitness,BarcodeFreqArray)
    D_mean_fitness[t] = plot_S1S2(Pop,listlargeLNs,t-1,t,D_mean_fitness,BarcodeFreqArray)
    plot_new_mean_fitness(Pop,D_mean_fitness,t)
    #print(D_mean_fitness)
    listlargeLNs = find_largeLNs(Pop,t,BarcodeFreqArray)
    #listADP[t-T_ini] = find_Adaptives(Pop,t,D_mean_fitness,BarcodeFreqArray)
    #listADP_all.extend(listADP[t-T_ini])
    #listADP_all = list(set(listADP_all))


#print(np.size(listADP_all))

np.save(OutputFilePath+"P"+str(Pop)+"_Dx.npy", D_mean_fitness)

#BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)
'''
D_fitness = np.load(OutputFilePath+"P"+str(Pop)+"_Dx.npy")

old_fitness = read_Fitness(Pop)
new_fitness = [old_fitness[t] - D_fitness[t] for t in range(np.size(D_fitness))]

new_mean_fitness = read_MeanFitness(Pop) - D_fitness
print(new_mean_fitness)
cal_mean_fitness = [sum([(f1+f2)*s/2 for f1,f2,s in zip(BarcodeFreqArray[t],BarcodeFreqArray[t+1],new_fitness[t])]) for t in range(TotalTimePoint-1)]
#cal_mean_fitness = [sum([f1*s for f1,s in zip(BarcodeFreqArray[1+t],new_fitness[t])]) for t in range(TotalTimePoint-1)]
print(cal_mean_fitness)


plt.figure()
x = [2*t+1 for t in range(np.size(new_mean_fitness))]
plt.plot(x,new_mean_fitness,color='red',marker='o',ls='solid',label='new mean fitness')
plt.plot(x,cal_mean_fitness,color='goldenrod',marker='o',ls='solid',label='mean fitness direct calculation')
plt.legend(loc='best')
plt.xlabel("Time (cycle)")
plt.title("P"+str(Pop) + "  Mean Fitness (1/cycle)")
plt.savefig(OutputFilePath+"P"+str(Pop)+"_MeanFitness_check.png")
'''
'''
a = np.load(OutputFilePath+"P"+str(Pop)+"_newADPs_dS_2.npz")

D_fitness = a['Dx']
listADP = a['ADP']

listADP_all = []
for idx in range(listADP.shape[0]):
    listADP_all.extend(listADP[idx])

listADP_all = list(set(listADP_all))

old_fitness = read_Fitness(Pop)
new_fitness = [old_fitness[t] - D_fitness[t] for t in range(np.size(D_fitness))]

ADP_maxS = []
ADP_aveS = []
for bc in listADP_all:
    max_s = 0
    ave_s = 0
    normalized = 0
    for idx in range(listADP.shape[0]):
        if bc in listADP[idx]:
            s = new_fitness[idx+T_ini][bc]
            max_s = max(max_s,s)
            ave_s = ave_s + s
            normalized = normalized +1
    ave_s = ave_s/normalized
    ADP_maxS.append(max_s)
    ADP_aveS.append(ave_s)

plt.figure()
binwidth = 0.05
bins = np.arange(min(ADP_maxS), max(ADP_maxS) + binwidth, binwidth)
plt.hist(ADP_maxS,bins)
plt.title("NLs="+str(np.size(listADP_all)))
plt.savefig(OutputFilePath+"P"+str(Pop)+"DFE_ADP_max.png")

plt.figure()
binwidth = 0.05
bins = np.arange(min(ADP_aveS), max(ADP_aveS) + binwidth, binwidth)
plt.hist(ADP_aveS,bins)
plt.title("NLs="+str(np.size(listADP_all)))
plt.savefig(OutputFilePath+"P"+str(Pop)+"DFE_ADP_ave.png")

plt.figure()
binwidth = 0.05
ADP_T13 = new_fitness[6][listADP[3]]
bins = np.arange(min(ADP_T13), max(ADP_T13) + binwidth, binwidth)
plt.hist(ADP_T13,bins)
plt.title("NLs="+str(np.size(ADP_T13)))
plt.savefig(OutputFilePath+"P"+str(Pop)+"DFE_ADP_Cycle13.png")

plt.figure()
binwidth = 0.05
ADP_T11 = new_fitness[5][listADP[2]]
bins = np.arange(min(ADP_T11), max(ADP_T11) + binwidth, binwidth)
plt.hist(ADP_T11,bins)
plt.title("NLs="+str(np.size(ADP_T11)))
plt.savefig(OutputFilePath+"P"+str(Pop)+"DFE_ADP_Cycle11.png")
'''

'''
t_mid = t_mid + 1
listADP = find_Adaptives(Pop,t_mid,D_mean_fitness,BarcodeFreqArray)
D_mean_fitness[t_mid+1] = plot_S1S2(Pop,listADP,t_mid,t_mid+1,D_mean_fitness,BarcodeFreqArray)
plot_new_mean_fitness(Pop,D_mean_fitness,t_mid)
print(D_mean_fitness)
#print(np.size(listADP),np.size(Adaptive_NLs),np.size(listsameADP))
'''

'''
plt.figure()
plt.xlabel('Time (cycle)')
plt.ylabel('Log10 BC frequency')
plt.title("P"+str(Pop)+"  Grey: old-ADP, Blue: new-ADP, Red: both-ADP ("+str(np.size(listsameADP))+")")
x = [2*t for t in range(t_mid-2,t_mid+2)]
for bc in Adaptive_NLs:
    #count = [BarcodeCountsArray[t][bc] for t in range(t_mid-2,t_mid+2)]
    y = [np.log10(BarcodeCountsArray[t][bc]/AllCountsArray[t]+(10**-8)*(BarcodeCountsArray[t][bc]==0)) for t in range(t_mid-2,t_mid+2)]
    plt.plot(x,y,color='grey',linewidth=0.1,alpha=1)
for bc in listADP:
    #count = [BarcodeCountsArray[t][bc] for t in range(t_mid-2,t_mid+2)]
    y = [np.log10(BarcodeCountsArray[t][bc]/AllCountsArray[t]+(10**-8)*(BarcodeCountsArray[t][bc]==0)) for t in range(t_mid-2,t_mid+2)]
    plt.plot(x,y,color='blue',linewidth=0.2,alpha=1)
for bc in listsameADP:
    #count = [BarcodeCountsArray[t][bc] for t in range(t_mid-2,t_mid+2)]
    y = [np.log10(BarcodeCountsArray[t][bc]/AllCountsArray[t]+(10**-8)*(BarcodeCountsArray[t][bc]==0)) for t in range(t_mid-2,t_mid+2)]
    plt.plot(x,y,color='red',linewidth=0.2,alpha=1)
plt.savefig(OutputFilePath+"P"+str(Pop)+"_ADP_LNs_Cycle"+str(2*t_mid+1)+".png")
'''


#DFE_shift(Pop,t_mid)
#Adaptive_NLs = read_Adpative_LNs(Pop)

#for Pop in range(1,11):
#    DFE_shift(Pop,t_mid)

'''
s1 = read_Fitness(Pop)[0:time_end+1]

s1_mean = np.mean(s1, axis=0)
s1_tend = s1[time_end]
s1_SEM = np.std(s1, axis=0)/(s1.shape[0])**0.5  # standard error of mean
s1_CV = np.abs(np.asarray([sem/mean*(s1.shape[0])**0.5 for sem,mean in zip(s1_SEM, s1_mean)]))
s1_max = np.max(s1[1:], axis=0)
s1_sort = np.sort(s1[1:], axis=0)
s1_sort = s1_sort[::-1] # reverse (decending order)
s1_max2 = np.mean(s1_sort[0:2], axis=0)

listCVsmall = np.nonzero(s1_CV<1)[0]
listPositiveS = np.nonzero(s1_mean>0)[0]

BarcodeCountsArray = readBarcodeCount(Pop)
listNonzeros = np.nonzero(np.min(BarcodeCountsArray,axis=0)>0)[0]
#ListCleanAdaptive = list(set(listCVsmall)&set(listPositiveS)&set(Adaptive_NLs))
ListCleanAdaptive = np.asarray(list(set(Adaptive_NLs)&set(listNonzeros)))#list(set(listNonzeros)&set(Adaptive_NLs))
print(np.size(ListCleanAdaptive),np.size(Adaptive_NLs))
#plot_CleanAdaptive_BCFeq(Pop,time_end,Adaptive_NLs,ListCleanAdaptive)
print(ListCleanAdaptive[0:10])
BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)
print(BarcodeFreqArray.shape)
s1 = read_Fitness(Pop)
for t in range(TotalTimePoint-2):
    list1 = np.nonzero(BarcodeCountsArray[t]>10)[0]
    list2 = np.nonzero(BarcodeCountsArray[t+1]>10)[0]
    ListCleanAdaptive = list(set(Adaptive_NLs)&set(listNonzeros))
    x = s1[t][ListCleanAdaptive]
    y = s1[t+1][ListCleanAdaptive]
    freq = BarcodeFreqArray[t][ListCleanAdaptive]
    colormap = plt.cm.jet(np.linspace(0,1,101)) 
    colmax = -2
    colmin = -6
    colbin = (colmax-colmin)/100
    line_max = max(np.max(x),np.max(y))
    line_min = min(np.min(x),np.min(y))
    line = np.arange(line_min, line_max, 0.01)
    plt.figure()
    plt.plot(line,line, 'k--')
    for bc in range(np.size(ListCleanAdaptive)):
        s = np.log10(freq[bc])
        if s>=colmax:
            s= colmax
        elif s< colmin:
            s = colmin
        colorIndx = (int)((s-colmin)/colbin)
        plt.scatter(x[bc],y[bc], s=50, facecolors='none', edgecolors=colormap[colorIndx])
    plt.title("Pop "+str(Pop) + ",  NLs="+str(np.size(ListCleanAdaptive)))
    plt.xlabel("s  cycle " + str(2*t+1))
    plt.ylabel("s  cycle " + str(2*t+3))
    plt.savefig(OutputFilePath+"CleanAdaptive_Pop"+str(Pop)+"_s"+str(t)+"_s"+str(t+1)+".png")
    plt.close('all')


'''
'''
binwidth = 0.1

finess = read_Fitness(Pop)

for t in range(TotalTimePoint-2):
    plt.figure()
    data = finess[t][ListCleanAdaptive]
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    plt.hist(data,bins,label="t"+str(t),color= 'blue', alpha=0.3)
    data = finess[t+1][ListCleanAdaptive]
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    plt.hist(data,bins,label="t"+str(t+1),color= 'green', alpha=0.3)
    plt.legend(loc='best')
    plt.savefig(OutputFilePath+"CleanAdaptive_Dist_Pop"+str(Pop)+"_T"+str(t)+".png")

Ds = []
Var_Ds = []
binwidth = 0.1

for t in range(TotalTimePoint-2):
    data = finess[t][ListCleanAdaptive]
    data2 = finess[t+1][ListCleanAdaptive]
    ds = data2 - data
    plt.figure()
    bins = np.arange(min(ds), max(ds) + binwidth, binwidth)
    plt.hist(ds,bins,label="t"+str(t),color= 'black', alpha=0.3)
    plt.xlabel("ds")
    Ds.append(np.median(ds))
    Var_Ds.append(np.var(ds))
    plt.title("Mean "+str(np.mean(ds))+" Median "+str(np.median(ds))+" Var "+str(np.var(ds)))
    plt.savefig(OutputFilePath+"CleanAdaptive_Dist_ds_Pop"+str(Pop)+"_T"+str(t)+".png")
print("Ds",Ds)
plt.close('all')
#dx=[]
for t in range(TotalTimePoint-2):
    plt.figure()
    data = finess[t][ListCleanAdaptive]
    #mean_s1 = np.mean(data)
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    plt.hist(data,bins,label="t"+str(t),color= 'blue', alpha=0.3)
    data2 = finess[t+1][ListCleanAdaptive]
    #mean_s2 = np.mean(data2)
    #dx.append(mean_s2 - mean_s1)
    #data = data2 - dx[t]
    data = data2 - Ds[t]
    bins = np.arange(min(data), max(data) + binwidth, binwidth)
    plt.hist(data,bins,label="t"+str(t+1),color= 'green', alpha=0.3)
    plt.legend(loc='best')
    plt.title("median(s(t+1))-median(s(t))="+str(Ds[t]))
    plt.savefig(OutputFilePath+"CleanAdaptive_Dist_Alignment_Pop"+str(Pop)+"_T"+str(t)+".png")
#print("dx ",dx)
plt.close('all')
t0=3
#sum_dx = [np.sum(dx[t0:t]) for t in range(t0,np.size(dx)+1)]
Ds_new = [0,0]
for t in range(0,np.size(Ds)):
    if Var_Ds[t] < 0.5:
        Ds_new.append(Ds[t])
    else:
        Ds_new.append(0)

sum_Ds = [np.sum(Ds[t0:t]) for t in range(t0,np.size(Ds)+1)]
print("sum Ds",sum_Ds)


mean_fitness = read_MeanFitness(Pop)
print("Mean Fitness",mean_fitness)
Alignment_mean_fitness = [m for m in mean_fitness]
for t in range(t0,TotalTimePoint-1):
    #Alignment_mean_fitness[t] = Alignment_mean_fitness[t]+sum_dx[t-t0]
    Alignment_mean_fitness[t] = Alignment_mean_fitness[t]-sum_Ds[t-t0]
print("Align Mean Fitness",Alignment_mean_fitness)
np.save(OutputFilePath+"Pop"+str(Pop)+"_Alignming_MeanFitness.npy",Alignment_mean_fitness)
x = np.asarray([1+2*i for i in range(np.size(mean_fitness))])
print(x)
plt.plot(x,Alignment_mean_fitness,color='blue',marker='o',ls='solid',label='Alignming x')
#mean_fitness = read_MeanFitness(Pop)
plt.plot(x,mean_fitness,color='black',marker='o',ls='solid',label='x')
plt.legend(loc='best')
plt.xlabel('Time (cycle)')
plt.ylabel('Mean Fitness (1/cycle)')
plt.savefig(OutputFilePath+"AlignmingMeanFitness_Pop"+str(Pop)+".png")

'''
'''
plt.figure()
plt.plot(s1_mean,s1_SEM,color = bluish_green, marker='.', ls='None')
plt.xlabel("Mean (fintess) over time")
plt.ylabel("Standard Error of Mean SEM (fitness)")
plt.title("Population "+str(Pop)+", NLs="+str(np.size(Adaptive_NLs)))
plt.savefig(OutputFilePath+"AdaptiveFitness_SEM2Mean_Pop"+str(Pop)+".png")
'''
'''
plt.figure()
plt.plot(s1_mean[Adaptive_NLs],s1_CV[Adaptive_NLs],color = bluish_green, marker='.', ls='None')
dash_x = np.arange(0,2,0.1)
dash_y = np.arange(0,5,0.1)
plt.plot(dash_x,[2 for i in range(np.size(dash_x))],'k--')
plt.plot([0 for i in range(np.size(dash_y))],dash_y,'k--')
plt.xlabel("Mean (fintess) over time")
plt.ylabel("Coefficient of Variation CV (fitness)")
plt.title("Population "+str(Pop)+", NLs="+str(np.size(Adaptive_NLs)))
plt.xlim(-0.5,1.5)
plt.ylim(0,5)
plt.savefig(OutputFilePath+"AdaptiveFitness_CV2Mean_Pop"+str(Pop)+".png")
'''
'''
plt.figure()
binwidth = 0.1
data = s1_mean[ListCleanAdaptive]
bins = np.arange(min(data), max(data) + binwidth, binwidth)
plt.hist(data,bins, label="T_ave S",color='blue', alpha = 0.3)
data = s1_tend[ListCleanAdaptive]
bins = np.arange(min(data), max(data) + binwidth, binwidth)
plt.hist(data,bins,label="cycle7 S",color='red', alpha = 0.3)
data = s1_max2[ListCleanAdaptive]
bins = np.arange(min(data), max(data) + binwidth, binwidth)
plt.hist(data,bins,label="max2 S",color='black', alpha = 0.3)
plt.legend(loc='best')
plt.xlabel("Fitness (1/cycle)")
plt.title("Population "+str(Pop)+" NLs="+str(np.size(ListCleanAdaptive)))
plt.savefig(OutputFilePath+"CleanAdaptive_DFE_Pop"+str(Pop)+"_compare3_max2"+".png")
'''



'''
Pop = 1
y1 = read_MeanFitness(Pop)
Pop = 6
y2 = read_MeanFitness(Pop)

x = np.asarray([1+2*i for i in range(np.size(y1))])
plt.figure()
plt.xlim(0,np.max(x)+1)
plt.ylim(-0.05,0.8)
plt.plot(x,y1,color='black',marker='o',ls='solid',label='Mono-culture')
plt.plot(x,y2,color=bluish_green,marker = 'o',ls= 'solid',label='Co-culture')

for Pop in range(2,6):
    y = read_MeanFitness(Pop)
    plt.plot(x,y,color='black',marker='o',ls='solid')
for Pop in range(7,11):
    y = read_MeanFitness(Pop)
    plt.plot(x,y,color=bluish_green,marker='o',ls='solid')

plt.xlabel('Time (cycle)',size=15)
plt.legend(loc='best',fontsize=15)
plt.title('Mean Fitness (1/cycle)',size=20)
plt.savefig(OutputFilePath+"MeanFitness_1&8.png")
'''
'''
fig, ax = plt.subplots()
width = 0.35       # the width of the bars
barX = [0,1]
barY = [y1[-1], y2[-1]]

ax.bar(barX[0],barY[0],width, color="black")
ax.bar(barX[1],barY[1],width, color=bluish_green)
ax.set_xticks([idx+width/2 for idx in barX])
ax.set_xticklabels(['Mono-culture', 'Co-culture'],fontsize=20)
ax.set_xlim(-0.3,1.7)
ax.set_title("Cycle-9 Mean Fintess (1/cycle)",size=20)
plt.savefig(OutputFilePath+"MeanFitness_Cycle9_1&6.png")
'''



