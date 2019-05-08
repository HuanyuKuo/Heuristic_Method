import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

###########################################################################
## Global variables
###########################################################################
Pop="7"
TotalTimePoint = 9
GENperTimePoint = 20
#BottleNeckSize = 5*(10**6)
IterationMax = 5
#BarcodeCountPath = "/mnt/c/Users/Ruby/Desktop/AllPopReadCounts/SourceFile/"
#OutputFilePath = "/mnt/c/Users/Ruby/Desktop/AllPopReadCounts/Pop"+Pop+"/"
BarcodeCountPath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/SourceFile/"
OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/Pop"+Pop+"/"
###########################################################################
## Functions
###########################################################################
def readBarcodeCount():
    global Pop, TotalTimePoint, BarcodeCountPath
    FileName = "Pop"+Pop+"BowtieReadCounts"
    FilePathName = BarcodeCountPath +FileName+".txt"
    file = open(FilePathName,"r")
    triplets=file.read().split()
    A=np.array(triplets, dtype=np.int32)
    file.close()
    return np.transpose(np.reshape(A,(-1,TotalTimePoint)))

def guessNeutrals(BarcodeCountsArray, AllCountsArray, ListNonZeroBarcodeIndex):
    global TotalTimePoint
    CutLogY = -5
    CutX = 4
    ListNeutralBarcodeIndex = []
    ListNonNeutralBarcodeIndex =[]
    NeutralCounts = [0 for i in range(TotalTimePoint)]
    NonNeutralCounts = [0 for i in range(TotalTimePoint)]
    for bc in ListNonZeroBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        y = np.log10([count[t]/AllCountsArray[t]+(10**-8)*(count[t]==0) for t in range(TotalTimePoint)])
        s = y[CutX]
        if s < CutLogY: #cutoff
            NeutralCounts = [aa+bb for aa, bb in zip(count,NeutralCounts)]
            ListNeutralBarcodeIndex.append(bc) 
        else:
            NonNeutralCounts = [aa+bb for aa, bb in zip(count,NonNeutralCounts)]
            ListNonNeutralBarcodeIndex .append(bc)
    return (NeutralCounts, NonNeutralCounts, ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex)

def iterationNeutrals(cutCV,cutMean, Abs_Coefficient_of_Variation,Mean_Relative_Fitness, BarcodeCountsArray, ListBarcodeIndex):
    global TotalTimePoint, TotalNumberNonzeroBarcodes
    ListNeutralBarcodeIndex=[]
    ListNonNeutralBarcodeIndex = []
    NeutralCounts = [0 for i in range(TotalTimePoint)]
    NonNeutralCounts = [0 for i in range(TotalTimePoint)]
    for bc in ListBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        if (Abs_Coefficient_of_Variation[bc]>cutCV or Mean_Relative_Fitness[bc]<cutMean):
            NeutralCounts = [aa+bb for aa, bb in zip(count,NeutralCounts)]
            ListNeutralBarcodeIndex.append(bc)
        else:
            NonNeutralCounts = [aa+bb for aa, bb in zip(count,NonNeutralCounts)]
            ListNonNeutralBarcodeIndex.append(bc)
    return (NeutralCounts, NonNeutralCounts, ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex)

def RecursionNeutralList(ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex, NeutralCounts, NonNeutralCounts, BarcodeCountsArray):
    newListNeutralBarcodeIndex = []
    newNeutralCounts = [0 for i in range(TotalTimePoint)]
    threshold = 0.001
    check = False
    for bc in ListNeutralBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        maxFreq = np.max([count[t]/NeutralCounts[t] for t in range(TotalTimePoint)])
        if (maxFreq < threshold):
            newListNeutralBarcodeIndex.append(bc)
            newNeutralCounts = [aa+bb for aa, bb in zip(count,newNeutralCounts)]
        else:
            NonNeutralCounts = [aa+bb for aa, bb in zip(count,NonNeutralCounts)]
            ListNonNeutralBarcodeIndex.append(bc)
            check = True
    return (check, newNeutralCounts, NonNeutralCounts, newListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex)
def getTimeAverageFitness(Relative_Fitness, BoolTrustedBarcodeIndexTime, ListBarcodeIndex):
    TimeAverageFitness = [0 for bc in range(TotalNumberBarcodes)]
    TimeCVFitness = [0 for bc in range(TotalNumberBarcodes)]
    for bc in ListBarcodeIndex:
        average = np.average(Relative_Fitness[bc],weights= BoolTrustedBarcodeIndexTime[bc])
        variance = np.average((Relative_Fitness[bc]-average)**2, weights=BoolTrustedBarcodeIndexTime[bc])
        CoefficientVariation = math.sqrt(variance)/average
        TimeAverageFitness[bc] = average
        TimeCVFitness[bc] = math.fabs(CoefficientVariation)
    return (TimeAverageFitness, TimeCVFitness)

def writeListToFile(writefilename, my_list):
    with open(writefilename, 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)
    f.close()

def plotBarcodeFreq(BarcodeCountsArray,AllCountsArray,ListBarcodeIndex): # plotCleanBarcodeFreq() 
    global Pop, TotalTimePoint, OutputFilePath, TotalNumberNonzeroBarcodes
    plt.figure(figsize=(8.5,5.5))
    colormap = plt.cm.rainbow(np.linspace(0,1,101)) 
    colmax = -3#np.log10(0.001)#np.max(BarcodeFreqArray[ENDTIMEPOINT]))
    colmin = -5#np.log10(0.00001)#np.min(BarcodeFreqArray[ENDTIMEPOINT]))
    colbin = (colmax-colmin)/100
    plt.xlabel('Time (point)')
    plt.ylabel('log10 Barcode Frequency')
    numLineage = 0
    totalCountsTEND = 0;
    x = [2*t+1 for t in range(TotalTimePoint)]
    for bc in ListBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        y = [np.log10(count[t]/AllCountsArray[t]+(10**-8)*(count[t]==0)) for t in range(TotalTimePoint)]
        s = y[4]
        if s>=colmax:
            colorIndx = 100
        elif s<colmin:
            colorIndx = 0
        else:
            colorIndx = (int)((s-colmin)/colbin)
        plt.plot(x,y,color=colormap[colorIndx], linewidth=0.3)
    plt.title("P" + Pop +"  NL="+str(np.size(ListBarcodeIndex)))
    plt.savefig(OutputFilePath+"BarcodeFreq_P"+Pop+'.png',dpi = 800)
    plt.close('all')

def plotRelativeFreq(BarcodeCountsArray,NeutralCounts,ListNonNeutralBarcodeIndex, ListNeutralBarcodeIndex,colmax,colmin,Iteration):
    global Pop, OutputFilePath, TotalTimePoint
    plt.figure(figsize=(8.5,5.5))
    plt.xlabel('Time (point)')
    plt.title('P'+Pop+'  Relative Frequency   It'+str(Iteration)+'')
    colormap = plt.cm.rainbow(np.linspace(0,1,101)) 
    colbin = (colmax-colmin)/100
    x = [2*t+1 for t in range(TotalTimePoint)]
    for bc in ListNeutralBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        y = [np.log10(count[t]/NeutralCounts[t]+(10**-8)*(count[t]==0)) for t in range(TotalTimePoint)]
        plt.plot(x,y,color='grey', linewidth=0.1)
    plt.savefig(OutputFilePath+'Iteration'+str(Iteration)+'Neutral_Relative_Frequency_P'+str(Pop)+'.png',dpi = 800)
    for bc in ListNonNeutralBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        y = [np.log10(count[t]/NeutralCounts[t]+(10**-8)*(count[t]==0)) for t in range(TotalTimePoint)]
        s = y[-1]
        if s>=colmax:
            colorIndx = 100
        elif s<colmin:
            colorIndx = 0
        else:
            colorIndx = (int)((s-colmin)/colbin)
        plt.plot(x,y,color=colormap[colorIndx], linewidth=0.1)
    plt.savefig(OutputFilePath+'Iteration'+str(Iteration)+'_Relative_Frequency_P'+str(Pop)+'.png',dpi = 800)
    plt.close('all')

def plotNeutralGroups(NeutralCounts,NonNeutralCounts, AllCountsArray,Iteration):
    global Pop, OutputFilePath, TotalTimePoint
    NeutralFreq = [NeutralCounts[t]/AllCountsArray[t] for t in range(TotalTimePoint)]
    NonNeutralFreq = [NonNeutralCounts[t]/AllCountsArray[t] for t in range(TotalTimePoint)]
    x = [2*t+1 for t in range(TotalTimePoint)]
    fig, ax1 = plt.subplots(figsize=(6.5,5))
    ax1.set_xlabel('Time (point)')
    ax1.set_ylabel('Neutral group')
    ax1.plot(x, NeutralFreq,'ko')
    ax1.tick_params(axis='y', labelcolor='k')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Others', color='red')  # we already handled the x-label with ax1
    #y = [1-f for f in NeutralFreq]
    ax2.plot(x, NonNeutralFreq, 'ro')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("P"+Pop+"  Barcode Frequency of Subgroup (It "+str(Iteration)+")")
    plt.savefig(OutputFilePath+'Iteration'+str(Iteration)+'_Neutral_Frequency_P'+Pop+'.png',dpi = 400)
    fig, ax1 = plt.subplots(figsize=(6.5,5))
    ax1.set_xlabel('Time (point)')
    ax1.set_ylabel('Neutral group')
    ax1.plot(x, np.log10(NeutralFreq),'ko')
    ax1.tick_params(axis='y', labelcolor='k')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Others', color='red')  # we already handled the x-label with ax1
    #y = [1-f for f in NeutralFreq]
    ax2.plot(x, np.log10(NonNeutralFreq), 'ro')
    ax2.tick_params(axis='y', labelcolor='red')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title("P"+Pop+"  Barcode Frequency of Subgroup (It "+str(Iteration)+")")
    plt.savefig(OutputFilePath+'Iteration'+str(Iteration)+'_Neutral_Log10Frequency_P'+Pop+'.png',dpi = 400)
    plt.close('all')

def plotMeanFitness(REF_Fitness,Iteration):
    global Pop, OutputFilePath, TotalTimePoint
    Negative_REF_Fitness = [-1*r for r in REF_Fitness]
    plt.figure(figsize=(6.5,5))
    plt.xlabel('Time (cycle)')
    x = [(t+0.5)*2 for t in range(TotalTimePoint-1)]
    #plt.plot(x,Mean_Fitness[1:ENDTIMEPOINT],'bo-',label='Average over lineages')
    plt.plot(x,Negative_REF_Fitness,'ko-',label = '-1* Reference Fitness')
    plt.title("Mean Fitness (1/cycle)   It "+str(Iteration))
    #plt.legend(loc="best")
    plt.savefig(OutputFilePath+'Iteration'+str(Iteration)+'_Mean_Fitness_P'+str(Pop)+'.png',dpi = 400)
    plt.close('all')

def plotCV2Mean(Mean_Relative_Fitness, Abs_Coefficient_of_Variation,ListNonNeutralBarcodeIndex, ListNeutralBarcodeIndex, ListAdaptiveBarcodeIndex, Iteration, cutCV, cutMean):
    global OutputFilePath, Pop
    x1 = [Mean_Relative_Fitness[bc] for bc in ListNeutralBarcodeIndex]
    y1 = [Abs_Coefficient_of_Variation[bc] for bc in ListNeutralBarcodeIndex]
    x2 = [Mean_Relative_Fitness[bc] for bc in ListNonNeutralBarcodeIndex]
    y2 = [Abs_Coefficient_of_Variation[bc] for bc in ListNonNeutralBarcodeIndex]
    x3 = [Mean_Relative_Fitness[bc] for bc in ListAdaptiveBarcodeIndex]
    y3 = [Abs_Coefficient_of_Variation[bc] for bc in ListAdaptiveBarcodeIndex]
    plt.figure(figsize = (6.5,5))
    plt.xlabel('Mean of relative fitness')
    plt.ylim(0,5)
    plt.plot(x2,y2,linestyle='None',color = 'blue',marker='.',markersize=3)
    plt.plot(x1,y1,linestyle='None',color = 'grey',marker='.',markersize=3)
    plt.plot(x3,y3,linestyle='None',color = 'red',marker='.',markersize=3)
    xCVline = [i*np.max(x2)/100 for i in range(100)]
    yCVline = [cutCV for i in range(100)]
    xMeanline = [cutMean for i in range(100)]
    yMeanline = [i*5/100 for i in range(100)]
    plt.plot(xCVline,yCVline,'k',linestyle='--')
    plt.plot(xMeanline,yMeanline,'k',linestyle='--')
    plt.title("ABS(CV) Coefficient of Variation")
    plt.savefig(OutputFilePath+'Iteration'+str(Iteration)+'_CV2Mean_P'+str(Pop)+'.png',dpi = 400)
    plt.close('all')

def plotCheckAdaptive(BarcodeCountsArray,NeutralCounts, AllCountsArray, ListAdaptiveBarcodeIndex, ListNonNeutralBarcodeIndex,Iteration):
    plt.figure(1,figsize=(8.5,5.5))
    plt.xlabel('Time (point)')
    plt.ylabel('log10 Barcode Frequency')
    plt.figure(2,figsize=(8.5,5.5))
    plt.xlabel('Time (point)')
    plt.ylabel('log10 Relative Barcode Frequency')
    x = [2*t+1 for t in range(TotalTimePoint)]
    for bc in ListNonNeutralBarcodeIndex:
        count = [BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)]
        y = [np.log10(count[t]/AllCountsArray[t]+(10**-8)*(count[t]==0)) for t in range(TotalTimePoint)]
        y2 = [np.log10(count[t]/NeutralCounts[t]+(10**-8)*(count[t]==0)) for t in range(TotalTimePoint)]
        if bc in ListAdaptiveBarcodeIndex:
            col = 'red'
        else:
            col = 'blue'
        plt.figure(1)
        plt.plot(x,y,color=col, linewidth=0.1)
        plt.figure(2)
        plt.plot(x,y2,color=col, linewidth=0.1)
    plt.figure(1)
    plt.title("P" + Pop +"  Iteration "+str(Iteration)+" NL(Adpative)="+str(np.size(ListAdaptiveBarcodeIndex)))
    plt.savefig(OutputFilePath+"Iteratoin"+str(Iteration)+"_NonNeural_BarcodeFreq_P"+Pop+'.png',dpi = 800)
    plt.figure(2)
    plt.title("P" + Pop +"  Iteration "+str(Iteration)+" NL(Adpative)="+str(np.size(ListAdaptiveBarcodeIndex)))
    plt.savefig(OutputFilePath+"Iteratoin"+str(Iteration)+"_NonNeural_Relative_BarcodeFreq_P"+Pop+'.png',dpi = 800)
    plt.close('all')
###########################################################################
# Variables
###########################################################################
ListNonZeroBarcodeIndex = []
ListExtinctBarcodeIndex = []
ListNeutralBarcodeIndex = []
ListNonNeutralBarcodeIndex =[]
ListTrustedBarcodeIndex = []

###########################################################################
##  Main     
#####################################################################

BarcodeCountsArray = readBarcodeCount()

# Total Read and ListBarcodeIndex
TotalNumberBarcodes = np.shape(BarcodeCountsArray)[1]
TotalNumberNonzeroBarcodes = np.sum(np.sum(BarcodeCountsArray,axis=0)>0)
ListNonZeroBarcodeIndex = np.nonzero(np.sum(BarcodeCountsArray,axis=0)>0)[0]
ListNonExtinctBarcodeIndex = np.nonzero(BarcodeCountsArray[-1]!=0)[0]


Relative_Fitness = [[0 for t in range(TotalTimePoint-1)] for i in range(TotalNumberBarcodes)]
BoolTrustedBarcodeIndexTime = [[0 for t in range(TotalTimePoint-1)] for i in range(TotalNumberBarcodes)]

AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(TotalTimePoint)]
print(AllCountsArray)

#BarcodeFreqArray =  [[(count+1)/(AllCountsArray[t]+TotalNumberNonzeroBarcodes) for count in BarcodeCountsArray[t]] for t in range(TotalTimePoint)]
#plotBarcodeFreq(BarcodeCountsArray,AllCountsArray,ListNonZeroBarcodeIndex) # This step takes time since the # lineages is huge

NeutralCounts, NonNeutralCounts, ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex = guessNeutrals(BarcodeCountsArray, AllCountsArray, ListNonZeroBarcodeIndex)

for Iteration in range(0,1+IterationMax):
    print("Iteration"+str(Iteration))
    plotNeutralGroups(NeutralCounts,NonNeutralCounts,AllCountsArray,Iteration)
    colmax = 0
    colmin = -4
    REF_Fitness = [np.log(NeutralCounts[t+1]/AllCountsArray[t+1]/NeutralCounts[t]*AllCountsArray[t])/2 for t in range(TotalTimePoint-1)]
    plotMeanFitness(REF_Fitness,Iteration)
    print([-1*r for r in REF_Fitness])
    NonZeroBarcodeCountsArray = BarcodeCountsArray+0.4*(BarcodeCountsArray==0)
    count1 = np.matrix.transpose(np.asarray([NonZeroBarcodeCountsArray[t] for t in range(TotalTimePoint-1)]))
    count2 = np.matrix.transpose(np.asarray([NonZeroBarcodeCountsArray[t+1] for t in range(TotalTimePoint-1)]))
    neutral_count1 = [ NeutralCounts[t] for t in range(TotalTimePoint-1)]
    neutral_count2 = [ NeutralCounts[t+1] for t in range(TotalTimePoint-1)]
    for bc in range(TotalNumberBarcodes):
        Relative_Fitness[bc] = (np.log(count2[bc])-np.log(count1[bc])-np.log(neutral_count2)+np.log(neutral_count1))/2
        BoolTrustedBarcodeIndexTime[bc] = [count1[bc][t]!=0.4 or count2[bc][t]!=0.4 for t in range(TotalTimePoint-1)] 
    ListTrustedBarcodeIndex = np.nonzero(np.sum(BarcodeCountsArray>10,axis=0)>4)[0]
    ListAdaptiveBarcodeIndex = np.intersect1d(ListTrustedBarcodeIndex, ListNonExtinctBarcodeIndex)
    ListAdaptiveBarcodeIndex = np.intersect1d(ListAdaptiveBarcodeIndex, ListNonNeutralBarcodeIndex)
    Mean_Relative_Fitness, Abs_Coefficient_of_Variation = getTimeAverageFitness(Relative_Fitness,BoolTrustedBarcodeIndexTime, ListNonZeroBarcodeIndex)
    cutCV = 2
    cutMean = 0.0
    plotCV2Mean(Mean_Relative_Fitness, Abs_Coefficient_of_Variation,ListNonNeutralBarcodeIndex, ListNeutralBarcodeIndex, ListAdaptiveBarcodeIndex, Iteration, cutCV, cutMean)
    if (Iteration !=IterationMax):
        NeutralCounts, NonNeutralCounts, ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex = iterationNeutrals(cutCV,cutMean,Abs_Coefficient_of_Variation,Mean_Relative_Fitness,BarcodeCountsArray, ListNonZeroBarcodeIndex)
        if (Iteration >1):
            check = True
            for i in range(1):
                print('recursion')
                check, NeutralCounts, NonNeutralCounts, ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex = RecursionNeutralList(ListNeutralBarcodeIndex, ListNonNeutralBarcodeIndex, NeutralCounts, NonNeutralCounts, BarcodeCountsArray)
    #else:
    #    plotCheckAdaptive(BarcodeCountsArray,NeutralCounts, AllCountsArray, ListAdaptiveBarcodeIndex, ListNonNeutralBarcodeIndex,Iteration)
'''
meanfitness = [-1*r for r in REF_Fitness]
outputfilehead= OutputFilePath+"P"+Pop+"_It"+str(Iteration)
outputfilename = outputfilehead+"_MeanFitness.txt"
writeListToFile(outputfilename,meanfitness)
outputfilename = outputfilehead+"_ListNeutralBarcodeIndex.txt"
writeListToFile(outputfilename,ListNeutralBarcodeIndex)
outputfilename = outputfilehead+"_ListNonNeutralBarcodeIndex.txt"
writeListToFile(outputfilename,ListNonNeutralBarcodeIndex)
outputfilename = outputfilehead+"_ListAdaptiveBarcodeIndex.txt"
writeListToFile(outputfilename,ListAdaptiveBarcodeIndex)
outputfilename = outputfilehead+"_RelativeFitness.txt"
np.savetxt(outputfilename,Relative_Fitness)
#plotRelativeFreq(BarcodeCountsArray,NeutralCounts,ListNonNeutralBarcodeIndex, ListNeutralBarcodeIndex,colmax,colmin,Iteration) # This step takes time if the # lineages is huge
#plotBarcodeFreq(BarcodeCountsArray,AllCountsArray,ListNeutralBarcodeIndex)
'''
'''
plt.figure()
for bc in ListNeutralBarcodeIndex[0:5000]:
    x = [2*t+1 for t in range(TotalTimePoint)]
    y = [BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)]
    plt.plot(x,np.log10(y), linewidth=0.3)
plt.savefig(OutputFilePath+'Neutral_Freq.png')

plt.figure()
for bc in ListNeutralBarcodeIndex[0:5000]:
    x = [2*t+1 for t in range(TotalTimePoint)]
    y = [BarcodeFreqArray[t][bc]/NeutralFreq[t] for t in range(TotalTimePoint)]
    plt.plot(x,np.log10(y), linewidth=0.3)
plt.savefig(OutputFilePath+'Neutral_Relative_Freq.png')
'''