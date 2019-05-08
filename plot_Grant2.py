import numpy as np
import matplotlib
import math
matplotlib.use("Agg")
from matplotlib import pyplot as plt
#from scipy.optimize import curve_fit
from lmfit import Model, Parameters
from scipy import special
import sys
import decimal
sys.float_info.max

matplotlib.rc('xtick', labelsize=15)     
matplotlib.rc('ytick', labelsize=15)

###########################################################################
## Global variables
###########################################################################
global OutputFilePath, BarcodeCountPath, TotalTimePoint, bluish_green

TotalTimePoint = 9
BarcodeCountPath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/SourceFile/"
#OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/0307/FreqCut3.5CV0.5/"
#OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190407 poster/"
OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/0507_dat_SV/"
bluish_green = "#009E73"


def readBarcodeCount(Pop):
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

def plot_sCV2sMean(Pop,sMean,sCV,listLarge,freq):
    NLs = np.size(listLarge)
    colmax = -2
    colmin = -3
    colbin = (colmax-colmin)/100
    plt.figure()
    colormap = plt.cm.jet(np.linspace(0,1,101)) 
    for bc in listLarge:
        s = np.log10(freq[bc])
        if s>=colmax:
            s= colmax
        elif s< colmin:
            s = colmin
        colorIndx = (int)((s-colmin)/colbin)
        plt.scatter(sMean[bc],sCV[bc],s=10, facecolors='none', edgecolors=colormap[colorIndx])
    plt.ylim(0,5)
    plt.xlim(0,1.5)
    plt.xlabel("Mean of fitness")
    #plt.ylabel("abs CV of fitness")
    plt.ylabel("abs CV of fitness/u")
    plt.title("P"+str(Pop)+" Cycle"+str(5)+"-"+str(9)+" NLs="+str(NLs))
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_s2CV_Cycle"+str(9)+".png")
    plt.close('all')

def plot_DFE(Pop,fitness,listADP):
    dat = fitness
    FilePathName = OutputFilePath+"P"+str(Pop)+"_DFEofADP_dat.txt"
    file = open(FilePathName,"w+")
    file.write("Number of Adpative Lineages\t"+str(np.size(fitness)))
    for i in range(0,np.size(fitness)):
        file.write("\n"+str(int(listADP[i]))+"\t"+str(fitness[i]))
    file.close()
    binwidth = 0.1
    bins = np.arange(min(dat), max(dat) + binwidth, binwidth)
    if Pop <= 5:
        plt.hist(dat,bins,color='grey',normed=True, label="P"+str(Pop)+" (N="+str(np.size(dat))+")" )
    else:
        plt.hist(dat,bins,color=bluish_green,normed=True, label="P"+str(Pop)+" (N="+str(np.size(dat))+")" )
    plt.title("Distribution of Fitness for Adpative Lineages ")
    plt.xlabel("Fitness (1/cycle)")
    #plt.xlabel("Fitness (1/gen)")
    #plt.legend(loc='best')
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_ADP_DFE.png")
    plt.close('all')

def plot_meanfitness(Pop,meanfitness):
    dat = meanfitness
    FilePathName = OutputFilePath+"P"+str(Pop)+"_MeanFitness_dat.txt"
    file = open(FilePathName,"w+")
    for i in range(0,TotalTimePoint-1):
        file.write(str(dat[i])+"\n")
    file.close()
    x = [2*t+1 for t in range(TotalTimePoint-1)]
    if Pop <= 5:
        plt.plot(x,new_mean_fitness,marker='o',ls='solid',color='black')
    else:
        plt.plot(x,new_mean_fitness,marker='o',ls='solid',color=bluish_green)
    plt.title("Mean Fitness")
    plt.xlabel("Fitness (1/cycle)")
    #plt.xlabel("Fitness (1/gen)")
    #plt.legend(loc='best')
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_MeanFitness.png")
    plt.close('all')

def plot_DFE_population(Pop,fitness,w):
    dat = fitness
    binwidth = 0.01
    bins = np.arange(min(dat), max(dat) + binwidth, binwidth)
    if Pop <= 5:
        plt.hist(dat,bins,weights=w, color='grey',normed=True, label="P"+str(Pop)+" (N="+str(np.size(dat))+")" )
    else:
        plt.hist(dat,bins,weights=w, color=bluish_green,normed=True, label="P"+str(Pop)+" (N="+str(np.size(dat))+")" )
    plt.title("Population Distribution of Fitness for Adpative Lineages ")
    #plt.xlabel("Fitness (1/cycle)")
    plt.xlabel("Fitness (1/gen)")
    plt.legend(loc='best')
    plt.savefig(OutputFilePath+"P"+str(Pop)+"_ADP_PDFE.png")
    plt.close('all')

def plot_2DEF(dat1,dat2):
    D = decimal.Decimal
    binwidth = 0.1
    bins1 = np.arange(min(dat1), max(dat1) + binwidth, binwidth)
    bins2 = np.arange(min(dat2), max(dat2) + binwidth, binwidth)
    data1 = np.asarray([D(str(item/np.sum(dat1))) for item in dat1],dtype=float)
    data2 = np.asarray([D(str(item/np.sum(dat2))) for item in dat2])
    plt.figure()
    n1 = plt.hist(dat1,bins1,width=binwidth, color='grey',alpha = 0.6, label = 'Mono-culture (N='+str(np.size(dat1))+')')
    n2 = plt.hist(dat2,bins2,width=binwidth, color=bluish_green, alpha=0.3, label = 'Co-culture (N='+str(np.size(dat2))+')')
    plt.clf()
    n1 = n1[0]/np.sum(n1[0])
    n2 = n2[0]/np.sum(n2[0])
    plt.bar(bins1[:-1],n1, width=binwidth,color='grey',alpha = 0.6, label = 'Mono-culture (N='+str(np.size(dat1))+')')
    plt.bar(bins2[:-1],n2, width=binwidth,color=bluish_green,alpha = 0.3, label = 'Co-culture (N='+str(np.size(dat2))+')')
    plt.legend(loc='best')
    plt.xlabel("Fitness (1/cycle)")
    #plt.xlabel("Fitness (1/gen)")
    plt.title("Distribution of Fitness for Adaptives Lineages")
    #plt.savefig(OutputFilePath+"Compare_ADP_DFE.png")
    plt.savefig(OutputFilePath+"1&8_ADP_DFE_0314.pdf")
    plt.close('all')

def plot_2DEF_population(dat1,dat2,w1,w2):
    binwidth = 0.1
    bins1 = np.arange(min(dat1), max(dat1) + binwidth, binwidth)
    bins2 = np.arange(min(dat2), max(dat2) + binwidth, binwidth)
    
    plt.hist(dat1,bins1,color='grey',alpha = 0.6, weights = w1, label = 'Mono-culture (N='+str(np.size(dat1))+')',normed =True)
    plt.hist(dat2,bins2,color=bluish_green, alpha=0.3,weights = w2, label = 'Co-culture (N='+str(np.size(dat2))+')',normed =True)
    plt.legend(loc='best')
    plt.xlabel("Fitness (1/cycle)")
    #plt.xlabel("Fitness (1/gen)")
    plt.title("Population Distribution of Fitness for Adaptives Lineages")
    #plt.savefig(OutputFilePath+"Compare_ADP_PDFE.png")
    plt.savefig(OutputFilePath+"1&8_ADP_PDFE.pdf")
    plt.close('all')

def output_Adpatives(Pop,listADP,fitness):
    file = open(OutputFilePath+"P"+str(Pop)+"_ADP_Fitness_Cycle9.txt",'w')
    for idx in range(np.size(listADP)):
        file.write(str(listADP[idx])+"\t"+str(fitness[idx])+"\n")

def input_Adaptives(Pop):
    #InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/0306/CutFreq3CV1/"
    InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/0307/FreqCut3.5CV0.5/"
    file = open(InputFilePath+"P"+str(Pop)+"_ADP_Fitness_Cycle9.txt",'r')
    triplets = file.read().split()
    A = np.array(triplets, dtype=np.float32)
    A = np.transpose(np.reshape(A,(-1,2)))
    ADP_list = np.asarray([f for f in A[0]])
    ADP_list.astype(int)
    ADP_fitness = np.asarray([f for f in A[1]])
    return ADP_list, ADP_fitness

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def plotBarcodeFreq(Pop,BarcodeFreqArray,ListBarcodeIndex,colorweight): # plotCleanBarcodeFreq() 
    plt.figure(figsize=(10,5.5))
    # Colormap Setting
    mymap = matplotlib.cm.seismic
    colormap = plt.cm.seismic(np.linspace(0,1,101)) 
    colmax = 1.2#-3#np.log10(0.001)#np.max(BarcodeFreqArray[ENDTIMEPOINT]))
    colmin = 0#-5#np.log10(0.00001)#np.min(BarcodeFreqArray[ENDTIMEPOINT]))
    colbin = (colmax-colmin)/100
    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,1],[0,1]]
    levels = np.arange(colmin, colmax+colbin, colbin)
    CS3 = plt.contourf(Z, levels, cmap=mymap)
    plt.clf() # clear countourf plot
    # Plotting what I actually want
    plt.xlabel('Time (cycle)')
    plt.ylabel('log10 Barcode Frequency')
    numLineage = 0
    totalCountsTEND = 0;
    x = [2*t for t in range(TotalTimePoint)]
    tmp_colorweight = colorweight[ListBarcodeIndex]
    SortListBC = [x for _,x in sorted(zip(tmp_colorweight,ListBarcodeIndex), key=lambda x: x[0])]
    for bc in ListBarcodeIndex:#SortListBC:
        y = np.log10([BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)])
        s = colorweight[bc]
        if s>=colmax:
            colorIndx = 100
        elif s<colmin:
            colorIndx = 0
        else:
            colorIndx = (int)((s-colmin)/colbin)
        plt.plot(x,y,color=colormap[colorIndx], linewidth=0.3)
    for bc in SortListBC[-100:]:
        y = np.log10([BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)])
        s = colorweight[bc]
        if s>=colmax:
            colorIndx = 100
        elif s<colmin:
            colorIndx = 0
        else:
            colorIndx = (int)((s-colmin)/colbin)
        plt.plot(x,y,color=colormap[colorIndx], linewidth=0.3)
    plt.title("P" + str(Pop) +"  NL="+str(np.size(ListBarcodeIndex)))
    clb = plt.colorbar(CS3,fraction=0.046, ticks=list(np.arange(colmin, colmax+(colmax-colmin)/3,(colmax-colmin)/3)))
    clb.set_label('fitness (1/cycle)', labelpad=0, y=-0.03, rotation=0, fontsize = 10)
    clb.ax.tick_params(labelsize=9)
    plt.savefig(OutputFilePath+"BarcodeFreq_P"+str(Pop)+'_v4.pdf',dpi = 800)
    plt.close('all')


def plotBarcodeFreq2(Pop,BarcodeFreqArray,BarcodeCountsArray,ListBarcodeIndex,ListADP,colorweight): # plotCleanBarcodeFreq() 
    plt.figure(figsize=(10,5.5))
    # Colormap Setting
    colormap = plt.cm.YlGnBu(np.linspace(0,1,13)) 
    colmax = 12#-3#np.log10(0.001)#np.max(BarcodeFreqArray[ENDTIMEPOINT]))
    colmin = 0#-5#np.log10(0.00001)#np.min(BarcodeFreqArray[ENDTIMEPOINT]))
    colbin = (colmax-colmin)
    plt.xlabel('Time (cycle)')
    plt.ylabel('log10 Barcode Frequency')
    numLineage = 0
    totalCountsTEND = 0;
    x = [2*t for t in range(TotalTimePoint)]
    for bc in ListBarcodeIndex:#SortListBC:
        y = np.log10([BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)])
        if not bc in ListADP:    
            count = np.asarray([BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)])
            colorIdx = 0
            if count[-1] != 0:
                colorIdx = TotalTimePoint
            else:
                for idx in range(TotalTimePoint-1,1,-1):
                    if count[idx]==0 and count[idx-1]>0:
                        colorIdx = idx
                        break
            plt.plot(x,y,color=colormap[colmax-colorIdx], alpha=0.5,linewidth=0.2)
        '''
        else:
            plt.plot(x,y,color='red', alpha = 1,linewidth=0.3)
        '''
    for bc in ListADP:
        y = np.log10([BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)])
        plt.plot(x,y,color='red', alpha = 0.2,linewidth=0.2)
    plt.title("P" + str(Pop) +"  NL="+str(np.size(ListBarcodeIndex)))
    plt.savefig(OutputFilePath+"BarcodeFreq_P"+str(Pop)+'_v5.5.pdf',dpi = 800)
    plt.close('all')

def plotBarcodeFreq3(Pop,BarcodeFreqArray,BarcodeCountsArray,ListBarcodeIndex,ListADP,colorweight): # plotCleanBarcodeFreq() 
    arr = BarcodeFreqArray[TotalTimePoint-1][ListBarcodeIndex]
    ADP = np.where(arr == np.max(arr))[0]
    NEU = np.where(arr == np.min(arr))[0]
    print(ADP,NEU)
    plt.figure(figsize=(10,5.5))
    # Colormap Setting
    colormap = plt.cm.YlGnBu(np.linspace(0,1,13)) 
    colmax = 12#-3#np.log10(0.001)#np.max(BarcodeFreqArray[ENDTIMEPOINT]))
    colmin = 0#-5#np.log10(0.00001)#np.min(BarcodeFreqArray[ENDTIMEPOINT]))
    colbin = (colmax-colmin)
    plt.xlabel('Time (cycle)')
    plt.ylabel('log10 Barcode Frequency')
    numLineage = 0
    totalCountsTEND = 0;
    x = [2*t for t in range(TotalTimePoint)]
    '''
    for bc in ListBarcodeIndex:#SortListBC:
        y = np.log10([BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)])
        if not bc in ListADP:    
            count = np.asarray([BarcodeCountsArray[t][bc] for t in range(TotalTimePoint)])
            colorIdx = 0
            if count[-1] != 0:
                colorIdx = TotalTimePoint
            else:
                for idx in range(TotalTimePoint-1,1,-1):
                    if count[idx]==0 and count[idx-1]>0:
                        colorIdx = idx
                        break
            plt.plot(x,y,color=colormap[colmax-colorIdx], alpha=0.5,linewidth=0.2)
        #
        #else:
        #    plt.plot(x,y,color='red', alpha = 1,linewidth=0.3)
        #
    for bc in ListADP:
        y = np.log10([BarcodeFreqArray[t][bc] for t in range(TotalTimePoint)])
        plt.plot(x,y,color='red', alpha = 0.1,linewidth=0.2)
    '''
    y = np.log10([BarcodeFreqArray[t][ADP] for t in range(TotalTimePoint)])
    print(y)
    plt.plot(x,y,color='red', alpha = 1,linewidth=2)
    y = np.log10([BarcodeFreqArray[t][NEU] for t in range(TotalTimePoint)])
    plt.plot(x,y,color='black', alpha = 1,linewidth=2)
    print(y)
    plt.title("P" + str(Pop) +"  NL="+str(np.size(ListBarcodeIndex)))
    plt.savefig(OutputFilePath+"BarcodeFreq_P"+str(Pop)+'_v6.5.pdf',dpi = 800)
    plt.close('all')
'''

T_end = 5
Pop=1
mean_fitness = read_MeanFitness(Pop)
InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
new_mean_fitness1 = mean_fitness - D_fitness
#f1 = [f/9 for f in new_mean_fitness1[0:T_end]]
f1 = [f for f in new_mean_fitness1[0:T_end]]
P1=Pop

Pop=8
mean_fitness = read_MeanFitness(Pop)
InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
new_mean_fitness2 = mean_fitness - D_fitness
#f2 = [f/7 for f in new_mean_fitness2[0:T_end]]
f2 = [f for f in new_mean_fitness2[0:T_end]]
x = [2*t+1 for t in range(T_end)]
plt.figure()
plt.plot(x,f1,color='k', marker = 'o', ls='solid',label='Mono-culture')
plt.plot(x,f2,color=bluish_green, marker ='o', ls='solid',label='Co-culture')
plt.legend(loc='best')
plt.xlim(0,2*T_end)
plt.xlabel("Time (cycle)")
#plt.title("Mean Fitness (1/gen)")
plt.title("Mean Fitness (1/cycle)")
plt.savefig(OutputFilePath+str(P1)+"&"+str(Pop)+"_MeanFitness.pdf")
plt.close('all')
'''

'''
#for Pop in range(1,11):
for Pop in range(1,2):
    ADP_list, ADP_fitness = input_Adaptives(Pop)

    BarcodeCountsArray = readBarcodeCount(Pop)
    BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)
    listLarge = np.nonzero(BarcodeFreqArray[4]>10**-3.5)[0]
    
    InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
    D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
    old_fitness = read_Fitness(Pop)
    new_fitness = np.asarray([old_fitness[T] - D_fitness[T] for T in range(TotalTimePoint-1)])

    mean_fitness1 = 0
    mean_fitness2 = 0
    freq_sum1 = 0
    freq_sum2 = 0
    for idx in range(np.size(ADP_list)):
        bc = int(ADP_list[idx])
        mean_fitness1 = mean_fitness1 + (BarcodeFreqArray[4][bc]+BarcodeFreqArray[5][bc])/2* new_fitness[4][bc]#ADP_fitness[idx]
        freq_sum1 = freq_sum1 + (BarcodeFreqArray[4][bc] + BarcodeFreqArray[5][bc])/2
    for bc in listLarge:
        mean_fitness2 = mean_fitness2 + (BarcodeFreqArray[4][bc]+BarcodeFreqArray[5][bc])/2* new_fitness[4][bc]#ADP_fitness[idx]
        freq_sum2 = freq_sum2 + (BarcodeFreqArray[4][bc] + BarcodeFreqArray[5][bc])/2
    print("P", Pop, mean_fitness1, freq_sum1, mean_fitness2, freq_sum2)

'''


Tini = 2
Tend = 4
Pop =10

InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
old_fitness = read_Fitness(Pop)
new_fitness = np.asarray([old_fitness[T] - D_fitness[T] for T in range(Tini,Tend+1)])
ave_new_fitness = np.mean(new_fitness,axis =0)
print(ave_new_fitness)
new_mean_fitness = read_MeanFitness(Pop) - D_fitness
print(new_mean_fitness)
BarcodeCountsArray = readBarcodeCount(Pop)
ListBarcodeIndex = np.nonzero(np.max(BarcodeCountsArray,axis=0)>0)[0]
print(np.size(ListBarcodeIndex))
ADP_list, ADP_fitness = input_Adaptives(Pop)
ADP_list.astype(int)
#print(ADP_list)
#BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)
#plotBarcodeFreq2(Pop,BarcodeFreqArray,BarcodeCountsArray,ListBarcodeIndex,ADP_list,ave_new_fitness) # plotCleanBarcodeFreq() 
#plotBarcodeFreq3(Pop,BarcodeFreqArray,BarcodeCountsArray,ListBarcodeIndex,ADP_list,ave_new_fitness) # plotCleanBarcodeFreq() 
plot_DFE(Pop,new_fitness[Tend-Tini][list(ADP_list)],list(ADP_list))
plot_meanfitness(Pop,new_mean_fitness)
'''
Tini = 1
Tend = 4
FreqCut = -3.5
CVCut = 1

for Pop in range(1,11):

    BarcodeCountsArray = readBarcodeCount(Pop)
    BarcodeFreqArray, AllCountsArray = get_Barcode_Freq(BarcodeCountsArray)

    InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
    D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
    old_fitness = read_Fitness(Pop)
    new_fitness = np.asarray([old_fitness[T] - D_fitness[T] for T in range(Tini,Tend+1)])

    listLarge = np.nonzero(np.log10(BarcodeFreqArray[Tend])>FreqCut)[0]
    print(Pop,listLarge[0:10])
    mean_fitness = np.mean(new_fitness,axis=0)
    std_fitness = np.std(new_fitness,axis=0)
    CV_fitness = np.asarray(np.abs([std/m/m for std,m in zip(std_fitness,mean_fitness)]))
    list1 = np.nonzero(CV_fitness < CVCut)[0]
    list2 = np.nonzero(mean_fitness > 0)[0]
    listADP = list(set(list1) & set(list2) & set(listLarge)) 
    output_Adpatives(Pop,listADP, new_fitness[Tend-Tini][listADP])
    plot_sCV2sMean(Pop,mean_fitness,CV_fitness,listLarge,BarcodeFreqArray[Tend])
    plot_DFE(Pop,new_fitness[Tend-Tini][listADP])
    mean_fitness1 = 0
    freq_sum1 = 0
    for idx in range(np.size(listADP)):
        bc = int(listADP[idx])
        mean_fitness1 = mean_fitness1 + (BarcodeFreqArray[4][bc]+BarcodeFreqArray[5][bc])/2* new_fitness[Tend-Tini][bc]#ADP_fitness[idx]
        freq_sum1 = freq_sum1 + (BarcodeFreqArray[4][bc] + BarcodeFreqArray[5][bc])/2
    print(mean_fitness1,freq_sum1)
'''
'''
Mono_ADP_fitness = []
Mono_ADP_freq = []
for Pop in range(1,6):
    
    file = open(OutputFilePath+"P"+str(Pop)+"_ADP_Fitness_Cycle9.txt",'r')
    triplets = file.read().split()
    file.close()
    A = np.array(triplets, dtype=np.float32)
    A = np.transpose(np.reshape(A,(-1,2)))
    listADP = [int(bc) for bc in A[0]]
    fitness = [f/9 for f in A[1]]

    BarcodeCountsArray = readBarcodeCount(Pop)
    AllCountsArray = AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(0,len(BarcodeCountsArray))]
    freq = [(BarcodeCountsArray[4][bc]/AllCountsArray[4] + BarcodeCountsArray[5][bc]/AllCountsArray[5])/2 for bc in listADP]
    
    mean_fitness = np.sum(np.asarray([freq[bc]*fitness[bc] for bc in range(np.size(fitness))]))
    print(Pop, mean_fitness)
    plot_DFE(Pop,fitness)
    plot_DFE_population(Pop,fitness,freq)

    Mono_ADP_fitness.extend(fitness)
    Mono_ADP_freq.extend(freq)

Co_ADP_fitness = []
Co_ADP_freq=[]
for Pop in range(6,11):
    
    file = open(OutputFilePath+"P"+str(Pop)+"_ADP_Fitness_Cycle9.txt",'r')
    triplets = file.read().split()
    file.close()
    A = np.array(triplets, dtype=np.float32)
    A = np.transpose(np.reshape(A,(-1,2)))
    listADP = [int(bc) for bc in A[0]]
    fitness = [f/7 for f in A[1]]

    BarcodeCountsArray = readBarcodeCount(Pop)
    AllCountsArray = AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(0,len(BarcodeCountsArray))]
    freq = [(BarcodeCountsArray[4][bc]/AllCountsArray[4] + BarcodeCountsArray[5][bc]/AllCountsArray[5])/2 for bc in listADP]

    mean_fitness = np.sum(np.asarray([freq[bc]*fitness[bc] for bc in range(np.size(fitness))]))
    print(Pop, mean_fitness)
    plot_DFE(Pop,fitness)
    plot_DFE_population(Pop,fitness,freq)

    Co_ADP_fitness.extend(fitness)
    Co_ADP_freq.extend(freq)

plot_2DEF(Mono_ADP_fitness,Co_ADP_fitness)
plot_2DEF_population(Mono_ADP_fitness,Co_ADP_fitness,Mono_ADP_freq,Co_ADP_freq)
'''

'''
Pop=1
file = open(OutputFilePath+"P"+str(Pop)+"_ADP_Fitness_Cycle9.txt",'r')
triplets = file.read().split()
file.close()
A = np.array(triplets, dtype=np.float32)
A = np.transpose(np.reshape(A,(-1,2)))
listADP = [int(bc) for bc in A[0]]
#fitness1 = [f/9 for f in A[1]]
fitness1 = A[1]
BarcodeCountsArray = readBarcodeCount(Pop)
AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(0,len(BarcodeCountsArray))]
freq1 = [(BarcodeCountsArray[4][bc]/AllCountsArray[4] + BarcodeCountsArray[5][bc]/AllCountsArray[5])/2 for bc in listADP]
    
Pop=8
file = open(OutputFilePath+"P"+str(Pop)+"_ADP_Fitness_Cycle9.txt",'r')
triplets = file.read().split()
file.close()
A = np.array(triplets, dtype=np.float32)
A = np.transpose(np.reshape(A,(-1,2)))
listADP = [int(bc) for bc in A[0]]
#fitness2 = [f/7 for f in A[1]]
fitness2 = A[1]
BarcodeCountsArray = readBarcodeCount(Pop)
AllCountsArray = [sum(BarcodeCountsArray[i]) for i in range(0,len(BarcodeCountsArray))]
freq2 = [(BarcodeCountsArray[4][bc]/AllCountsArray[4] + BarcodeCountsArray[5][bc]/AllCountsArray[5])/2 for bc in listADP]

plot_2DEF(fitness1,fitness2)
#plot_2DEF_population(fitness1,fitness2,freq1,freq2)

'''

'''
plt.figure()
x = [2*t+1 for t in range(TotalTimePoint-1)]
plt.xlabel("Time (cycle)")
plt.title("Mean Fitness (1/cycle)")
plt.ylim(-0.05,3)
for Pop in range(1,11):
    OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
    D_fitness = np.load(OutputFilePath+"P"+str(Pop)+"_Dx.npy")
    new_mean_fitness = read_MeanFitness(Pop) - D_fitness
    if (Pop < 6):
        colorIdx = 'black'
    elif (Pop >= 6):
        colorIdx = bluish_green
    if Pop==1:
        plt.plot(x,new_mean_fitness,marker='o',ls='solid',color=colorIdx,label="Mono-culture")
    elif Pop == 6:
        plt.plot(x,new_mean_fitness,marker='o',ls='solid',color=colorIdx,label="Co-culture")
    else:
        plt.plot(x,new_mean_fitness,marker='o',ls='solid',color=colorIdx)
    print(new_mean_fitness)
plt.legend(loc = 'upper left')
plt.savefig("/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/"+"MeanFitness_allTime_0312.pdf")
'''

'''
for T in range(3,8):
#for T in range(4,5):
    cycle = 2*T+1
    mono_mean_fitness = []
    for Pop in range(1,6):
        InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
        D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
        mono_mean_fitness.append( read_MeanFitness(Pop)[T] - D_fitness[T])
        #old_fitness = read_Fitness(Pop)
        #new_fintess = np.asarray(old_fitness[T] - D_fitness[T])
        #np.save(OutputFilePath+"RelativeFitness_P"+str(Pop)+"_cycle9.npy",new_fintess)
    #mono_mean_fitness = np.asarray(mono_mean_fitness)/9
    mono_mean_fitness = np.asarray(mono_mean_fitness)
    mean_mono = np.mean(mono_mean_fitness)
    std_mono = np.std(mono_mean_fitness)/np.size(mono_mean_fitness)**0.5
    co_mean_fitness = []
    for Pop in range(6,11):
        InputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/P"+str(Pop)+"/"
        D_fitness = np.load(InputFilePath+"P"+str(Pop)+"_Dx.npy")
        co_mean_fitness.append( read_MeanFitness(Pop)[T] - D_fitness[T])
        #old_fitness = read_Fitness(Pop)
        #new_fintess = np.asarray(old_fitness[T] - D_fitness[T])
        #np.save(OutputFilePath+"RelativeFitness_P"+str(Pop)+"_cycle9.npy",new_fintess)
    #co_mean_fitness = np.asarray(co_mean_fitness)/7
    co_mean_fitness = np.asarray(co_mean_fitness)
    mean_co = np.mean(co_mean_fitness)
    std_co = np.std(co_mean_fitness)/np.size(co_mean_fitness)**0.5
    OutputFilePath = "/mnt/c/Users/Ruby/Google Drive/Lab Project/AllPopReadCounts/20190227_NSF_Grant/"
    fig, ax = plt.subplots()
    width = 0.35       # the width of the bars
    Xlabel = ['Mono-culture', 'Co-culture']
    barX = np.arange(len(Xlabel))+width
    barY = [mean_mono, mean_co]
    error = [std_mono, std_co]
    ax.bar(barX[0],barY[0], yerr=error[0], width=width, align='center', color = "black")
    ax.bar(barX[1],barY[1], yerr=error[1], width=width, align='center', color = bluish_green)
    ax.set_xticks([idx for idx in barX])
    ax.set_xticklabels(['Mono-culture', 'Co-culture'],fontsize=20)
    ax.set_xlim(-0.3,2.0)
    ax.set_title(" Mean Fintess (1/cycle)",size=20)
    #ax.set_title(" Mean Fintess (1/gen)",size=20)
    plt.savefig(OutputFilePath+"0312MeanFitness_Cycle"+str(cycle)+".pdf")

'''