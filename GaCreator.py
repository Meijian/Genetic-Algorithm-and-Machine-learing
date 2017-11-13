import sys
import os
import pandas as pd
import numpy as np
import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


#simulate a dataset with genes expression data and sample informatiopn
def dataSimulator (geneNum,sampSize):
    colName=range(1,geneNum+1,1)
    tmp = pd.DataFrame(data=np.random.uniform(low=-5, high=5, size=(sampSize,geneNum)), columns=colName)
    #df = pd.DataFrame(np.random.randn(r,c), columns = ['gene'+str(i) for i in range(c)], index = list((string.uppercase+string.lowercase)[0:r])) #Add alphabetic column names, need to check string module
    tmp['sex'] = pd.Series(np.random.randint(0,2,size=sampSize))
    tmp['age'] = pd.Series(np.random.randint(45,75,size=sampSize))
    tmp['status'] = pd.Series(np.random.randint(0,4,size=sampSize))
    return tmp

#Create a initial population with leave-one-our manner
def iniChr(geneIndx):
    allChr = [list(range(1,geneIndx+1)) for n in range(geneIndx)]
    for i in range(0,geneIndx):
        allChr[i][i]=0
    return allChr

##The classifier used to predict the cancer type, the accuracy score is returned for each chromosome
def classifier(chr):
    ncol=len(chr.columns)
    Y=chr[chr.columns[ncol-1]]
    X=chr[chr.columns[list(range(ncol-1))]]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
    
    CART = DecisionTreeClassifier()
    CART.fit(X_train, y_train)
    predictions = CART.predict(X_test)
    return accuracy_score(y_test, predictions)  

##Select the best ML method using cross-validation
def modelSelection(curPop):
    meanFit=[]
    mlnames=[]
    for r in range(geneNum):
        d=list(curPop[r])
        for i in range(len(curPop[r])):
            if curPop[r][i] == 0:
                d.remove(0)
        tmp=pd.DataFrame(data=oriData[d])
        tmp[['sex','age','status']]=oriData[['sex','age','status']]    
        ncol=len(tmp.columns)
        Y=tmp[tmp.columns[ncol-1]]
        X=tmp[tmp.columns[list(range(ncol-1))]]        
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,random_state=0)
        print("data is ready\t")
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        models.append(('ANN',MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(12,), random_state=1)))
        
        results = []
        meanScore=[]
        names = []
    
        folds=10
        seed=7
        scoring = 'accuracy'
        validation_size = 0.2
    
        for name, model in models:
            kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
            cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            meanScore.append(cv_results.mean())
            names.append(name)
            print("Accuracy: %s - %f (+/-%f)" % (name, cv_results.mean(), cv_results.std()))
        
        mlnames=names
        meanFit.append(meanScore)
        fig = plt.figure()
        prefix='Algorithm Comparison'
        title='%s %d'%(prefix,r)
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        plt.boxplot(results)
        ax.set_xticklabels(names)
        pre='MLmodelComparisons'
        post='.png'
        plotName='%s%d%s'%(pre,r,post)
        plt.savefig(plotName)
        
    meanFit2=pd.DataFrame(data=meanFit)
    mlPer=[]
    for i in range(7):
        mlPer.append(meanFit2[i].mean())
    fig=plt.figure()
    plt.plot(mlPer,'g',linewidth=2.0)
    fig.suptitle('ML performance comparison')
    ax = fig.add_subplot(111)
    mlnames=names #remove
    ax.set_xticklabels(mlnames)
    plt.ylabel('Fitness scores')
    plt.xlabel('Algorithms')    
    plt.savefig('ML performace comparison.png')
    
    
##Fitness score estimation for each chromosome of current population
def fitEstimator(curPop):
    accu = []
    if not isinstance(curPop[0],list):
        d=list(iniPoint)
        for i in range(len(iniPoint)):
            if iniPoint[i] == 0:
                d.remove(0)
        tmp=pd.DataFrame(data=oriData[d])
        tmp[['sex','age','status']]=oriData[['sex','age','status']]
        #Machine learning algorithms, call classifier----------------------
        accu.append(classifier(tmp))
    else:
        for r in range(geneNum):
            d=list(curPop[r])
            for i in range(len(curPop[r])):
                if curPop[r][i] == 0:
                    d.remove(0)
            tmp=pd.DataFrame(data=oriData[d])
            tmp[['sex','age','status']]=oriData[['sex','age','status']]
            #Machine learning algorithms, call classifier----------------------
            accu.append(classifier(tmp))
    return accu

##A method to perform gournment selection with group size=4. 
##A total of 20 chromosomes are selected for reproduction, duplicate selections
##are allowed
def selector(fitScore):
    grpSize=8
    selected=[]
    for x in range(len(fitScore)):
        group=random.sample(range(len(fitScore)), grpSize)
        selected.append(findMax(fitScore,group))
    return selected

##Cross over operation to perform 
def crossOver(curPop,selChrs):
    crossRate=0.6
    nextPop=[]
    for n in range(0,geneNum,2):
        if random.random()<0.6:
            crpt=random.randint(0,19)
            newChr1=deepcopy(curPop[selChrs[n]][0:crpt]+curPop[selChrs[n+1]][crpt:20])
            newChr2=deepcopy(curPop[selChrs[n+1]][0:crpt]+curPop[selChrs[n]][crpt:20])
            nextPop.append(mutator(newChr1))
            nextPop.append(mutator(newChr2))
        else:
            newChr3=deepcopy(curPop[selChrs[n]])
            newChr4=deepcopy(curPop[selChrs[n+1]])
            nextPop.append(mutator(newChr3))
            nextPop.append(mutator(newChr4)) 
    return nextPop

##Mutation operation to perform mutation at each 
def mutator(curChr):
    mutRate=0.05
    for m in range(len(curChr)):
        if random.random()<mutRate:
            if curChr[m]==0:
                curChr[m]=m+1
            else:
                curChr[m]=0
    return curChr
    

##A method to find the highest accuracy score and its index, and return both of them
def findMax(fitScore,grp):
    MAX=0
    idx=0
    for g in range(4):
        if fitScore[grp[g]]>MAX:
            MAX=fitScore[grp[g]]
            idx=grp[g]
    return idx
        
    
##Perform reproduction with initial population for user-defined generations
def reproduction (iniPop,generation):
    popFit=[]
    maxFit=[]
    minFit=[]
    curPop=deepcopy(iniPop)
    for g in range(generation): 
        scores=fitEstimator(curPop)
        popFit.append(sum(scores)/len(scores))
        maxFit.append(max(scores))
        minFit.append(min(scores))
        result=selector(scores)
        curPop=crossOver(curPop,result)

    return popFit, curPop, scores, maxFit, minFit

##Plot average fitness and maximum fitness score for user-defined generations
def plotFit(result,postfix,numIter):
    plt.figure()
    plt.suptitle('GA performance')
    l1,l2,l3, = plt.plot(popFit[0][0:numIter],'r',
             popFit[3][0:numIter],'b',
             popFit[4][0:numIter],'g')
    #plt.figlegend((l1,l2,l3),('Average','Maximum','Minimum'),'center right')
    plt.legend((l1,l2,l3),('Average','Maximum','Minimum'),loc='lower right')
    plt.ylabel('Fitness Scores')
    plt.xlabel('Generations')
    pre='Fitness score change'
    end='.png'
    pltName='%s %s%d%s'%(pre,postfix,numIter,end)
    plt.savefig(pltName)
    


def MetroHast (iterNum,mut):
    iniPoint=list(range(1,21))
    score=[]
    curPoint=list(iniPoint)
    for j in range(iterNum):
        curScore=fitEstimator(curPoint)[0]
        score.append(curScore)
        tmp=list(curPoint)
        nextPoint=mutator(tmp)
        nextScore=fitEstimator(nextPoint)[0]
        if (nextScore/curScore)>=1:
            curPoint=list(nextPoint)
        elif (nextScore/curScore)>random.random():
            curPoint=list(nextPoint)
        else:
            curPoint=list(curPoint)
            del score[len(score)-1]
    plt.figure()
    plt.suptitle('M-H performance - fitness')
    plt.plot(score,'b')
    plt.ylabel('Fitness Scores')
    plt.xlabel('Generations')
    pre='M-H fitness score change'
    end='.png'
    mid='_'
    pltName='%s %d%s%.3f%s%s'%(pre,iterNum,mid,mut,mid,end)
    plt.savefig(pltName)    
    return score


            
            
        
        
  
    

##-------------------------------------------------------------------------
#Function calling area, main function
geneNum = 200
sampSize = 1000
oriData = dataSimulator(geneNum,sampSize)
#oriData.to_csv('geneEx_simulation.csv', encoding = 'utf-8')
oriData = pd.read_csv('geneEx_simulation.csv')  #Readin saved simulation data
Pop1 = iniChr(geneNum)
popFit=reproduction(Pop1,1000)
plotFit(popFit,'grp8',1000)
##tmp=pd.DataFrame(data=popFit[1])
##tmp.to_csv("Current Population after reproduction.csv")
score=MetroHast(200,0.05)
