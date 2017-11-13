import GaCreator
geneNum = 100
sampSize = 500
oriData = dataSimulator(geneNum,sampSize)
Pop1 = iniChr(geneNum)
popFit=reproduction(Pop1,1000)