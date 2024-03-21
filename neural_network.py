from cmath import tanh, exp
import random as rd

def sigmoid(x, derivate = False):
    if derivate:
        return exp(x)/(1+exp(-x))**2
    return 1/(1+exp(-x))
def th(x, derivate = False):
    if derivate:
        return 1-tanh(x)**2
    return tanh(x)
def leakyReLu(x, eps = 0, derivate = False):
    if derivate:
        if x < 0:
            return  eps
        return 1
    return max(eps,x)
class actFunc:
    def __init__(self, funcName):
        if funcName == "tanh":
            self.func = tanh
        elif funcName == "sigmoid":
            self.func = sigmoid
        elif funcName == "leakyReLu":
            self.func = leakyReLu
    
    
class Neur:
    def __init__(self, val = 0):
        self.biais = 0.0
        self.val = val
        self.weightIndexes = []
        self.backLinks = []
        
    def calcVal(self, givenVal, weight):
        return givenVal*weight + self.biais
        
    def activate(self, fct, new_val):
        self.val = fct(new_val)
        return self.val
    
                    
            
class Network:
    def __init__(self, fct = actFunc("tanh")):
        self.nbLayers = 0
        self.network = [[]]
        self.layers = []
        self.fct = fct
    
    def createLayer(self, list):
        layer = []
        for val in list:
            layer.append(Neur(val))
        return layer
    
    def print(self):
        for layer in self.network:
            for neur in layer:
                print(neur.backLinks)
        
    def setNetwork(self, nbHiddenLayers, out_layerSize, hiddenLayerSize):
        self.nbLayers = nbHiddenLayers + 2
        self.network += [[Neur()]*hiddenLayerSize]*nbHiddenLayers
        self.network += [[Neur()]*out_layerSize]
        
    def setActivFunc(self,fct):
        self.fct = fct
        
    def initWeights(self):
        for i in range(1, self.nbLayers):
            for j in range(len(self.network[i])):
                for k in range(len(self.network[i-1])):
                    self.network[i][j].backLinks.append(rd.uniform(0.0,1.0))
                    
    def forwProp(self, in_data):
        in_dataLayer = self.createLayer(in_data)
        self.network[0] = in_dataLayer + self.network
        for i in range(1, self.nbLayers):
            for j in range(len(self.network[i])):
                tmp_val = 0
                for k in range(len(self.network[i][j].backLinks)):
                    weight = self.network[i][j].backLinks[k]
                    tmp_val += self.network[i][j].calcVal(self.network[i-1][j].val, weight)
                self.network[i][j].activate(self.fct, tmp_val)
            
        return self.network[-1]
    
    def showLinks(self):
        for layer in self.network:
            for neur in layer:
                print(neur.backLinks)
                
    def showNeurInfos(self, layer, index):
        print(self.network[layer][index].backLinks)

network = Network()
network.setNetwork(50,2,3)
network.setActivFunc(tanh)
network.initWeights()

#network.showLinks()
input_list  = [1,2,3,4]
nbIterations = int(input("nombre d'itérations : "))

for i in range(nbIterations):
    layer_out = network.forwProp(input_list)
    print([neur.val for neur in layer_out])
                
        
        
        
        



    