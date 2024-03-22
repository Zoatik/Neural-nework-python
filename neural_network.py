from math import tanh, exp
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
class ActFunc:
    def __init__(self, funcName):
        if funcName == "tanh":
            self.func = tanh
        elif funcName == "sigmoid":
            self.func = sigmoid
        elif funcName == "leakyReLu":
            self.func = leakyReLu
    
    
class Neur:
    def __init__(self, val = 0.0):
        self.biais = 0.0
        self.val = val
        self.weightIndexes = []
        self.linksWeight = []
        
    def calcVal(self, givenVal, weight):
        return givenVal*weight + self.biais
        
    def activate(self, fct, new_val):
        self.val = fct(new_val)
        return self.val
    
                    
          
class Network:
    #Network constructor with default activation function set to tanh
    def __init__(self, fct = ActFunc("tanh")):
        self.nbLayers = 0
        self.network = [[]]
        self.layers = []
        self.fct = fct
    
    #Layer creation from python list of values
    def createLayer(self, list):
        layer = []
        for val in list:
            layer.append(Neur(val))
        return layer
    
    #prints the network weights
    def print(self):
        for layer in self.network:
            for neur in layer:
                form_list = [ '%.2f' % elem for elem in neur.linksWeight]
                print(form_list, end=' ')
            print()
    
    def setInput(self, inputData):
        in_dataLayer = self.createLayer(inputData)
        self.network.insert(0,in_dataLayer)
        
    #Set the network 2d list with the given sizes
    def setNetwork(self, nbHiddenLayers, out_layerSize, hiddenLayerSize):
        self.nbLayers = nbHiddenLayers + 2
        self.network = [[Neur() for i in range(hiddenLayerSize)] for j in range(nbHiddenLayers)]
        self.network += [[Neur()for i in range(out_layerSize)]]
        
    def setActivFunc(self, fct):
        self.fct = ActFunc(fct).func
    
    #initialize the weights for all possible links
    def initWeights(self):
        for i in range(1, self.nbLayers):
            for j in range(len(self.network[i])):
                for k in range(len(self.network[i-1])):
                    self.network[i][j].linksWeight.append(rd.uniform(0.0,1.0))
    
    #executes the forward propagation from a given input data              
    def forwProp(self):
        #self.print()
        for i in range(1, self.nbLayers):
            for j in range(len(self.network[i])):
                tmp_val = 0
                for k in range(len(self.network[i-1])):
                    weight = self.network[i][j].linksWeight[k]
                    tmp_val += self.network[i][j].calcVal(self.network[i-1][k].val, weight)
                self.network[i][j].activate(self.fct, tmp_val)
            
        return self.network[-1]
    
    def showLinks(self):
        for layer in self.network:
            for neur in layer:
                print(neur.linksWeight)
                
    def showNeurInfos(self, layer, index):
        print(self.network[layer][index].linksWeight)

#####_Main_#####
#setting up the network
network = Network()
network.setNetwork(3,2,10)
network.setActivFunc("leakyReLu")

input_list  = [1,2,3,4]
network.setInput(input_list)
network.print()
network.initWeights()
print()
network.print()

nbIterations = int(input("nombre d'itérations : "))

for i in range(nbIterations):
    layer_out = network.forwProp()
    
    print(i+1, "/", nbIterations, ": DONE")
    print(['%.2f' % neur.val for neur in layer_out])
