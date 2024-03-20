from cmath import tanh
import random as rd

class Neur:
    def __init__(self, val = 0):
        self.biais = rd.uniform(0.0,1.0)
        self.val = val
        self.weightIndexes = []
        self.back_links = []
        
    def calcVal(self, givenVal, weight):
        return givenVal*weight + self.biais
    def activate(self, fct, new_val):
        self.val = fct(new_val)
        return self.val
    
class Layer:
    def __init__(self, size, list = []):
        self.neur_list = []
        self.size = size
        if(list == []):
            for i in range(size):
                self.neur_list.append(Neur())
        else:
            for i in range(size):
                self.neur_list.append(Neur(list[i]))
    def print(self):
        for i in range(self.size):
            print(self.neur_list[i].val)
                    
            
class Network:
    def __init__(self):
        self.nbLayers = 0
        self.layers = []
        self.fct = tanh
        
    def setLayers(self, nbHiddenLayers, in_layerSize, out_layerSize, hiddenLayerSize):
        self.nbLayers = nbHiddenLayers + 2
        self.layers.append(Layer(in_layerSize))
        for i in range(nbHiddenLayers):
            self.layers.append(Layer(hiddenLayerSize))
        self.layers.append(Layer(out_layerSize))
        
    def setActivFunc(self,fct):
        self.fct = fct
        
    def initWeights(self):
        for i in range(1, self.nbLayers):
            for j in range(self.layers[i].size):
                for k in range(self.layers[i-1].size):
                    self.layers[i].neur_list[j].back_links.append(rd.uniform(0.0,1.0))
                    
    def process(self, in_data):
        fstLayer = Layer(len(in_data), in_data)
        self.layers[0] = fstLayer
        for i in range(1, self.nbLayers):
            for j in range(self.layers[i].size):
                tmp_val = 0
                for k in range(len(self.layers[i].neur_list[j].back_links)):
                    weight = self.layers[i].neur_list[j].back_links[k]
                    tmp_val += self.layers[i].neur_list[j].calcVal(self.layers[i].neur_list[j].val, weight)
                print("new val: ", self.layers[i].neur_list[j].activate(self.fct, tmp_val))
        out_data = self.layers[-1]
        return out_data
    
    def showLinks(self):
        for layer in self.layers:
            for neur in layer.neur_list:
                print(neur.back_links)
                
    def showNeurInfos(self, layer, index):
        print(self.layers[layer].neur_list[index].back_links)
                    
network = Network()
network.setLayers(3,4,1,3)
network.setActivFunc(tanh)
network.initWeights()

network.showLinks()
input_list  = [1,2,3,4]
layer_out = network.process(input_list)
layer_out.print()    
                
        
        
        
        