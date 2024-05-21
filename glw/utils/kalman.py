import numpy as np
import copy

class kalmanFilter:
    def __init__(self, numdim=1, LastP=0.02, Now_P=0.0, Kg=0.2, Q=0.001, R=0.1) -> None:
        """
         Initialize the object with default values. This is called by __init__ and should not be called directly
         
         @param numdim - Number of dimensions to initialize
         
         @return self for method chaining ( no return value for self. __init__ ) Note : The constructor does not return anything
        """
        self.o_LastP = LastP
        self.o_Now_P = Now_P
        self.o_numdim = numdim
        self.o_Kg = Kg
        
        self.LastP = np.array([self.o_LastP] * self.o_numdim)
        self.Now_P = np.array([self.o_Now_P] * self.o_numdim)
        self.Kg = np.array([self.o_Kg] * self.o_numdim)
        
        self.Q = Q
        self.R = R
        
        self.out = np.array([0.0] * self.o_numdim)
        
    def forward(self, data):
        
        self.Now_P = self.LastP + self.Q
        self.Kg = self.Now_P / (self.Now_P + self.R)
        self.out = self.out + self.Kg * (data -self.out)
        self.LastP = (1-self.Kg) * self.Now_P
        return copy.deepcopy(self.out)
    
    def stable(self, data):
        self.LastP = np.array([self.o_LastP] * self.o_numdim)
        self.Now_P = np.array([self.o_Now_P] * self.o_numdim)
        self.Kg = np.array([self.o_Kg] * self.o_numdim)
        self.out = data


        


# if __name__ == '__main__':
#     k = kalmanFilter(3)
#     print(k.LastP)
#     print(k.forward(np.array([0,0,0.1])))