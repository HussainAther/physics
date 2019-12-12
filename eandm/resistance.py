"""
Calculate the resistance for a set of resistors
following electrical circuit theory.
"""

class resistor:
    def __init__(self,
                 resistance,
                 a=None,
                 b=None,
                 symbol="r"):
        self.resistance = resistance
        self.a = a
        self.b = b
        self.symbol = symbol
    def res(self): 
        return self.resistance
    def setVoltage(self, voltage): 
        self.voltage = voltage
    def current(self): 
        return self.voltage / self.res()
    def effect(self): 
        return self.current() * self.voltage
