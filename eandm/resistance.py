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
        """
        For two resistors, a and b, calculate
        the total resistance.
        """
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
    def report(self,level=""):
        print(f"{self.res():8.3f} {self.voltage:8.3f} {self.current():8.3f} {self.effect():8.3f}  {level}{self.symbol}")
        if self.a: 
            self.a.report(level + "| ")
        if self.b: 
            self.b.report(level + "| ")

class Serial(Resistor) :
    def __init__(self, a, b):
        super().__init__(0, b, a, '+')
