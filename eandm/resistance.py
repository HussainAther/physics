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

class Serial(Resistor):
    """
    Resistors in series.
    """
    def __init__(self, a, b):
        """
        Variables to add, initialize.
        """
        super().__init__(0, b, a, "+")
    def res(self): 
        """
        Add them in linear.
        """
        return self.a.res() + self.b.res()
    def setVoltage(self, voltage):
        """
        Calculate voltage drop.
        """
        ra = self.a.res()
        rb = self.b.res()
        self.a.setVoltage(ra/(ra+rb) * voltage)
        self.b.setVoltage(rb/(ra+rb) * voltage)
        self.voltage = voltage

class Parallel(Resistor):
    """
    Add the resistors in parallel.
    """
    def __init__(self,a, b):
        """
        Initialize resistors a and b.
        """ 
        super().__init__(0, b, a, '*')
    def res(self):
        """
        Add them in quadrature.
        """ 
        return 1 / (1 / self.a.res() + 1 / self.b.res())
    def setVoltage(self, voltage):
        """
        Add the voltages in linear.
        """
        self.a.setVoltage(voltage)
        self.b.setVoltage(voltage)
        self.voltage = voltage
