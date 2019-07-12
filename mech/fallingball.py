import sys, EulerFreeFall
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime

"""
This program uses the Euler method to solve for the motion of a
ball dropped from rest close to the surface of Earth . The
program depends on the function VelocityLinearDrag to execute
the Euler method.
"""

t = 0
g = 9.8

parser = argparse.ArgumentParser()

parser.add_argument("--filename", dest="filename", action="store", help="output filename", default=junk.dat")
