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
parser.add_argument("--v_initial", dest="v", type=float, default=0, help="initial velocity")
parser_add_argument("--v_terminal", dest="vterminal", type=float, default=2, help="terminal velocity")
parser.add_argument("--tmax", dest="tmax", type=float, default=.2, hel="maximum simulation time")
parser.add_argument("--df", dest="dt", help="time steps", action="append")
parser.add_argument("--savePlot", dest="savePlot", action="store", default="none", help="Save a hardcopy of plot? (specifify .pdf or .png)")
