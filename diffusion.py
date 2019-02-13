import numpy
from matplotlib import pyplot
import time, sys

"""

The one-dimensional diffusion equation is:

∂u/∂t = ν (∂2u/∂x2)

This equation has a second-order derivative that we will discretize. The second-order derivative can be represented geometrically
as the line tangent to the curve given by the first derivative. We will discretize the second-order
derivative with a Central Difference scheme: a combination of Forward Difference and Backward Difference of the first
derivative. Consider the Taylor expansion of ui+1 and ui−1 around ui:


ui+1=ui+Δx∂u∂x∣∣∣i+Δx22∂2u∂x2∣∣∣i+Δx33!∂3u∂x3∣∣∣i+O(Δx4)
ui−1=ui−Δx∂u∂x∣∣∣i+Δx22∂2u∂x2∣∣∣i−Δx33!∂3u∂x3∣∣∣i+O(Δx4)

"""
