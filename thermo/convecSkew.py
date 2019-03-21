import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from constants import constants as c
from new_thermo import convertSkewToTemp,convertTempToSkew,theta,wsat,thetaes

def convecSkew(figNum):
      """       
      Usage:  convecSkew(figNum)
      Input:  figNum = integer
       Takes any integer, creates figure(figNum), and plots a
       skewT logp thermodiagram.
      Output: skew=30 and the handle for the plot
      """
      fig=plt.figure(figNum)
      fig.clf()
      ax1=fig.add_subplot(111)
      yplot = range(1000,190,-10)
      xplot = range(-300,-139)
      pvals = np.size(yplot)
      tvals = np.size(xplot)
      temp = np.zeros([pvals, tvals])
      theTheta = np.zeros([pvals, tvals])
      ws = np.zeros([pvals, tvals])
      theThetae = np.zeros([pvals, tvals])      
      skew = 30 #skewness factor (deg C)

      # lay down a reference grid that labels xplot,yplot points 
      # in the new (skewT-lnP) coordinate system .
      # Each value of the temp matrix holds the actual (data)
      # temperature label (in deg C)  of the xplot, yplot coordinate.
      # pairs. The transformation is given by W&H 3.56, p. 78.  Note
      # that there is a sign difference, because rather than
      # taking y= -log(P) like W&H, I take y= +log(P) and
      # then reverse the y axis         
      
      for i in yplot:
            for j in xplot:
                  # Note that we don't have to transform the y
                  # coordinate, as it is still pressure.
                  iInd = yplot.index(i)
                  jInd = xplot.index(j)
                  temp[iInd, jInd] = convertSkewToTemp(j, i, skew)
                  Tk = c.Tc + temp[iInd, jInd]
                  pressPa = i * 100.
                  theTheta[iInd, jInd] = theta(Tk, pressPa)
                  ws[iInd, jInd] = wsat(Tk, pressPa)
                  theThetae[iInd, jInd] = thetaes(Tk, pressPa)
                  
      #
      # Contour the temperature matrix.
      #

      # First, make sure that all plotted lines are solid.
      mpl.rcParams['contour.negative_linestyle'] = 'solid'
      tempLabels = range(-40, 50, 10)
      tempLevs = ax1.contour(xplot, yplot, temp, tempLabels, \
                            colors='k')
      
      #
      # Customize the plot
      #
      ax1.set_yscale('log')
      locs = np.array(range(100, 1100, 100))
      labels = locs
      ax1.set_yticks(locs)
      ax1.set_yticklabels(labels) # Conventionally labels semilog graph.
      ax1.set_ybound((200, 1000))
      plt.setp(ax1.get_xticklabels(), weight='bold')
      plt.setp(ax1.get_yticklabels(), weight='bold')
      ax1.yaxis.grid(True)

      
      thetaLabels = range(200, 390, 10)
      thetaLevs = ax1.contour(xplot, yplot, theTheta, thetaLabels, \
                        colors='b')


      wsLabels =[0.1,0.25,0.5,1,2,3] + range(4, 20, 2) + [20,24,28]

      wsLevs = ax1.contour(xplot, yplot, (ws * 1.e3), wsLabels, \
                        colors='g')

      thetaeLabels = np.arange(250, 410, 10)
      thetaeLevs = ax1.contour(xplot, yplot, theThetae, thetaeLabels, \
                        colors='r') 
      
      # Transform the temperature,dewpoint from data coords to
      # plotting coords.
      ax1.set_title('skew T - lnp chart')
      ax1.set_ylabel('pressure (hPa)')
      ax1.set_xlabel('temperature (deg C)')

      #
      # Crop image to a more usable size
      #    
      

      TempTickLabels = range(-15, 40, 5)

      TempTickCoords = TempTickLabels
      skewTickCoords = convertTempToSkew(TempTickCoords, 1.e3, skew)
      ax1.set_xticks(skewTickCoords)
      ax1.set_xticklabels(TempTickLabels)

      skewLimits = convertTempToSkew([-15, 35], 1.e3, skew)

      ax1.axis([skewLimits[0], skewLimits[1], 300, 1.e3])
      
      #
      # Create line labels
      #
      fntsz = 9 # Handle for 'fontsize' of the line label.
      ovrlp = True # Handle for 'inline'. Any integer other than 0
                # creates a white space around the label.
                
      thetaeLevs.clabel(thetaeLabels, inline=ovrlp, fmt='%5d', fontsize=fntsz,use_clabeltext=True)
      tempLevs.clabel(inline=ovrlp, fmt='%2d', fontsize=fntsz,use_clabeltext=True)
      thetaLevs.clabel(inline=ovrlp, fmt='%5d', fontsize=fntsz,use_clabeltext=True)
      wsLevs.clabel(inline=ovrlp, fmt='%2d', fontsize=fntsz,use_clabeltext=True)
      #print thetaeLabels
      #
      # Flip the y axis
      #
      
      ax1.invert_yaxis()
      ax1.figure.canvas.draw()
      
      return skew, ax1

if __name__== "__main__":
      skew, ax1 =convecSkew(1)
      plt.show()
      
