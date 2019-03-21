import numpy as np
from constants import constants as c
import matplotlib.cbook as cbook
import numpy.testing as test

def convertSkewToTemp(xcoord, press, skew):
    """
    convertSkewToTemp(xcoord, press, skew)

    Determines temperature from knowledge of a plotting coordinate
    system and corresponding plot skew.
    
    Parameters
    - - - - - -
    xcoord : int
        X coordinate in temperature plotting coordinates.
    press : float
        Pressure (hPa).
    skew : int
        Skew of a given coordinate system.

    Returns
    - - - -
    Temp : float
        Converted temperature in degC.

    Examples
    - - - - -
    >>> test.assert_almost_equal(convertSkewToTemp(300, 8.e4, 30),638.6934,decimal=3)
    """
    Temp = xcoord  + skew * np.log(press);
    return Temp

def convertTempToSkew(Temp, press, skew):
    """
    convertTempToSkew(Temp, press, skew)

    Determines the transformed temperature in plotting coordinates.
    
    Parameters
    - - - - - -
    Temp : float
        Temperature (degC)
    press : float
        Pressure (hPa).
    skew : int
        Designated skew factor of temperature.

    Returns
    - - - -
    tempOut : float
        Converted temperature (degC).

    Examples
    - - - - -
    >>> test.assert_almost_equal(convertTempToSkew(30., 8.e4, 30),-308.693,decimal=3)
    """
    
    tempOut = Temp - skew * np.log(press);
    return tempOut

def findWvWl(Temp, wT, press):
    """
    findWvWl(Temp, wT, press)

    Computes the vapour and liquid water mixing ratios.

    Parameters
    - - - - - -
    Temp : float
        Temperature (K).
    wT : float
        Total water mixing ratio (kg/kg).
    press : float
        Pressure (Pa).


    Returns
    - - - -
    wv : float
        Water vapour mixing ratio (kg/kg).
    wl : float
        Liquid water mixing ratio (kg/kg).


    Raises
    - - - -
    AssertionError
        If any of the inputs are in vector form.

    Examples
    - - - - -
    >>> test.assert_almost_equal(findWvWl(250., 0.01, 8.e4),(0.00074, 0.00925),decimal=5)
    >>> test.assert_almost_equal(findWvWl(300., 0.01, 8.e4),(0.01, 0),decimal=4)
    """
    Temp=np.atleast_1d(Temp)
    press=np.atleast_1d(press)
    wT=np.atleast_1d(wT)
    if (np.size(Temp)*np.size(press)*np.size(wT)) != 1:
        raise AssertionError('need three scalars')
    wsVal = wsat(Temp, press)
    if wsVal[0] > wT[0]: #unsaturated
        wv = wT[0]
        wl = 0
    else:  #saturated
        wv = wsVal[0]
        wl = wT[0] - wv
    return wv, wl

def tinvert_thetae(thetaeVal, wT, p):
    """
    temp,wv,wl=tinvert_thetae(thetaeVal, wT, p)

    Uses a rootfinder to determine the temperature for which the
    pseudo equivilant potential temperature (thetaep) is equal to the
    equivilant potential temperature (thetae) of the parcel.

    Parameters
    - - - - - -
    thetaeVal : float
        Thetae of parcel (K).
    wtotal : float
        Total water mixing ratio (kg/kg).
    p : float
        Pressure of parcel in (Pa).

    Returns
    - - - -
    theTemp : float
        Temperature for which thetaep equals the parcel thetae (K).
    wv : float
        Vapor mixing ratio of the parcel (kg/kg).
    wl : float
        liquid water mixing ratio of the parcel (kg/kg) at 'p'.

    Raises
    - - - -
    IOError
        If 'p' is larger than 100000 Pa.

    Examples
    - - - - -
    >>> test.assert_array_almost_equal(tinvert_thetae(300., 0.001, 8.e4),(278.405, 0.001, 0),decimal=3)
    
    """
    import scipy.optimize
    
    if p > 1.e5:
        raise IOError('expecting pressure level less than 100000 Pa')
    # The temperature has to be somewhere between thetae
    # (T at surface) and -40 deg. C (no ice).    
    handle = Tchange
    theTemp = scipy.optimize.zeros.brenth(handle, 233.15, \
                                      thetaeVal, (thetaeVal, wT, p));
    [wv,wl] = findWvWl(theTemp, wT, p);
    return theTemp,wv,wl


def Tchange(Tguess, thetaeVal, wT, p):
    [wv, wl] = findWvWl(Tguess, wT, p);
    tdGuess = Tdfind(wv, p);
    # Iterate on Tguess until this function is
    # zero to within tolerance.
    return thetaeVal - thetaep(tdGuess, Tguess, p);


def Tdfind(wv, p):
    """
    Tdfind(wv, p)

    Calculates the due point temperature of an air parcel.

    Parameters
    - - - - - -
    wv : float
        Mixing ratio (kg/kg).
    p : float
        Pressure (Pa).

    Returns
    - - - -
    Td : float
        Dew point temperature (K).

    Examples
    - - - - -
    >>> test.assert_almost_equal(Tdfind(0.001, 8.e4),253.39429,decimal=4)

    References
    - - - - - -
    Emanuel 4.4.14 p. 117
    
    """
    e = wv * p / (c.eps + wv);
    denom = (17.67 / np.log(e / 611.2)) - 1.;
    Td = 243.5 / denom;
    Td = Td + 273.15;
    return Td


def esat(Temp):
    """
    esat(Temp)

    Calculates the saturation water vapor pressure over a flat
    surface of water at temperature 'T'.

    Parameters
    - - - - - -
    Temp : float or array_like
        Temperature of parcel (K).

    Returns
    - - - -
    esatOut : float or list
        Saturation water vapour pressure (Pa).

    Examples
    - - - - -
    >>> test.assert_almost_equal(esat(300.),3534.5196,decimal=3)
    >>> np.allclose(esat([300., 310.]),[3534.519, 6235.532])
    True

    References
    - - - - - -
    Emanuel 4.4.14 p. 117
      
    """
    # determine if Temp has been input as a vector
    is_scalar=True
    if cbook.iterable(Temp):
        is_scalar = False
    Temp=np.atleast_1d(Temp)
    Tc = Temp - c.Tc
    esatOut = 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))
    # if T is a vector
    if is_scalar:
        esatOut = esatOut[0]
    return esatOut

    
def LCLfind(Td, T, p):
    """
    LCLfind(Td, T, p)

    Finds the temperature and pressure at the lifting condensation
    level (LCL) of an air parcel.

    Parameters
    - - - - - -
    Td : float
        Dewpoint temperature (K).
    T : float
        Temperature (K).
    p : float
        Pressure (Pa)

    Returns
    - - - -
    Tlcl : float
        Temperature at the LCL (K).
    plcl : float
        Pressure at the LCL (Pa).

    Raises
    - - - -
    NameError
        If the air is saturated at a given Td and T (ie. Td >= T)
    
    Examples
    - - - - -
    >>> [Tlcl, plcl] =  LCLfind(280., 300., 8.e4)
    >>> print [Tlcl, plcl]
    [275.76250387361404, 59518.928699453245]
    >>> LCLfind(300., 280., 8.e4)
    Traceback (most recent call last):
        ...
    NameError: parcel is saturated at this pressure

    References
    - - - - - -
    Emanuel 4.6.24 p. 130 and 4.6.22 p. 129
    
    """
    hit = Td >= T;
    if hit is True:
        raise NameError('parcel is saturated at this pressure');

    e = esat(Td);
    ehPa = e * 0.01; #Bolton's formula requires hPa.
    # This is is an empircal fit from for LCL temp from Bolton, 1980 MWR.
    Tlcl = (2840. / (3.5 * np.log(T) - np.log(ehPa) - 4.805)) + 55.;

    r = c.eps * e / (p - e);
    #disp(sprintf('r=%0.5g',r'))
    cp = c.cpd + r * c.cpv;
    logplcl = np.log(p) + cp / (c.Rd * (1 + r / c.eps)) * \
              np.log(Tlcl / T);
    plcl = np.exp(logplcl);
    #disp(sprintf('plcl=%0.5g',plcl))

    return Tlcl, plcl


def wsat(Temp, press):
    """
    wsat(Temp, press)

    Calculates the saturation vapor mixing ratio of an air parcel.

    Parameters
    - - - - - -
    Temp : float or array_like
        Temperature in Kelvin.
    press : float or array_like
        Pressure in Pa.

    Returns
    - - - -
    theWs : float or array_like 
        Saturation water vapor mixing ratio in (kg/kg).

    Raises
    - - - -
    IOError
        If both 'Temp' and 'press' are array_like.

    Examples
    - - - - -
    >>> test.assert_almost_equal(wsat(300, 8e4),0.02875,decimal=4)
    >>> test.assert_array_almost_equal(wsat([300,310], 8e4),[0.0287, 0.0525],decimal=4)
    >>> test.assert_array_almost_equal(wsat(300, [8e4, 7e4]),[0.0287, 0.0330],decimal=4)
    >>> wsat([300, 310], [8e4, 7e4])
    Traceback (most recent call last):
        ...
    IOError: Can't have two vector inputs.

    """
    is_scalar_temp=True
    if cbook.iterable(Temp):
        is_scalar_temp = False
    is_scalar_press=True
    if cbook.iterable(press):
        is_scalar_press = False
    Temp=np.atleast_1d(Temp)
    press=np.atleast_1d(press)
    if (np.size(Temp) !=1) and (np.size(press) != 1):
        raise IOError, "Can't have two vector inputs."
    es = esat(Temp);
    theWs=(c.eps * es/ (press - es))
    theWs[theWs > 0.060]=0.06
    theWs[theWs < 0.0] = 0.
    if is_scalar_temp and is_scalar_press:
        theWs=theWs[0]
    return theWs


def theta(*args):
    """
    theta(*args)

    Computes potential temperature.
    Allows for either T,p or T,p,wv as inputs.
    

    Parameters
    - - - - - -
    T : float
        Temperature (K).
    p : float
        Pressure (Pa).


    Returns
    - - - -
    thetaOut : float
        Potential temperature (K).


    Other Parameters
    - - - - - - - - -
    wv : float, optional
        Vapour mixing ratio (kg,kg). Can be appended as an argument
        in order to increase precision of returned 'theta' value.
    
    
    Raises
    - - - -
    NameError
        If an incorrect number of arguments is provided.
    
    
    References
    - - - - - -
    Emanuel p. 111 4.2.11


    Examples
    - - - - -
    >>> theta(300., 8.e4) # Only 'T' and 'p' are input.
    319.72798180767984
    >>> theta(300., 8.e4, 0.001) # 'T', 'p', and 'wv' all input.
    319.72309475657323
    
    """
    if len(args) == 2:
        wv = 0;
    elif len(args) == 3:
        wv = args[2];
    else:
        raise NameError('need either T,p or T,p,wv');
    
    T = args[0];
    p = args[1];
    power = c.Rd / c.cpd * (1. - 0.24 * wv);
    thetaOut = T * (c.p0 / p) ** power;
    return thetaOut


def thetaes(Temp, press):
    """
    thetaes(Temp, press)

    Calculates the pseudo equivalent potential temperature of an air
    parcel.

    Parameters
    - - - - - -
    Temp : float
        Temperature (K).
    press : float
        Pressure (Pa).


    Returns
    - - - -
    thetaep : float
        Pseudo equivalent potential temperature (K).


    Notes
    - - -
    It should be noted that the pseudo equivalent potential
    temperature (thetaep) of an air parcel is not a conserved
    variable.


    References
    - - - - - -
    Emanuel 4.7.9 p. 132


    Examples
    - - - - -
    >>> test.assert_almost_equal(thetaes(300., 8.e4),412.9736,decimal=4)
    """
    # The parcel is saturated - prohibit supersaturation with Td > T.
    Tlcl = Temp;
    wv = wsat(Temp, press);
    thetaval = theta(Temp, press, wv);
    power = 0.2854 * (1 - 0.28 * wv);
    thetaep = thetaval * np.exp(wv * (1 + 0.81 * wv) * \
                                (3376. / Tlcl - 2.54))
    #
    # peg this at 450 so rootfinder won't blow up
    #
    if thetaep > 450.:
        thetaep = 450;
    return thetaep




def thetaep(Td, T, p):
    """
    thetaep(Td, T, p)

    Calculates the pseudo equivalent potential temperature of a
    parcel. 


    Parameters
    - - - - - -
    Td : float
        Dewpoint temperature (K).
    T : float
        Temperature (K).
    p : float
        Pressure (Pa).


    Returns
    - - - -
    thetaepOut : float
        Pseudo equivalent potential temperature (K).


    Notes
    - - -
    Note that the pseudo equivalent potential temperature of an air
    parcel is not a conserved variable.


    References
    - - - - - -
    Emanuel 4.7.9 p. 132


    Examples
    - - - - -
    >>> test.assert_almost_equal(thetaep(280., 300., 8.e4),344.998307,decimal=5) # Parcel is unsaturated.
    >>> test.assert_almost_equal(thetaep(300., 280., 8.e4),321.53029,decimal=5) # Parcel is saturated.
    """
    if Td < T:
        #parcel is unsaturated
        [Tlcl, plcl] = LCLfind(Td, T, p);
        wv = wsat(Td, p);
    else:
        #parcel is saturated -- prohibit supersaturation with Td > T
        Tlcl = T;
        wv = wsat(T, p);
    
    # $$$   disp('inside theate')
    # $$$   [Td,T,wv]
    thetaval = theta(T, p, wv);
    power = 0.2854 * (1 - 0.28 * wv);
    thetaepOut = thetaval * np.exp(wv * (1 + 0.81 * wv) \
                                   * (3376. / Tlcl - 2.54));
    #
    # peg this at 450 so rootfinder won't blow up
    #
    if(thetaepOut > 450.):
        thetaepOut = 450;
    return thetaepOut


def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()

