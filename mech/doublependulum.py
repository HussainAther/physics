
def positionDoublePendulum(y, x):
    """
    theta1 -> y[0]
    theta2 -> y[1]
    p1 -> y[2]
    p2 -> y[3]
    """
    t1,t2 = y[0],y[1]
    p1,p2 = y[2],y[3]
    cs = cos(t1-t2)
    ss = sin(t1-t2)
    tt = 1./(1+ss**2)
    c1 = p1*p2*ss*tt
    c2 = (p1**2+2*p2**2-2*p1*p2*cs)*cs*ss*tt**2
    return array([ (p1-p2*cs)*tt, (2*p2-p1*cs)*tt, -2*sin(t1)-c1+c2, -sin(t2)+c1-c2])

def energyDoublePendulum(y):
    t1,t2 = y[0],y[1]
    p1,p2 = y[2],y[3]
    cs = cos(t1-t2)
    ss = sin(t1-t2)
    tt = 1./(1+ss**2)
    return 0.5*(p1**2 + 2*p2**2 - 2*p1*p2*cs)*tt + (3.-2.*cos(t1)-cos(t2))
