"""
geodesics, curvature(?)

shape
"""

import numpy as np
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d.axes3d as p3
#from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits

from infotopo import geodesic
reload(geodesic)



from infotopo.models import twoexp 
#reload(twoexp)
#from twoexp import get_f

f = twoexp.get_f(1,2,3)
p0 = [1,1]

def plot_image_transition(f, decade=4, npt=500, pts=None, xyzlabels=['','',''], 
               filepath='', lam=0,
               color='b', alpha=0.5, shade=False, edgecolor='none', 
               **kwargs_surface):
    """
    Plot the image of the predict function, aka "models manifold".
    
    Input:
        decade: how many decades to cover
        npt: number of points for each parameter
        pts: a list of 3-tuples for the points to be marked
    """
    ps = [np.logspace(np.log10(p0_i)-decade/2, np.log10(p0_i)+decade/2, npt) 
          for p0_i in p0]
    pss = np.meshgrid(*ps)
    
    # make a dummy function that takes in the elements of an input vector 
    # as separate arguments
    def _f(*p):
        return f(p)
    
    def eye(p):
        return np.array([p[0], p[1], p[0]*0])
    
    def _eye(*p):
        return eye(p)
    
    yss = _f(*pss) + _eye(*pss)*lam
    if len(yss) > 3:
        #yss = pca(yss, k=3)
        xyzlabels = ['PC1', 'PC2', 'PC3']
        pass
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")
    ax.plot_surface(*yss, color=color, alpha=alpha, shade=shade, 
                    edgecolor=edgecolor, **kwargs_surface)
                    
    if pts is not None:
        ax.scatter(*np.array(pts).T, color='r', alpha=1)
        
    ax.set_xlabel(xyzlabels[0])
    ax.set_ylabel(xyzlabels[1])
    ax.set_zlabel(xyzlabels[2])
    
    ax.set_xlim(0,100)
    ax.set_zlim(0,100)
    ax.set_zlim(0,100)

    plt.show()
    plt.savefig(filepath)
    plt.close()

#plot_image_transition(f, lam=1)
"""
from matplotlib.widgets import Slider, Button, RadioButtons

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plt.subplots_adjust(left=0.25, bottom=0.25)

def calculate_and_plot(ax, a,f):
    xs = np.linspace(0.0, 1.0, 201)
    ys = np.linspace(0.0, 1.0, 201)
    xss, yss = np.meshgrid(xs, ys)
    zss = a0*np.sin(2*np.pi*f0*xss)

    ax.plot_surface(xss, yss, zss, color='red')
    #plt.draw()
    #return surf
#plt.axis([0, 1, -10, 10])

a0 = 5
f0 = 3
#global surf
calculate_and_plot(ax, a0,f0)

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axamp  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

pslider = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)

def update(ax, val):
    #global surf
    #surf.remove()
    a = samp.val
    f = sfreq.val
    ax.clear()
    calculate_and_plot(ax, a, f)
    plt.draw()


sfreq.on_changed(lambda val: update(ax, val))
samp.on_changed(lambda val: update(ax, val))





resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()
"""


xss, yss = np.meshgrid(np.arange(0,2*np.pi,0.2), np.arange(0,2*np.pi,0.2))

U = np.cos(xss)
V = np.sin(yss)

#1
fig = plt.figure()
ax = fig.add_subplot(111)
Q = ax.quiver(U, V)
qk = ax.quiverkey(Q, 0.5, 0.92, 2, 'a', labelpos='W',
                  fontproperties={'weight': 'bold'})
#l,r,b,t = ax.axis()
#dx, dy = r-l, t-b
#ax.axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])

plt.show()