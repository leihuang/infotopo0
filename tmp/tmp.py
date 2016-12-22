import numpy as np
import pandas as pd
import copy
import cPickle

"""
class Foo(object):
    #@property
    #def _constructor(self):
    #    return Foo
    
    def __init__(self, df):
        self._ = df
        
    #def __getattr__(self, attr):
    #    return getattr(self._, attr)
    
    
df = pd.DataFrame(np.random.randint(0,2,(3,2)), columns=['A','B'])
foo = Foo(df)
#foo_cp = copy.deepcopy(foo)

foo.r = 0.5

fh = open('tmp', 'w')
cPickle.dump(foo, fh)
fh.close()

fh2 = open('tmp')
foo2 = cPickle.load(fh2)
fh2.close()

# two problems: copy and serialization...

"""

class Door:
    colour = 'brown'

    def __init__(self, number, status):
        self.number = number
        self.status = status

    @classmethod
    def knock(cls):
        print("Knock!")

    @classmethod
    def paint(cls, colour):
        cls.colour = colour

    def open(self):
        print 'open'
        
    def close(self):
        self.status = 'closed'


class SecurityDoor:
    locked = True
    
    def __init__(self, number, status):
        self.door = Door(number, status)
        
    def open(self):
        print "here" 
        
    def __getattr__(self, attr):
        return getattr(self.door, attr)
    
sd = SecurityDoor(1, 'open')
sd.open()    