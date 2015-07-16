import numpy as np
import pandas as pd
import copy
import cPickle

## FIXME ***: 
# what is happening here??

def f(p1, p2):
    return p1 + p2

p1s = [1,2,3]
p2s = [10,20,30]
p1ss, p2ss = np.meshgrid(p1s, p2s)

a

from util2 import butil
reload(butil)
Series, DF = butil.Series, butil.DF

# TypeError
df = DF([[1,2],[2,3]])
reload(butil)
df = DF([[1,2],[2,3]])  # TypeError: super(type, obj): obj must be an instance or subtype of type

# good
df = butil.DF([[1,2],[2,3]])
reload(butil)
df = butil.DF([[1,2],[2,3]])




## FIXME ***: 
# post a solution on stackoverflow: 
# http://stackoverflow.com/questions/29569005/error-in-copying-a-composite-object-consisting-mostly-of-pandas-dataframe
# detailing the problems
# outlining the solution (subclassing)
# update github
# provide the link to the solution



class Foo(object):
    """
    Foo is composed mostly of a pd.DataFrame, and behaves like it too. 
    """
    def __init__(self, df, attr_custom=None):
        self._ = df
        self.attr_custom = attr_custom

    # the following code allows Foo objects to behave like pd.DataFame,
    # and I want to keep this behavior.
    def __getattr__(self, attr):
        out = getattr(self._, attr)
        if callable(out):
            out = _wrap(out)
        return out
    
    def __repr__(self):
        return self._.__repr__()


def _wrap(f):
    def f_wrapped(*args, **kwargs):
        out = f(*args, **kwargs)
        try:
            return Foo(out)
        except:
            return out
    return f_wrapped


df = pd.DataFrame([[1,2,3],[4,5,6]])
ser = pd.Series([1,2,3])
foo = Foo(df, 1)

## foo can't be pickled, but can when not overriding __getattr__
fh = open('tmp', 'w')
cPickle.dump(foo, fh)
fh.close()

#fh = open('tmp')
#foo2 = cPickle.load(fh)
#fh.close()

## foo can't be copied, but can when not overriding __getattr__
#foo_cp = copy.deepcopy(foo)
#foo_cp = foo.copy()

## custom attributes of pd.DataFrame can't be serialized
df.a = 1

fh = open('tmp', 'w')
cPickle.dump(df, fh)
fh.close()

fh = open('tmp')
df = cPickle.load(fh)
fh.close()

print hasattr(df, 'a')

## but it can be easily bypassed by writing some functions
def dump(df, filepath):
    fh = open(filepath, 'w')
    cPickle.dump((df, df.__dict__), fh)
    fh.close()
    
def load(df, filepath):
    fh = open(filepath)
    df, attrs = cPickle.load(fh)
    fh.close()
    df.__dict__ = attrs
    return df

df.a = 1
dump(df, 'tmp')
df = load(df, 'tmp')

print hasattr(df, 'a')

"""
def f(b=2):
    a = 1
    print "in f"
    
    def g(x):
        print "in g"
        return x+a
    return g

g = f()
"""

