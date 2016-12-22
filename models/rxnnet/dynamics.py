"""
Simplify and improve SloppyCell.ReactionNetworks.Dynamics

- integrate:
    * return ndarray-based objects
    * time
    * efficiently resume integration (right now it starts from scratch rather from where it was left off)
    * move trajectory.py here
    * no events
    * multiple integrators (compare performance)
    
"""


class Trajectory(object):
    pass


def integrate(net, times, sens=False, integrator='daskr'):
    """
    
    Input: 
        times: if a list, no dense output; if a tuple, dense output
        sens: 
    """
    pass
    