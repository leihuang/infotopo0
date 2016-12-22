import unittest
import numpy as np
#from examples.revrxn import net
#from examples import revrxn
from infotopo.models.rxnnet.examples.mar2 import net as net_mar2
from infotopo.models.rxnnet import model


class TestModel(unittest.TestCase):
    
    def test_basic(self):
        
        global net_mar2

        net_mar2 = net_mar2.copy()
        
        k1, A1 = np.random.lognormal(size=2)
        
        net_mar2.update(k1=k1, A1=A1)
        self.assertEqual(net_mar2.vals.k1, k1)
        self.assertEqual(net_mar2.vals.A1, A1)

        
        """
        # need to add a new parameter to change the network structure and
        # force it to recompile
        local_net.addParameter(id='dummy_par',value=1.0)
        local_net.disable_deriv_funcs()
        local_net.compile()

        funcs_no_derivs = ['res_function', 'alg_deriv_func', 'alg_res_func',\
                           'integrate_stochastic_tidbit', 'root_func']
        self.assertEqual(local_net._dynamic_funcs_python.keys(),
                         funcs_no_derivs)
        traj = Dynamics.integrate(local_net, tlist)

        self.assertAlmostEqual(traj.get_var_val('X0',4.8), 
                               0.618783392, 5)
        self.assertAlmostEqual(traj.get_var_val('X1',21.6), 
                               0.653837775, 5)
        """
suite = unittest.makeSuite(TestModel)

if __name__ == '__main__':
    unittest.main()
