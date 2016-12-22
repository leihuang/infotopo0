import unittest
#from examples.revrxn import net
#from examples import revrxn
from infotopo.models.rxnnet.examples.revrxn import net as net_revrxn
from infotopo.models.rxnnet.examples.cycle import net as net_cycle
from infotopo.models.rxnnet.examples.mar2 import net as net_mar2 



class TestMCA(unittest.TestCase):
    
    def test_basic(self):
        
        global net_revrxn
        global net_cycle
        global net_mar2
        
        net_revrxn = net_revrxn.copy()
        net_mar2 = net_mar2.copy()
        
        net_mar2.update(k1=1, k2=1, A1=1, A2=0)
        self.assertEqual(net_mar2.s.tolist(), [0.5])
        self.assertEqual(net_mar2.J.tolist(), [0.5, 0.5]) 

        net_mar2.update(k1=3, k2=1, A1=2, A2=1)
        self.assertEqual(net_mar2.s.tolist(), [1.75])
        self.assertEqual(net_mar2.J.tolist(), [0.75, 0.75]) 

        
        
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
suite = unittest.makeSuite(TestMCA)

if __name__ == '__main__':
    unittest.main()
