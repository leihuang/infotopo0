import unittest
#from examples.revrxn import net
#from examples import revrxn
from infotopo.models.rxnnet.examples.revrxn import net as net_revrxn
from infotopo.models.rxnnet.examples.cycle import net as net_cycle



class TestStructure(unittest.TestCase):
    
    def test_basic(self):
        
        global net_revrxn
        global net_cycle
        
        net_revrxn = net_revrxn.copy()
        
        self.assertEqual(net_revrxn.P.values.tolist(), [[1.0, 1.0]]) 

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
suite = unittest.makeSuite(TestStructure)

if __name__ == '__main__':
    unittest.main()