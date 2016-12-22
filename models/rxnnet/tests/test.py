import glob
import unittest
#import sys


def run_all_tests():
    all_tests = unittest.TestSuite()

    testfiles = glob.glob('test_*.py')
    mesgs = []
    all_test_mods = []
    for file in testfiles:
        module = file[:-3]
        mod = __import__(module)
        all_test_mods.append(mod)
        if hasattr(mod, 'suite'):
            all_tests.addTest(mod.suite)
    
    """
    if not '-v' in sys.argv:
        SloppyCell.Utility.disable_warnings()
    if not SloppyCell.disable_c:
        print '*' * 80
        print 'Running tests with C compilation enabled.'
        print '*' * 80
        unittest.TextTestRunner(verbosity=2).run(all_tests)
    SloppyCell.ReactionNetworks.Network_mod.Network.disable_c = True
    print '*' * 80
    print 'Running tests with C compilation disabled.'
    print '*' * 80
    """
    unittest.TextTestRunner(verbosity=2).run(all_tests)

    for mod in all_test_mods:
        if hasattr(mod, 'message'):
            print mod.message

if __name__ == '__main__':
    run_all_tests()