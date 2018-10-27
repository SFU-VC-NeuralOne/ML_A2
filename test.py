import random
import unittest
import numpy as np
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def test_something(self):
        number = np.arange(0, 10)
        for i in range (0,10):

            sgd_order = random.choice(number)
            print('sgd order',sgd_order)
            print('before deleting',number)
            number = np.delete(number, (number == sgd_order).nonzero())
            print('after',number)




if __name__ == '__main__':
    unittest.main()
