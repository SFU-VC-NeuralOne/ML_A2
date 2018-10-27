import unittest
import matplotlib.pyplot as plt


class MyTestCase(unittest.TestCase):
    def test_something(self):
        plt.plot([20, 30], [40, 50], 'ko-')
        plt.show()



if __name__ == '__main__':
    unittest.main()
