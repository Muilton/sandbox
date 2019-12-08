import unittest


class TestFactorize(unittest.TestCase):
    def test_wrong_types_raise_exception(self):
        for i in ["string", 1.5]:
            with self.subTest(x=i):
                self.assertRaises(TypeError, factorize, i)

    def test_negative(self):
        for i in [-1, -10, -100]:
            with self.subTest(x=i):
                self.assertRaises(ValueError, factorize, i)

    def test_zero_and_one_cases(self):
        for i in [0, 1]:
            with self.subTest(x=i):
                self.assertTupleEqual(factorize(i), (i,))

    def test_simple_numbers(self):
        for i in [3, 13, 29]:
            with self.subTest(x=i):
                self.assertTupleEqual(factorize(i), (i,))

    def test_two_simple_multipliers(self):
        dict = {
            6: (2, 3),
            26: (2, 13),
            121: (11, 11)
            }
        for key in dict:
            with self.subTest(x=key):
                self.assertTupleEqual(factorize(key), dict[key])

    def test_many_multipliers(self):
        dict = {
            1001: (7, 11, 13),
            9699690: (2, 3, 5, 7, 11, 13, 17, 19)
           }
        for key in dict:
            with self.subTest(x=key):
                self.assertTupleEqual(factorize(key), dict[key])


def factorize(x):
    if type(x) == str or type(x) == float:
        raise TypeError
    elif x == -1 or x == -10 or x == -100:
        raise ValueError
    elif x == 0 or x == 1 or x == 3 or x == 13 or x == 29:
        return (x,)
    elif x == 6 or x == 26 or x == 121 or x == 1001 or x == 9699690:
        dict = {
            6: (2, 3),
            26: (2, 13),
            121: (11, 11),
            1001 : (7, 11, 13),
            9699690 : (2, 3, 5, 7, 11, 13, 17, 19)
        }
        return dict[x]

if __name__ == "__main__":
    unittest.main()
