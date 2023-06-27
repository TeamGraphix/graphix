import unittest
import numpy as np
from graphix.kraus import to_kraus


class TestKraus(unittest.TestCase):
    def test_to_kraus_fail(self):
        # data type is invalid
        with self.assertRaises(TypeError):
            to_kraus(1)
        with self.assertRaises(TypeError):
            to_kraus("hello")
        with self.assertRaises(TypeError):
            to_kraus({})

        # (i) single unitary matrix A
        A = [[0, 1, 2], [3, 4, 5]]
        with self.assertRaises(ValueError):
            to_kraus(np.asarray(A, dtype=complex))
        with self.assertRaises(ValueError):
            to_kraus(A)
        with self.assertRaises(ValueError):
            to_kraus([])

        # (ii) single Kraus set
        with self.assertRaises(ValueError):
            to_kraus([[np.asarray(A, dtype=complex)]])
        with self.assertRaises(ValueError):
            to_kraus([[A]])
        with self.assertRaises(ValueError):
            to_kraus([[]])

        # (iii) generalized Kraus set
        with self.assertRaises(ValueError):
            to_kraus([[np.asarray(A, dtype=complex)], [np.asarray(A, dtype=complex)]])
        with self.assertRaises(ValueError):
            to_kraus([[A], [A]])
        with self.assertRaises(ValueError):
            to_kraus([[A], []])
        with self.assertRaises(ValueError):
            to_kraus([[A], [[]]])
        with self.assertRaises(ValueError):
            to_kraus([[], [A]])
        with self.assertRaises(ValueError):
            to_kraus([[[]], [A]])
        with self.assertRaises(ValueError):
            to_kraus([[], []])
        with self.assertRaises(ValueError):
            to_kraus([[A], [A, A]])

    def test_to_kraus_success(self):
        # (i) single unitary matrix A
        A = [[0, 1], [2, 3]]
        kraus = to_kraus(A)
        np.testing.assert_array_equal(kraus[0][0], np.asarray(A, dtype=complex))
        self.assertEqual(kraus[1], None)

        kraus = to_kraus(np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][0], np.asarray(A, dtype=complex))
        self.assertEqual(kraus[1], None)

        # (ii) single Kraus set
        B = [[4, 5], [6, 7]]
        kraus = to_kraus([A, B])
        np.testing.assert_array_equal(kraus[0][0], np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][1], np.asarray(B, dtype=complex))
        self.assertEqual(kraus[1], None)

        kraus = to_kraus([np.asarray(A, dtype=complex), np.asarray(B, dtype=complex)])
        np.testing.assert_array_equal(kraus[0][0], np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][1], np.asarray(B, dtype=complex))
        self.assertEqual(kraus[1], None)

        # (iii) generalized Kraus set
        C = [[8, 9], [10, 11]]
        D = [[12, 13], [14, 15]]
        kraus = to_kraus([[A, B], [C, D]])
        np.testing.assert_array_equal(kraus[0][0], np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][1], np.asarray(B, dtype=complex))
        np.testing.assert_array_equal(kraus[1][0], np.asarray(C, dtype=complex))
        np.testing.assert_array_equal(kraus[1][1], np.asarray(D, dtype=complex))


if __name__ == "__main__":
    unittest.main()
