import unittest
import numpy as np
from graphix.kraus import to_kraus, _is_kraus_op


class TestKraus(unittest.TestCase):
    def test_to_kraus_fail(self):
        A_wrong = [[0, 1, 2], [3, 4, 5]]
        A = [[0, 1], [2, 3]]

        # data type is invalid
        with self.assertRaises(TypeError):
            to_kraus(1)
        with self.assertRaises(TypeError):
            to_kraus("hello")
        with self.assertRaises(TypeError):
            to_kraus({})

        # (i) single unitary matrix A
        with self.assertRaises(ValueError):
            to_kraus([np.asarray(A_wrong, dtype=complex), 1])
        with self.assertRaises(ValueError):
            to_kraus([A_wrong, 1])
        with self.assertRaises(ValueError):
            to_kraus([A, 1j])

        # (ii) single Kraus set
        with self.assertRaises(ValueError):
            to_kraus([[np.asarray(A_wrong, dtype=complex)]])
        with self.assertRaises(ValueError):
            to_kraus([[A_wrong]])
        with self.assertRaises(ValueError):
            to_kraus([[]])
        with self.assertRaises(ValueError):
            to_kraus([A, A])
        with self.assertRaises(ValueError):
            to_kraus([[A, A], [A, A]])
        with self.assertRaises(AssertionError):
            to_kraus([[A, 1], [A, A]])

        # (iii) generalized Kraus set
        with self.assertRaises(ValueError):
            to_kraus([[], 1])
        with self.assertRaises(ValueError):
            to_kraus([[[A, 1]], [1]])
        with self.assertRaises(ValueError):
            to_kraus(np.array([[[A, 1]], [1]]))
        with self.assertRaises(ValueError):
            to_kraus(np.array([[[A, 1]], [[A, A]]]))
        with self.assertRaises(ValueError):
            to_kraus(np.array([[[A, A]], [[A, A]]]))

    def test_to_kraus_success(self):
        # (i) single unitary matrix A
        A = [[0, 1], [2, 3]]
        kraus = to_kraus((A, 1))
        np.testing.assert_array_equal(kraus[0][0].data, np.asarray(A, dtype=complex))
        self.assertEqual(kraus[0][0].qarg, 1)
        self.assertEqual(kraus[1], None)

        kraus = to_kraus((np.asarray(A, dtype=complex), 1))
        np.testing.assert_array_equal(kraus[0][0].data, np.asarray(A, dtype=complex))
        self.assertEqual(kraus[0][0].qarg, 1)
        self.assertEqual(kraus[1], None)

        # (ii) single Kraus set
        B = [[4, 5], [6, 7]]
        kraus = to_kraus([(A, 1), (B, 2)])
        np.testing.assert_array_equal(kraus[0][0].data, np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][1].data, np.asarray(B, dtype=complex))
        self.assertEqual(kraus[0][0].qarg, 1)
        self.assertEqual(kraus[0][1].qarg, 2)
        self.assertEqual(kraus[1], None)

        kraus = to_kraus([(np.asarray(A, dtype=complex), 1), (np.asarray(B, dtype=complex), 2)])
        np.testing.assert_array_equal(kraus[0][0].data, np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][1].data, np.asarray(B, dtype=complex))
        self.assertEqual(kraus[0][0].qarg, 1)
        self.assertEqual(kraus[0][1].qarg, 2)
        self.assertEqual(kraus[1], None)

        # (iii) generalized Kraus set
        C = [[8, 9], [10, 11]]
        D = [[12, 13], [14, 15]]
        kraus = to_kraus([[(A, 1), (B, 2)], [(C, 3), (D, 4)]])
        np.testing.assert_array_equal(kraus[0][0].data, np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[0][1].data, np.asarray(B, dtype=complex))
        np.testing.assert_array_equal(kraus[1][0].data, np.asarray(C, dtype=complex))
        np.testing.assert_array_equal(kraus[1][1].data, np.asarray(D, dtype=complex))

    def test__is_kraus_op_fail(self):
        np.testing.assert_equal(_is_kraus_op(1), False)
        np.testing.assert_equal(_is_kraus_op("hello"), False)
        np.testing.assert_equal(_is_kraus_op([]), False)
        np.testing.assert_equal(_is_kraus_op([[], []]), False)
        np.testing.assert_equal(_is_kraus_op([[0, 1, 2], [3, 4, 5]]), False)

    def test__is_kraus_op_success(self):
        A = [[0, 1], [2, 3]]
        np.testing.assert_equal(_is_kraus_op((A, 1)), True)
        np.testing.assert_equal(_is_kraus_op((np.asarray(A, dtype=complex), 1)), True)


if __name__ == "__main__":
    unittest.main()
