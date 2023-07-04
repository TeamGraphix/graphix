import unittest
import numpy as np
from graphix.kraus import to_kraus, generate_dephasing_kraus, _is_kraus_op


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

    def test_to_kraus_success(self):
        # (i) single unitary matrix A
        A = [[0, 1], [2, 3]]
        kraus = to_kraus((A, 1))
        np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
        self.assertEqual(kraus[0].qarg, 1)

        kraus = to_kraus((np.asarray(A, dtype=complex), 1))
        np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
        self.assertEqual(kraus[0].qarg, 1)

        # (ii) single Kraus set
        B = [[4, 5], [6, 7]]
        kraus = to_kraus([(A, 1), (B, 2)])
        np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[1].data, np.asarray(B, dtype=complex))
        self.assertEqual(kraus[0].qarg, 1)
        self.assertEqual(kraus[1].qarg, 2)

        kraus = to_kraus([(np.asarray(A, dtype=complex), 1), (np.asarray(B, dtype=complex), 2)])
        np.testing.assert_array_equal(kraus[0].data, np.asarray(A, dtype=complex))
        np.testing.assert_array_equal(kraus[1].data, np.asarray(B, dtype=complex))
        self.assertEqual(kraus[0].qarg, 1)
        self.assertEqual(kraus[1].qarg, 2)

    def test_generate_dephasing_kraus_fail(self):
        with self.assertRaises(AssertionError):
            generate_dephasing_kraus(2, 1)
        with self.assertRaises(AssertionError):
            generate_dephasing_kraus(0.5, "1")

    def test_generate_dephasing_kraus_success(self):
        p = 0.5
        qarg = 1
        dephase_kraus = generate_dephasing_kraus(p, qarg)
        np.testing.assert_array_equal(dephase_kraus[0].data, np.asarray(np.sqrt(1 - p) * np.eye(2), dtype=complex))
        np.testing.assert_array_equal(dephase_kraus[1].data, np.asarray(np.sqrt(p) * np.diag([1, -1]), dtype=complex))
        self.assertEqual(dephase_kraus[0].qarg, qarg)
        self.assertEqual(dephase_kraus[1].qarg, qarg)

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
