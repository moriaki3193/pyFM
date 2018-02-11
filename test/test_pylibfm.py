# -*- coding: utf-8 -*-
import unittest
import numpy as np
from scipy import sparse as sp
from pyfm import pylibfm


class TestFM(unittest.TestCase):

    def setUp(self):
        self.model = pylibfm.FM()
        self.features = sp.csr_matrix(np.matrix([
           #  Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
           # A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
            [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
            [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
            [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
            [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
            [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
            [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
            [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
        ]))
        self.target = [5, 3, 1, 4, 5, 1, 5]

    def test_model_prediction(self):
        # [START making a test data]
        X_test = sp.csr_matrix(np.matrix([
            [1, 0, 0,  0,  0,  0,  1,   0.3, 0.3, 0.3, 0,     17,   0,  0,  1,  0 ],
        ]))
        y_test = [1]
        # [END making a test data]
        self.model.fit(self.features, self.target)
        y_pred = self.model.predict(X_test)
        print(y_pred)
        print(y_test)
        self.assertTrue(isinstance(y_pred[0], np.float64))


if __name__ == '__main__':
    unittest.main()