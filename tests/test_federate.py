import unittest
from veector import Veector

class TestVeector(unittest.TestCase):
    def test_init(self):
        veector = Veector()
        self.assertIsNotNone(veector.db)
        self.assertFalse(veector.use_neural_storage)

    def test_compute(self):
        veector = Veector()
        tensor = [[0, [0, 0, 0], [5, 3], 2], [0, [0, 0, 0], [1, 0, 0], 1], [1, 0, 0], [0, 1, 0], []]
        result = veector.compute(tensor)
        self.assertEqual(result, 8)

    def test_parallel_compute(self):
        veector = Veector()
        operations = [[1, 0, 0], [1, 0, 0]]
        data_list = [[5, 3], [2, 4]]
        results = veector.parallel_compute(operations, data_list)
        self.assertEqual(results[0], 8)
        self.assertEqual(results[1], 6)

    def test_federated_train(self):
        veector1 = Veector(use_neural_storage=True)
        veector2 = Veector(use_neural_storage=True)
        veector1.federated_train([veector2])

if __name__ == "__main__":
    unittest.main()
