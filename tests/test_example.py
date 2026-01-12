"""
Example test file to demonstrate testing structure.
"""

import unittest


class TestExample(unittest.TestCase):
    """Example test class."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def tearDown(self):
        """Tear down test fixtures."""
        pass

    def test_example(self):
        """Example test case."""
        self.assertEqual(1 + 1, 2)

    def test_example_fail(self):
        """Example test case that would fail if uncommented."""
        # Uncomment to see a failing test
        # self.assertEqual(1 + 1, 3)


if __name__ == '__main__':
    unittest.main()