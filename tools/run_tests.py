import unittest
import sys

def main():
    test_args = []
    if len(sys.argv) > 1:
        test_args = ['-p', sys.argv[1]]
    unittest.main(module=None, argv=['unittest', 'discover'] + test_args, verbosity=2)