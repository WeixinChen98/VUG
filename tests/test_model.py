import os
import unittest

from recbole_cdr.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


def quick_test(config_dict):
    objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=False)


class TestRecommender(unittest.TestCase):
    def test_vug(self):
        config_dict = {
            'model': 'VUG',
            'train_epochs': ["BOTH:1"],
        }
        quick_test(config_dict)



if __name__ == '__main__':
    unittest.main()
