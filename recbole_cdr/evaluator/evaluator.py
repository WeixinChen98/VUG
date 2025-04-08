# -*- encoding: utf-8 -*-
# @Time    :   2021/6/25
# @Author  :   Zhichao Feng
# @email   :   fzcbupt@gmail.com

"""
recbole.evaluator.evaluator
#####################################
"""

from recbole.evaluator.register import metrics_dict
from recbole.evaluator.collector import DataStruct
from collections import OrderedDict

from logging import getLogger


class Evaluator(object):
    """Evaluator is used to check parameter correctness, and summarize the results of all metrics.
    """

    def __init__(self, config):
        self.config = config
        self.metrics = [metric.lower() for metric in self.config['metrics']]
        self.metric_class = {}

        for metric in self.metrics:
            self.metric_class[metric] = metrics_dict[metric](self.config)

        self.logger = getLogger()
        self.decimal_place = config['metric_decimal_place']


    def evaluate(self, dataobject: DataStruct):
        """Calculate all the metrics. It is called at the end of each epoch.

        Args:
            dataobject (DataStruct): It contains all the information needed for metrics.

        Returns:
            collections.OrderedDict: Contains overall, overlapped, and non-overlapped evaluation results.
        """
        # Overall evaluation
        result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject)
            result_dict.update(metric_val)

        
        # Overlapped users evaluation
        overlap_result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject, group = "overlap")
            overlap_result_dict.update(metric_val)

        # Non-overlapped users evaluation
        non_overlap_result_dict = OrderedDict()
        for metric in self.metrics:
            metric_val = self.metric_class[metric].calculate_metric(dataobject, group = "non_overlap")
            non_overlap_result_dict.update(metric_val)

        # UGF evaluation
        ugf_result_dict = OrderedDict()
        for key in overlap_result_dict.keys():
            # if key in non_overlap_result_dict:  # Ensure the key exists in both dictionaries
            ugf = abs(overlap_result_dict[key] - non_overlap_result_dict[key])
            ugf_result_dict[key] = round(ugf, self.decimal_place)


        return {
            'overall': result_dict,
            'overlapped': overlap_result_dict,
            'non_overlapped': non_overlap_result_dict, 
            'ugf': ugf_result_dict
        }