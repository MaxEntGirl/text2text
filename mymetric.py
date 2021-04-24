from transformers import dataset
from collections import Counter
_CITATION = None
_DESCRIPTION = None
_KWARGS_DESCRIPTION = None


def f1_score(prediction, ground_truth):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



class Universal_Dependency(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features({
                'predictions': datasets.Value('int64' if self.config_name != 'sts-b' else 'float32'),
                'references': datasets.Value('int64' if self.config_name != 'sts-b' else 'float32'),
            }),
            codebase_urls=[],
            reference_urls=[],
            format='numpy'
        )

    def _compute(self, predictions, references):
        return {"accuracy": f1_score(predictions, references)}