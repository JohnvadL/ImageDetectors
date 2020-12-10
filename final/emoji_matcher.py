import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, log_loss, accuracy_score

class EmojiMatcher(object):

  def __init__(self, label_path, metric):
    """
    Args:
      label_path: path to csv containing emoji labels
      metric: scoring metric to use; one of 'acc, mse, bce, kappa'
    """
    label_df = pd.read_csv('emoji_labels.csv')
    self.emoji_ids = label_df['unicode']
    self.labels = label_df.iloc[:, 1:].to_numpy()
    metrics = {'acc': lambda y1, y2: accuracy_score(y2, y1 > 0.5),
               'mse': lambda y1, y2: np.dot(y1-y2, y1-y2)/18,
               'bce': lambda y1, y2: log_loss(y2, y1, labels=[0,1]),
               'kappa': lambda y1, y2: cohen_kappa_score(y1 > 0.5, y2)}
    try:
      self.metric = metrics[metric]
    except KeyError:
      raise ValueError('{} is not a valid metric. Choose from {}.'.format(metric, list(metrics.keys())))
    
  def _score(self, predicted, labels):
    return self.metric(predicted, labels)

  def predict(self, model_out):
  	"""
  	Output the emoji with the highest score according to the 
  	defined metric.
  	"""
    scores = [self._score(model_out, e_label) for e_label in self.labels]
    return self.emoji_ids[np.argmax(scores)]

if __name__ == '__main__':
	sample_pred = np.array([0.95, 0.1 , 0, 0, 0.7, 0, 0, 0.2, 0.8 , 0, 0, 0, 0.3, 0, 0, 0, 0, 0]) # U+1F60A
	matcher = EmojiMatcher('emoji_labels.csv', 'kappa')
	matcher.predict(sample_pred)