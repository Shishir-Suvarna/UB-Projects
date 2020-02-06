import tensorflow as tf
import csv
import os.path
'''
class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step)
'''
class CsvLogger:
    def __init__(self, filepath, filename="log.csv", data=None):
        self.log_path = filepath
        self.log_name = filename
        self.csv_path = os.path.join(self.log_path, self.log_name)
        self.fieldsnames = ['xy_loss','wh_loss', 'c_loss', 'classProb_loss','noObj_loss']
        with open(self.csv_path, 'w') as f:
          writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
          writer.writeheader()

    def scalar_summary(self, loss):
        data = {}
        for tag, value in loss.items():
          data[tag] = value
        if data is not None:
          self.write(data)

    def write(self, data):
        with open(self.csv_path, 'a') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldsnames)
            writer.writerow(data)
