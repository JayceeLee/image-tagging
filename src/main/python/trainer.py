
import dataset
import tensorflow as tf
from sklearn import metrics

def do_training(
    training_features,
    training_labels,
    validation_features,
    validation_labels,
    classes_size,
    batch_size = 100,
    hidden_units = [100, 100],
    learning_rate = 0.05,
    steps = 1000,
    periods = 5):

  steps_per_period = steps / periods

  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
    training_features)
  classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    n_classes=classes_size,
    hidden_units=hidden_units,
    optimizer=tf.train.AdagradOptimizer(learning_rate=learning_rate),
    gradient_clip_norm=5.0
  )
  
  training_errors = []
  validation_errors = []
  print("Training model...")
  # Note that we track the model's error using the function called
  # logistic loss. This is not to be confused with the loss function.
  # LinearClassifier defines its own loss function internally.
  print("LogLoss:")
  for period in range (0, periods):
    classifier.fit(
      training_features,
      training_labels,
      steps=steps_per_period,
      batch_size=batch_size
    )
    predictions_training = list(classifier.predict_proba(
      training_features, as_iterable=True))
    predictions_validation = list(classifier.predict_proba(
      validation_features, as_iterable=True))

    log_loss_training = metrics.log_loss(
      training_labels, predictions_training)
    log_loss_validation = metrics.log_loss(
      validation_labels, predictions_validation)
    training_errors.append(log_loss_training)
    validation_errors.append(log_loss_validation)
    print("  period %02d : %3.2f" % (period, log_loss_training))
  
  final_predictions = list(classifier.predict(
    validation_features, as_iterable=True))
  accuracy_validation = metrics.accuracy_score(
    validation_labels, final_predictions)
  print("Final accuracy (on validation data): %0.2f" % accuracy_validation)
  return classifier

def main():
  print("Loading data...")
  indices_to_tags, all_images, all_tags = dataset.load_data('/Users/ashmore/NetBeansProjects/ImageTagging/output/')
  print("Done!")
  print(indices_to_tags)
  validation_size = int(len(all_images)*.3)
  training_features = all_images[:-validation_size]
  training_labels = all_tags[:-validation_size]
  validation_features = all_images[validation_size:]
  validation_labels = all_tags[validation_size:]
  classes_size = len(indices_to_tags)
  do_training(training_features, training_labels, validation_features, validation_labels, classes_size)

if __name__ == '__main__':
  main()
