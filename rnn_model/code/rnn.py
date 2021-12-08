import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_data
import os
import sys

job_id = int(sys.argv[1])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Kills an annoying OS warning!! only helpful when running locally, on CPU

class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class inputs a sentence, and outputs a probability distribution over diagnoses. 

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # initialize vocab_size, embedding_size
        self.vocab_size = vocab_size
        self.embedding_size = 64
        self.hidden_size = 128 # size of hidden states (memory and carry) in LSTM

        # Initialize embeddings
        self.embeddings = tf.Variable(tf.random.normal([self.vocab_size, self.embedding_size], stddev=0.1))

        # Initialize LSTM
        self.LSTM = tf.keras.layers.LSTM(self.hidden_size, return_state=True, return_sequences=True)

        # Dense layer
        self.dense = tf.keras.layers.Dense(3, activation="softmax")
        
        # Choose optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def call(self, inputs):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the diagnosis probabilities as a tensor
        """
        
        # FIRST LAYER: Embedding lookup
        embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
        embeddings = tf.expand_dims(embeddings, axis=0)
        
        # SECOND LAYER: Long short term memory
        all_hidden_states, final_hidden_state, final_cell_state = self.LSTM(embeddings, initial_state=None)

        # THIRD LAYER: Dense layer, with softmax activation
        probs = self.dense(final_hidden_state) # (1, 3)

        return probs

    def loss(self, probs, label):
        """
        Calculates average cross entropy loss of the prediction
        """
        return tf.math.reduce_mean(tf.keras.metrics.sparse_categorical_crossentropy(label, probs))


def train(model : Model, train_inputs, train_labels, epoch):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training)
    :param train_labels: train labels (all labels for training)
    :returns: total average batch
    """

    num_batches = len(train_labels)
    total_avg_loss = 0

    num_correct, num_preds = 0, 0
    for batch_no in range(num_batches):

        train_input = train_inputs[batch_no]
        train_label = train_labels[batch_no]

        with tf.GradientTape() as tape:
            probs = model.call(train_input)

            if bool(tf.math.argmax(probs[0])==train_label):
                num_correct += 1
            num_preds += 1

            avg_loss = model.loss(probs, train_label)

        total_avg_loss += avg_loss

        # back-propagation
        gradients = tape.gradient(avg_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


        if batch_no > 0 and batch_no % 10 == 0 and False:
            print(f"total avg loss after epoch {epoch}, sentence {batch_no} is {total_avg_loss/(batch_no+1)}")
            sys.stdout.flush()

    avg_loss = float(total_avg_loss / (num_batches + 1))
    training_acc = num_correct / num_preds

    return avg_loss, training_acc


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples

    :returns: accuracy
    """
    num_predict_corrections = 0
    num_times_we_guessed_zero = 0

    num_batches = len(test_labels)
    for batch_no in range(num_batches):

        test_input = test_inputs[batch_no]
        test_label = test_labels[batch_no]

        probs = model.call(test_input)[0]
        predicted_diagnosis = tf.argmax(probs)
        #print("predicted, actual = ", predicted_diagnosis, test_labels[batch_no])
        if predicted_diagnosis == test_label:
            num_predict_corrections += 1
        if predicted_diagnosis == 0:
            num_times_we_guessed_zero += 1

    return num_predict_corrections / len(test_labels), num_times_we_guessed_zero / len(test_labels)


def main():

    # STEP 1: Decide what attribute we'll classify with respect to; options are ["AEHEVNT", "AEHCOMM", "CONCAT"]
    description = "AEHCOMM"

    # STEP 2: Load the training data
    if job_id == 0:
        print("TRAINING ON DE-BIASED DATASET, WITH AEHCOMM.")
        x_train, y_train = "../data/X_train_S.csv", "../data/y_train_S.csv"
        x_test, y_test = "../data/X_test_S.csv", "../data/y_test_S.csv"
    elif job_id == 1:
        print("TRAINING ON BIASED DATASET, WITH AEHCOMM.")
        x_train, y_train = "../data/X_train.csv", "../data/y_train.csv"
        x_test, y_test = "../data/X_test.csv", "../data/y_test.csv"     
    train_desc, train_diag, test_desc, test_diag, word_to_id = get_data(x_train, y_train, x_test, y_test, description)

    """
    counts_train = [0,0,0] # gives distribution over counts
    for i in range(len(train_diag)):
        counts_train[train_diag[i]] += 1   
    counts_test = [0,0,0]
    for i in range(len(test_diag)):
        counts_test[test_diag[i]] += 1
    print(counts_test, counts_train)
    """
    
    # STEP 3: Initialize model
    vocab_size = len(word_to_id)
    rnn_model = Model(vocab_size)

    # STEP 4: Train the model!!
    NUM_EPOCHS = 10
    avg_losses, training_accs = [], []
    for epoch in range(NUM_EPOCHS):
        avg_loss, training_acc = train(rnn_model, train_desc, train_diag, epoch)
        avg_losses.append(avg_loss)
        training_accs.append(training_acc)

        # STEP 5: Compute accuracy after every training epoch!!
        #accuracy_test, zero_proportion_test = test(rnn_model, test_desc, test_diag)
        #accuracy_train, zero_proportion_train = test(rnn_model, train_desc, train_diag)
        #print(f"Testing accuracy after epoch {epoch} is {accuracy_test}.")
        #print(f"Training accuracy after epoch {epoch} is {accuracy_train}.")
        #print(f"After this epoch, the model predicted 0 on train with proportion {zero_proportion_train}")
        #test_accuracies.append(accuracy_test)
        #train_accuracies.append(accuracy_train)
        #zero_proportions.append(zero_proportion)

    accuracy_test, _ = test(rnn_model, test_desc, test_diag)
    print("Test accuracy after last epoch: ", accuracy_test)

    print("\n------------------------------------------------")
    print("------  Done with training and testing!!! ------")
    print("------------------------------------------------\n")

    print("avg_losses = ", str(avg_losses))
    #print("test_accuracies = ", str(accuracies))
    print("train_accuracies = ", str(training_accs))
    #print("zero_proportions = ", str(zero_proportions))

    return None
    

if __name__ == '__main__':
    main()