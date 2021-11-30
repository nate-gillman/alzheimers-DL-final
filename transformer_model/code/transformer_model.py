# NOTE: Code stencil from HW4

import sys
import numpy as np
import tensorflow as tf
import transformer_funcs as transformer
from preprocess import get_data

from attenvis import AttentionVis

av = AttentionVis()

class Transformer(tf.keras.Model):
    def __init__(self, max_len, vocab_size):

        super(Transformer, self).__init__()

        self.vocab_size = vocab_size + 1 # account for padding token
        self.max_len = max_len # 318 for unbalanced, 163 for balanced
        self.batch_size = 32

        # 1) Define any hyperparameters and optimizer
        self.hidden_size = 128
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # 2) Define embeddings
        self.embedding_size = 64
        self.E = tf.Variable(tf.random.normal(shape=[self.vocab_size, self.embedding_size], stddev=0.1))

        # Create positional embedding
        self.pos_embed = transformer.Position_Encoding_Layer(self.max_len, self.embedding_size)

        # 3) Define transformer layer
        self.transformer = transformer.Transformer_Block(self.embedding_size, is_decoder=False)

        # 4) Define dense layer(s)
        self.dense1 = tf.keras.layers.Dense(3, activation="softmax")

    
    @tf.function
    def call(self, inputs):
        """
        :param inputs: batched ids corresponding to text
        :return the probabilities as a tensor, [batch_size x num_classes]
        """

        # 1) Add the positional embeddings to original embeddings
        embedding = tf.nn.embedding_lookup(self.E, inputs)
        embedding = self.pos_embed(embedding)

        # 2) Pass embeddings to transformer block
        x = self.transformer(embedding)

        # 3) Pooling layer to get correct output shape
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # 4) Apply dense layer(s) to generate probabilities
        probs = self.dense1(x)

        return probs # shape: [32, 3]

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def loss_function(self, prbs, labels, mask):
        """
		Calculates the model cross-entropy loss after one forward pass
		Please use reduce sum here instead of reduce mean to make things easier in calculating per symbol accuracy.

		:param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
		:param labels:  integer tensor, word prediction labels [batch_size x window_size]
		:param mask:  tensor that acts as a padding mask [batch_size x window_size]
		:return: the loss of the model as a tensor
		"""

		# Note: you can reuse this from rnn_model.
        # masked_labels = tf.multiply(tf.cast(labels, dtype=tf.float32), mask)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, prbs))
    

def train(model, inputs, labels):
    for i in range(0, len(inputs), model.batch_size):
        # get batch
        x = inputs[i:i+model.batch_size]
        y = labels[i:i+model.batch_size]

        # one hot labels
        y_true = np.zeros([len(x), 3])
        for i in range(len(y)):
            y_true[i] = np.array([np.eye(3)[int(y[i])]])

        # create mask
        mask = (x != -1).astype(int)

        # forward pass
        with tf.GradientTape() as tape:
            probs = model.call(x)
            loss = model.loss_function(probs, y_true, mask)
            acc = model.accuracy(probs, y_true)
            if i//model.batch_size % 4 == 0:
                print(f"Training on batch: {i}, loss: {loss}, accuracy: {acc}")

        # use optimizer to apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return None

def test(model, test_inputs, test_labels):
    # one hot labels
    y_true = np.zeros([len(test_inputs), 3])
    for i in range(len(test_labels)):
        y_true[i] = np.array([np.eye(3)[int(test_labels[i])]])
    predictions = model.call(test_inputs)
    return model.accuracy(predictions, y_true)

def main():
    # Usage: $ python transfomer_model.py [-b]
    # Optional -b flag to use dataset with balanced classes
    use_balanced_dataset = False
    if len(sys.argv) > 1 and sys.argv[1] == "-b":
        use_balanced_dataset = True
    
    # STEP 1: Decide what attribute we'll classify with respect to; options are ["AEHEVNT", "AEHCOMM", "CONCAT"]
    description = "AEHCOMM"

    # STEP 2: Load the training data
    print("Running preprocessing...")

    if use_balanced_dataset:
        x_train, y_train = "../data/X_train_S.csv", "../data/y_train_S.csv" # 386 total
        x_test, y_test = "../data/X_test_S.csv", "../data/y_test_S.csv" # 97 total
        # 163 max desc length in train
        # 2 min desc length in train
        # 148 max desc length in test
        # 1 min desc length in test
    else: 
        x_train, y_train = "../data/X_train.csv", "../data/y_train.csv"
        x_test, y_test = "../data/X_test.csv", "../data/y_test.csv"
        # descriptions are lists of varying lengths containing integer representations of clincal interaction descriptions
        # 318 max desc length in train
        # 1 min desc length in train
        # 181 max desc length in test
        # 2 min desc length in test

    train_desc, train_diag, test_desc, test_diag, word_to_id = get_data(x_train, y_train, x_test, y_test, description)

    # STEP 3: Pad train and test descriptions so inputs are equal lengths
    max_len = max(max([len(x) for x in train_desc]), max([len(x) for x in test_desc]))
    train_desc_pad = tf.keras.preprocessing.sequence.pad_sequences(train_desc, maxlen=max_len)
    test_desc_pad = tf.keras.preprocessing.sequence.pad_sequences(test_desc, maxlen=max_len)
    print("Preprocessing complete.")

    # STEP 4: Initialize model
    print("Initializing model...")
    vocab_size = len(word_to_id)
    model = Transformer(max_len, vocab_size)
    print("Model initialized!")

    # STEP 5: Train!
    print("Start training!")
    num_epochs = 20
    for i in range(num_epochs):
        print(f"Training on epoch {i}")
        train(model, train_desc_pad, train_diag)
    print("Training complete!")

    # STEP 6: Test
    test_acc = test(model, test_desc_pad, test_diag)
    print(f"Test accuracy: {test_acc}")
    # Unbalanced -- Training on 5 epochs, test accuracy: 0.5864979028701782
    # Balanced -- Training on 5 epochs, test accuracy: 0.22680412232875824
    # Balanced -- Training on 10 epochs, test accuracy: 0.41237112879753113
    # Balanced -- Training on 20 epochs, test accuracy: 0.41237112879753113

if __name__ == '__main__':
    main()