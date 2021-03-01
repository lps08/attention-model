# Neural machine translation with attention model PTBR-TO-ENGLISH
> Modelo treinado para fazer a tradução de frases em português para o inglês.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Tensorflow 2](https://img.shields.io/badge/Tensorflow-2.3-orange)](https://www.tensorflow.org/install?hl=pt-br)

The sequence-to-sequence model is a model commonly used to do translation tasks, where it can be called 'machine translation'. Thus, this model has an approach in which machine learning models are used to obtain a sequence as input, in a specific way, and convert this sequence to a representation in another form. Text translation is a good example of this powerful feature that the sequence-to-sequence model can offer.

From the seq2seq model, the attention mechanism was added, which are input processing techniques for neural networks that allow the network to focus on specific aspects of a complex entry, one at a time, until the entire set of data to be categorized. The goal is to divide complicated tasks into smaller areas of attention that are processed sequentially. Similar to how the human mind solves a new problem, dividing it into simpler tasks and solving them one by one. In the figure below, the attention model used in this project can be seen:

![bahdanau](https://blog.floydhub.com/content/images/2019/09/Slide38.JPG)

Project inspired by the implementation of the [tensorflow tutorial section](https://www.tensorflow.org/tutorials/text/nmt_with_attention) .

## Data preprocessing

Conversion to ASCII must be done first

```py
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
```
After defining the function to transform the sentence into ASCII, a new function with the objective of removing special characters and adding the start and end tags, is implemented. The tags for both the beginning and the end are intended to show the neural networks where the sentences start and end. The given function can be shown below:

```py
def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w
```
Note that for each sentence, before the sentence is returned, the tags are added.

The next step after doing these preprocessing, is to add tokens for each word in the dataset. The tokenization process is a way of separating a piece of text into smaller units called tokens. Here, tokens can be words, characters or subwords. Therefore, tokenization can be broadly classified into 3 types - word, character and subword tokenization.

The tokenization process can be defined according to the function below, using the Tokenizer class from the keras library:
```py
def tokenize(lang, max_len):
    #Call the Tokenizer class
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    #Fits for eatch setense
    lang_tokenizer.fit_on_texts(lang)
    
    #Transforming the sentenses in a text sequence
    tensor = lang_tokenizer.texts_to_sequences(lang)
    
    #Padding tensor to have the same lenght
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post', maxlen=max_len) # fix size value to all sentense
 
    return tensor, lang_tokenizer
```
## Models used
The models used in this project are: Encoder, One_Step_Decoder, Decoder, Encoder_decoder.

#### Encoder
The task of the encoder model is to read the input sequence and summarize the information in something called internal state vectors or context vector (in the case of LSTM, these are called hidden state and cell state vectors). This context vector aims to encapsulate the information for all input elements, in order to help the decoder to make accurate predictions.

The model can be seen according to the implementation below:
```py
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))
```

#### Attention model

The attention model aims to allow neural networks to approach the visual attention mechanism that humans use. Like people processing a new scene, the model studies a particular point in an image/text with intense focus, while perceiving the surrounding areas in, then adjusts the focal point as the network begins to understand the scene.

The attention model can be seen below:

```py
class Attention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

   # Applying the equation using to calculate the score of attention weights
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
```

#### One_Step_Decoder model

The One_Step_Decoder model can be thought of in two separate processes, training and inference. It is not that they have a different architecture, but they share the same architecture and its parameters. They have different strategies to feed the shared model. The decoder maintains a hidden state that passes from one stage of time to the next.

```py
class One_Step_Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super(One_Step_Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = Attention(self.dec_units)

    def call(self, x, enc_output, hidden):
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights
```

#### Decoder model
The model has the function of having, for each incoming batch, to do the processing according to the One_Step_Decoder Class. After processing for each item in the batch, the results are stored in a tensor and returned.

```py
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_size):
        super().__init__()
        self.batch_size = vocab_size
        #Intialize necessary variables and create an object from the class onestepdecoder
        self.onestep = One_Step_Decoder(vocab_size, embedding_dim, dec_units, self.batch_size)
        
    def call(self, input_to_decoder, encoder_output, state):
        #Store the output in tensorarray
        all_outputs = tf.TensorArray(tf.float32, input_to_decoder.shape[1], name="output_array")
        one_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for step in range(1, input_to_decoder.shape[1]):
            # passing enc_output to the decoder
            output, state, _ = self.onestep(one_input, encoder_output, state)

            all_outputs = all_outputs.write(step, output)

            # using teacher forcing
            dec_input = tf.expand_dims(input_to_decoder[:, step], 1)
        
        all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])
        
        return all_outputs #final sentense
```

#### Encoder-decoder model

This model aims to call both the Encoder model and the Decoder model to perform the training process.

```py
class Encoder_decoder(tf.keras.Model):

    def __init__(self, vocab_inp_size, vocab_tar_size, embedding_dim, output_length, enc_units, dec_units, batch_size):
        super().__init__()
        self.encoder =Encoder(vocab_inp_size, embedding_dim, enc_units, batch_size)
        self.decoder = Decoder(vocab_tar_size, embedding_dim, dec_units, batch_size)

    def call(self, data):
        
        state = self.encoder.initialize_hidden_state()

        encoder_output, encoder_state = self.encoder(data, state)

        output= self.decoder(data, encoder_output, encoder_state)

        return output
```

## Training

The training process is given by instantiating the Encoder-decoder class.

```py
model = Encoder_decoder(vocab_inp_size,
                        vocab_tar_size, 
                        embedding_dim, 
                        vocab_tar_size,
                        units, 
                        units, 
                        BATCH_SIZE)
```

#### Defining the optimizer and the loss function

```py
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)
```

After the model is instantiated, it is time to compile it.
```py
model.compile(optimizer = optimizer, loss = loss_function)
```

Applying the fit method to perform the training:

```py
history = model.fit(x = input_tensor_train[:-56],
          y = target_tensor_train[:-56],
          validation_data = (input_tensor_val[:-24], target_tensor_val[:-24]),
          batch_size = BATCH_SIZE,
          epochs=60,
          callbacks=[red_lr, ckpt, early_stop])
```

## Graph of the loss function in training
![bahdanau](/images/loss.png)

## Predict a sentence using the trained model

#### Function for plotting the weight chart according to the predicted words
```py
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()
```

#### Function to make the prediction

```py
def predict(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp)) #getting matrix vector to storage attentions vector predicted

  sentence = preprocess_sentence(sentence) # preprocess the input setense

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')] # transform all the inputs setense word in tokens 

  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post') # nomalization of the setense to have the same lenght as requirid in input layer network
  inputs = tf.convert_to_tensor(inputs) #converts the inputs in tensors

  result = ''

  hidden = [tf.zeros((1, units))] #Creating the hidden inicial state to pass to encoder layer
  enc_out, enc_hidden = model.layers[0](inputs, hidden) # predicting with encoder

  one_hidden = enc_hidden #encoder hidden state autput wil be passed to one step decoder
  one_input = tf.expand_dims([targ_lang.word_index['<start>']], 0) #first input will be the start token <start>

  for t in range(max_length_targ):
    predictions, one_hidden, attention_weights = model.layers[1].onestep(one_input,
                                                                         enc_out,
                                                                         one_hidden) #call the one step layer to get predicton on the input setense
    
    print(predictions.shape)
    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy() # getting the max value in predict correspondent to the most scored word

    result += targ_lang.index_word[predicted_id] + ' ' #concatenating eat word predicted

    if targ_lang.index_word[predicted_id] == '<end>': #checking if the sentense ends
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    one_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
```

#### Result of translation of a phrase in PTBR to EN.
![predict](/images/predict.png)

## Contributing

1. Fork it (<https://github.com/lps08/attention-model>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
