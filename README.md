# Neural machine translation with attention model PTBR-TO-ENGLISH
> Modelo treinado para fazer a tradução de frases em português para o inglês.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Tensorflow 2](https://img.shields.io/badge/Tensorflow-2.3-orange)](https://www.tensorflow.org/install?hl=pt-br)

O modelo sequence-to-sequence é um modelo comulmente usado para fazer tarefas de tradução, onde pode ser chamado de 'tradução de máquina'. Desse modo, esse modelo possui uma a abordagem na qual se utiliza modelos de aprendizado de máquina para obter uma sequência como entrada, de uma forma específica, e converter esta sequência para uma representação em outra forma. A tradução de textos é um bom exemplo desse poderoso recurso que o modelo sequence-to-sequence pode oferecer.

A partir do modelo de atenção, foi adicionado o mecanismo de atenção, que são técnicas de processamento de entrada para redes neurais que permitem que a rede se concentre em aspectos específicos de uma entrada complexa, um de cada vez, até que todo o conjunto de dados seja categorizado. O objetivo é dividir tarefas complicadas em áreas menores de atenção que são processadas sequencialmente. Semelhante a como a mente humana resolve um novo problema, dividindo-o em tarefas mais simples e resolvendo-as uma a uma. Na figura abaixo pode ser vista o modelo de atenção usado nesse projeto:

![bahdanau](https://blog.floydhub.com/content/images/2019/09/Slide38.JPG)

Projeto inspirado na implementação da [seção de tutorial do tensorflow](https://www.tensorflow.org/tutorials/text/nmt_with_attention) .

## Preprocessamento de texto

Primeiro deverá ser feita a conversão para ASCII

```py
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')
```
Após definida a função para transformar a sentensa em ASCII, uma nova função com o objetivo de remover caracteres especiais e adicionar as tags de inicio e fim, é implementada. As tags tanto para o inicio quanto para o fim tem o objetivo de mostrar para as redes neural onde as frases começam e terminam. A função dada pode ser mostrada logo abaixo:

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
Note que para cada sentença, entes do retorno da sentensa, as tags são adicionadas.

O proximo passo após fazer esses preprocessamentos, será adicionar tokens para cada palavra no dataset. O processo de tokenização é uma maneira de separar um pedaço de texto em unidades menores chamadas tokens. Aqui, os tokens podem ser palavras, caracteres ou subpalavras. Portanto, a tokenização pode ser amplamente classificada em 3 tipos - tokenização de palavra, caractere e subpalavra.

O processo de tokenização pode ser definida de acordo com a função abaixo, usando a classe Tokenizer da biblioteca do keras:

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
## Modelos usados
Os modelos usados que compõe esse projeto são: Encoder, One_Step_Decoder, Decoder, Encoder_decoder.

#### Encoder
A tarefa do modelo do encoder é lê a sequência de entrada e resume as informações em algo chamado de vetores de estado interno ou vetor de contexto (no caso do LSTM, esses são chamados de vetores de estado oculto e de estado de célula). Este vetor de contexto visa encapsular as informações para todos os elementos de entrada, a fim de ajudar o decodificador a fazer previsões precisas.

O modelo poder ser visto de acordo com a implementação abaixo:
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

#### Modelo Attention

O modelo de atenção tem o objetivo de permitir que as redes neurais se aproximem do mecanismo de atenção visual que os humanos usam. Como as pessoas processando uma nova cena, o modelo estuda um determinado ponto de uma imagem/texto com foco intenso, enquanto percebe as áreas ao redor em, então ajusta o ponto focal conforme a rede começa a entender a cena.

O modelo de atenção pode ser vista abaixo:

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

#### Modelo One_Step_Decoder

O modelo One_Step_Decoder pode ser pensado em dois processos separados, treinamento e inferência. Não é que eles tenham uma arquitetura diferente, mas eles compartilham a mesma arquitetura e seus parâmetros. É que eles têm estratégias diferentes para alimentar o modelo compartilhado. O decodificador mantém um estado oculto que passa de uma etapa de tempo para a próxima.

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

#### Modelo Decoder
O modelo tem como função de ter para cada batch de entrada, fazer o processamento de acordo com a Classe de One_Step_Decoder. Após fazer o processamento para cada item do batch, os resultados são armazenados em um tensor e retornado.

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

#### Modelo Encoder-decoder

Esse modelo tem o objetivo de chamar tanto o modelo de Encoder, quanto o de Decoder para fazer o processo de treinamento.

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

## Treinamento

O processo de treinamento é dado intanciando a classe Encoder-decoder.
```py
model = Encoder_decoder(vocab_inp_size,
                        vocab_tar_size, 
                        embedding_dim, 
                        vocab_tar_size,
                        units, 
                        units, 
                        BATCH_SIZE)
```

#### Definindo o otimizador e a função de perda

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

Após instanciado o modelo, é hora de compilá-lo.
```py
model.compile(optimizer = optimizer, loss = loss_function)
```

Aplicando o método fit para realizar o treinamento:
```py
history = model.fit(x = input_tensor_train[:-56],
          y = target_tensor_train[:-56],
          validation_data = (input_tensor_val[:-24], target_tensor_val[:-24]),
          batch_size = BATCH_SIZE,
          epochs=60,
          callbacks=[red_lr, ckpt, early_stop])
```

## Grafico da função de perda no treinamento
![bahdanau](/images/loss.png)

## Previsão usando o modelo treinado

#### Função para a plotagem do gráfico de pesos de acordo com as palavras previstas
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

#### Função para fazer a previsão

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

#### Resultado da tradução de uma frase em PTBR para EN.
![predict](/images/predict.png)

## Contributing

1. Faça o _fork_ do projeto (<https://github.com/lps08/attention-model>)
2. Crie uma _branch_ para sua modificação (`git checkout -b feature/fooBar`)
3. Faça o _commit_ (`git commit -am 'Add some fooBar'`)
4. _Push_ (`git push origin feature/fooBar`)
5. Crie um novo _Pull Request_
