# Neural machine translation with attention model
> Modelo treinado para fazer a tradução de frases em português para o inglês.

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![Tensorflow 2](https://img.shields.io/badge/Tensorflow-2.3-orange)](https://www.tensorflow.org/install?hl=pt-br)

O modelo sequence-to-sequence é um modelo comulmente usado para fazer tarefas de tradução, onde pode ser chamado de 'tradução de máquina'. Desse modo, esse modelo possui uma a abordagem na qual se utiliza modelos de aprendizado de máquina para obter uma sequência como entrada, de uma forma específica, e converter esta sequência para uma representação em outra forma. A tradução de textos é um bom exemplo desse poderoso recurso que o modelo sequence-to-sequence pode oferecer.

A partir do modelo de atenção, foi adicionado o mecanismo de atenção, que são técnicas de processamento de entrada para redes neurais que permitem que a rede se concentre em aspectos específicos de uma entrada complexa, um de cada vez, até que todo o conjunto de dados seja categorizado. O objetivo é dividir tarefas complicadas em áreas menores de atenção que são processadas sequencialmente. Semelhante a como a mente humana resolve um novo problema, dividindo-o em tarefas mais simples e resolvendo-as uma a uma. Na figura abaixo pode ser vista o modelo de atenção usado nesse projeto:

![bahdanau](https://blog.floydhub.com/content/images/2019/09/Slide38.JPG)

## Preprocessamento de texto

OS X & Linux:

```sh
npm install my-crazy-module --save
```

Windows:

```sh
edit autoexec.bat
```

## Exemplo de uso

Alguns exemplos interessantes e úteis sobre como seu projeto pode ser utilizado. Adicione blocos de códigos e, se necessário, screenshots.


## Contributing

1. Faça o _fork_ do projeto (<https://github.com/lps08/attention-model>)
2. Crie uma _branch_ para sua modificação (`git checkout -b feature/fooBar`)
3. Faça o _commit_ (`git commit -am 'Add some fooBar'`)
4. _Push_ (`git push origin feature/fooBar`)
5. Crie um novo _Pull Request_
