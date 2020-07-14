# Intent-Classification
Praca domowa, rekrutacja <br/>
Klasyfikacja intecji przy pomocy sieci neuronowej opartej na warstwach splotowych. Wykorzystałem dataset snips
[link tutaj](https://github.com/snipsco/nlu-benchmark). <br/>
Sieć zaimplementowana przy użyciu Pytorcha składa się z 3 warstw konwolucyjnych i 2 dwóch warstw fully connected.
Osiągnąłem skuteczność 97,86% na zbiorze testowym poniżej znajduje się wykres przedstawiający skuteczność osiągniętą w trakcie treningu. ![wykres skuteczności](https://github.com/knowosadko/Intent-Classification/blob/master/accuracy_graph.png) <br/> Natomiast poniżej znaduje się macierz błędów popełnionych na zbiorze testowym. ![macierz bledow](https://github.com/knowosadko/Intent-Classification/blob/master/confusion_matrix.png) <br/>
### Szczegóły dotyczące implementacji
Preprocessing polega na wyrzuceniu ze zdań znajdującyh się w zbiorze treningowym słów nienależących do języka angielskiego. 
Do reprezentacji słów użyłem modelu Fasttext, który wcześniej zostaje wyuczony na zbiorze treningowym przy uzyciu algorytmu uczenia bez nadzoru. Nastepnie każde ze zdań formuje macierz kwadratową, która stanowi wejście do sieci.
Architektura sieci wygląda następująco:


    ConvolutionalNetwork(
      (conv_layer1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (conv_layer2): Sequential(
        (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (conv_layer3): Sequential(
        (0): Conv2d(32, 64, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
      )
      (fc1): Linear(in_features=1024, out_features=200, bias=True)
      (fc2): Linear(in_features=200, out_features=7, bias=True)
    )

Poniżej osiągnięte wyniki:
Results:  97.86%

              precision    recall  f1-score   support

           0       1.00      0.97      0.98       100
           1       1.00      1.00      1.00       100
           2       1.00      0.98      0.99       100
           3       0.97      0.98      0.98       100
           4       1.00      0.99      0.99       100
           5       0.90      1.00      0.95       100
           6       0.99      0.93      0.96       100

    accuracy                           0.98       700
   macro avg       0.98      0.98      0.98       700
weighted avg       0.98      0.98      0.98       700
Wyniki osiągnięte dla learning rate = 0.001 i 8 epok.
