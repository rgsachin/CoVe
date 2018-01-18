# Context Vectors (CoVe)

This repo provides the Keras implementation of [MT-LSTM from the paper Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107) . For a high-level overview of why CoVe are great, check out the [post](https://einstein.ai/research/learned-in-translation-contextualized-word-vectors).

The Weights are ported from PyTorch implementation of MT-LSTM by the paper's authur - https://github.com/salesforce/cove

## Dependencies
Ported & tested on:

- keras==2.1.3
- tensorflow-gpu==1.4.1

For re-running PortFromPytorchToKeras.ipynb requires of PyTorch MT-LSTM implementation from site: https://github.com/salesforce/cove

## Usage

### Loading model
```
from keras.models import load_model
cove_model = load_model('Keras_CoVe.h5')
```

### Prediction
- input - GloVe vectors of dimension - (<batch_size>, <sentence_len>, 300)
- output - CoVe vectors of dimension - (<batch_size>, <sentence_len>, 600)
#### Example
```
cove_model.predict(np.random.rand(1,10,300))
```

### Padding
At the time of porting, keras has issue with using Masking along with Bidirectional layer - https://github.com/keras-team/keras/issues/3086 ,a short-cut fix is applied, where the output of the final Bi-LSTM is removed off of prediction for padded field, refer PortFromPytorchToKeras.ipynb for the shortcut fix
### Unknow words
For unknown words we recommend to use value other than ones used for padding, a small non-zero value say 1e-10 is recommended 

## Implementation Details
Refer to PortFromPytorchToKeras.ipynb 

## Reference 
1. [MT-LSTM from the paper Learned in Translation: Contextualized Word Vectors (McCann et. al. 2017)](https://arxiv.org/abs/1708.00107)
2. [MT-LSTM PyTorch implementation from which weights are ported](https://github.com/salesforce/cove)
3. [GloVe](https://nlp.stanford.edu/projects/glove/)
