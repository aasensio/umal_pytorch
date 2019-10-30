# Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians

This is a reimplementation in PyTorch of UMAL, a deep learning model to estimate 
the distribution of the target variable in any regression task. The original
[repository](https://github.com/BBVA/UMAL) uses Tensorflow and Keras.


> ***Axel Brando, Jose A. Rodriguez-Serrano, Jordi Vitria, Alberto Rubio. "Modelling heterogeneous distributions with an Uncountable Mixture of Asymmetric Laplacians." Advances in Neural Information Processing Systems. 2019.***

## Training

Although the implementation is general, we provide only training for the 
simple example of the heterogeneous output distributions. It can be retrained
with:

    python umal.py --lr=1e-2 --batch=1000 --epochs=2000 --ntaus=100 --patience=200

## Prediction

The prediction can be computed with:

    python test.py