### Lagrangian Duality for Constrained Deep Learning

This repository provides implementation of our Lagrangian Duality method for applications in Fairness and Transprecision Computing which was introduced in our paper,  <a href="https://arxiv.org/pdf/2001.09394.pdf" target="_blank">ECML' 20</a>.

### 1. Intro

The goal of our project is to investigate the applicability of Lagrangian duality for deep learning model under prediction constraints. The constraints might arise from social requirements such as the model should not favor certain groups of people than the other, see [Fairness subsection](#fair) . Or the constraints might come from some prior knowledge in which for a given set of inputs we know the ordering of their predictions, see [Transprecision Computing subsection](#trans).

#### <a name="fair"> 1.a Fairness </a>

One popular fairness definition is the Demographic Parity(DP) which requires the percentage of positive prediction outcomes across groups should be similar. See Eq. 10 in the paper. Please check the subfolder "fairness" and the notebook example to see how our model capture such DP fairness in learning the classifier. 


#### <a name="trans"> 1.b Transprecision Computing </a>

The movitation here comes from the need of reducing energy consumption. One solution for that is to reduce the precision (a.k.a number of bits) of all variables involved in the computation. However, reducing  the precision can upgrade the error of the target algorithm. Hence, we need to build a model which can predict such error based on the precision of the variables. In learning that model, we are equipped with the knowledge that higher precision configurations should generate more accurate results(smaller error). We impose such monotonic constraints in guiding the model to a more general pproximation of the target function.  Please check also the subfolder "tranprecision_computing" and the notebook example to see how our model works.




#### Requirements
```
python==3.7
torch==1.3.1
```

#### References

1. <a href="https://arxiv.org/pdf/2001.09394.pdf" target="_blank">Lagrangian Duality for Constrained Deep Learning</a> <br>
Ferdinando Fioretto, Pascal Van Hentenryck, Terrence W.K. Mak, Cuong Tran, Federico Baldo, Michele Lombardi.. <br>
In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), 2020.

 
