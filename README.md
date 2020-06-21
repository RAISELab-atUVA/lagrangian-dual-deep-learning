### Lagrangian Duality for Constrained Deep Learning

This repository provides implementation of our Lagrangian Duality method for applications in Fairness and Transprecision Computing which was introduced in our paper,  <a href="https://arxiv.org/pdf/2001.09394.pdf" target="_blank">ECML' 20</a>.

### 1. Intro

The goal of our project is to investigate the applicability of Lagrangian duality for deep learning model under prediction constraints. The constraints might arise from social requirements such as the model should not favor certain groups of people than the other, see [Fairness subsection](# Fairness) . Or the constraints might come from some prior knowledge in which for a given set of inputs we know the ordering of their predictions.

#### Fairness



#### Requirements
```
python==3.7
torch==1.3.1
```

#### References

1. <a href="https://arxiv.org/pdf/2001.09394.pdf" target="_blank">Lagrangian Duality for Constrained Deep Learning</a> <br>
Ferdinando Fioretto, Pascal Van Hentenryck, Terrence W.K. Mak, Cuong Tran, Federico Baldo, Michele Lombardi.. <br>
In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD), 2020.

 
