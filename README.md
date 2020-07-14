# Lagrangian Duality for Constrained Deep Learning

This repository provides the implementation of the Lagrangian Dual learning Framework (LDF) described in [Lagrangian Duality for Constrained Deep Learning](https://arxiv.org/pdf/2001.09394.pdf). 
It focuses on applications in _Fairness_ and _Transprecision Computing_. 


## About

The project develops a Lagrangian duality framework for learning applications that feature complex constraints. Such constraints arise in many science and engineering domains, where the task amounts to learning optimization problems which must be solved repeatedly and include hard physical and operational constraints. The framework also considers applications where the learning task must enforce constraints on the predictor itself, either because they are natural properties of the function to learn or because it is desirable from a societal standpoint to impose them.

The [paper]((https://arxiv.org/pdf/2001.09394.pdf)) demonstrates experimentally that Lagrangian duality brings significant benefits for these applications. In energy domains, the combination of Lagrangian duality and deep learning can be used to obtain state of the art results to predict optimal power flows, in energy systems, and optimal compressor settings, in gas networks. In transprecision computing, Lagrangian duality can complement deep learning to impose monotonicity constraints on the predictor without sacrificing accuracy. Finally, Lagrangian duality can be used to enforce fairness constraints on a predictor and obtain state-of-the-art results when minimizing disparate treatments.

In the following we report two example applications deep learning with for [group fairness constraints](#fair) and for [transprecision computing](#trans)


### <a name="fair"></a>Fairness
One popular fairness definition is the Demographic Parity(DP) which requires the percentage of positive prediction outcomes across groups should be similar. See Eq. 10 in the paper. Please check the subfolder "fairness" and the Demo_For_Bank_Data.ipynb  example to see how our model capture such DP fairness in learning the classifier. 

#### Example
    python3 run.py

### <a name="trans"></a>Transprecision Computing

Transprecision computing is the idea of reducing energy consumption by reducing the precision (a.k.a. number of bits) of the variables involved in a computation. It is especially important in low-power embedded platforms, which arise in many contexts such as smart wearable and autonomous vechicles. Increasing precision typically reduces the error of the target algorithm. However, it also increases the energy consumption, which is a function of the maximal number of used bits. 

The objective is to design a configuration _`d_l`_, i.e., a mapping from input computation to the precision for the variables involved in the computation. The sought configuration should balance precision and energy consumption, given a bound to the error produced by the loss in precision when the highest precision configuration is adopted.

However, given a configuration, computing the corresponding error can be very time-consuming and the task considered in this paper seeks to learn a mapping between configurations and error. This learning task is non-trivial, since the solution space precision-error is non-smooth and non-linear. 

The samples _(`d_l`, `y_l`)_ in the dataset represent, respectively, a configuration dl and its associated error _`y_l`_ obtained by running the configuration _`d_l`_ for a given computation. 
The problem _`O(d_l)`_ specifies the error obtained when using configuration _`d_l`_. Importantly, transcomputing expects a monotonic behavior: Higher precision configurations should generate more accurate results (i.e., a smaller error). Therefore, the structure of the problem imposes the learning task to require a dominance relation `<=` between instances of the dataset. More precisely, _`d_ <= d_2`_ holds if<br>
`\forall i \in [N] x1_i <= x2_i`<br>
where _`N`_ is the number of variables involved in the computation and _`x1_i`_ , _`x2_i`_ are the precision values for the variables in _`d_1`_ and _`d_2`_ respectively.

#### Example

       


#### Requirements
```python
python==3.7
torch==1.3.1
```

#### Cite As
```bibtex
@article{Fioretto:ECML20,
    title     = "A Lagrangian Dual Framework for Deep Neural Networks with Constraints Optimization",
    author    = "Ferdinando Fioretto and  Pascal {Van  Hentenryck} and Terrence {W.K. Mak} and Cuong Tran and Federico Baldo and Michele Lombardi",
    booktitle = "European Conference on Machine Learning and  Principles and Practice of Knowledge Discovery in Databases ({ECML-PKDD})",
    year      = "2020"
}
```
 
#### Contact
Ferdinando Fioretto <ffiorett@syr.edu><br>
Cuong Tran <cutran@syr.edu>
