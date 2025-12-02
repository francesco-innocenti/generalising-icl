# A Simple Generalisation of the Implicit Dynamics of In-Context Learning

Code for the NeurIPS 2025 workshop paper [**A Simple Generalisation of the 
Implicit Dynamics of In-Context Learning**](https://transformerstheory.github.io/pdf/34_innocenti_achour.pdf). 
This work is essentially an extension of the recent paper 
[Learning without training: The implicit dynamics of in-context learning](https://arxiv.org/abs/2507.16003).


## 💻 Reproduce results
Inside a virtual environment, first install all the requirements
```
pip install -r requirements.txt
```
To reproduce all the plots and results, then run
```
python test_theory.py
```


## 📝 Other notes
To experiment with different parameters (e.g. more transformer blocks), simply 
disable the `run_sweeps` flag
```
python test_theory.py --run_sweeps False --n_blocks 10
```
The theoretical implicit parameter updates are computed in `analytical.py`.
