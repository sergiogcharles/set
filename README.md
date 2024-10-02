# Spacetime-E(n) Transformer

This is the PyTorch implementation of Spacetime $E(n)$-Transformer: Equivariant Attention for Spatio-temporal Graphs. The EGNN implementation and charged N-body dataset generation is adapted from: [https://github.com/vgsatorras/egnn.git](https://github.com/vgsatorras/egnn). The classical N-body dataset generation is adapted from: [https://github.com/pmocz/nbody-python](https://github.com/pmocz/nbody-python).

### N-body system experiment

#### Create Charged N-body dataset
```
cd n_body_system/dataset
python generate_dataset.py
```

#### Run experiments

*Linear*  
```
python charged_linear_nbody.py
```

*MLP*  
```
python charged_mlp_nbody.py
```
  
*LSTM*  
```
python charged_lstm_nbody.py
```

*EGNN*  
```
python charged_egnnvel_nbody.py
```

*SET*  
```
python charged_set_nbody.py
```   


#### Create Classical N-body dataset
```
cd n_body_gravity/dataset
python generate_dataset.py
```

#### Run experiments

*EGNN*  
```
python classical_egnnvel_nbody.py
```

*EGNN SchNet*  
```
python classical_egnnvelschnet_nbody.py
```
  
*SE(3) Transformer*  
```
python classical_se3transformer_nbody.py
```

*SET*  
```
python classical_set_nbody.py
```   

