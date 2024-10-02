# Spacetime-E(n) Transformer

### N-body system experiment

#### Create Charged N-body dataset
```
cd n_body_system/dataset
python -u generate_dataset.py
```

#### Run experiments

*Linear*  
```
python linear_dynamics_baseline_nbody.py
```

*MLP*  
```
python mlp_dynamics_baseline_nbody.py
```
  
*LSTM*  
```
python spatiotemporal_lstm_nbody.py
```

*EGNN*  
```
python egnn_vel_baseline_nbody.py
```

*SET*  
```
python spatiotemporal_transformer_nbody.py
```   


#### Create Classical N-body dataset
```
cd n_body_gravity/dataset
python -u generate_dataset.py
```

#### Run experiments

*EGNN*  
```
python spatiotemporal_egnnvel_nbody_gravity.py
```

*EGNN SchNet*  
```
python spatiotemporal_egnnvelschnet_nbody_gravity.py
```
  
*SE(3) Transformer*  
```
python spatiotemporal_se3transformer_nbody_gravity.py
```

*SET*  
```
python spatiotemporal_transformer_nbody_gravity.py
```   

