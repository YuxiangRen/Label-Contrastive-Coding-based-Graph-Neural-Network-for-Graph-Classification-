
This is the implementation of paper:

> Label Contrastive Coding based Graph Neural Network for Graph Classification

### Requirements
The code is implemented in Python 3.7. Package used for development are just below.
```
networkx       
numpy              
scipy              
torch == 1.4.0
torch_geometric == 1.6.0
```


###Instructions for running the code

For LCGNN with different encoders, the training scripts are in separate files (e.g., ./for_gin).


1, Enter the for_gin file
```
cd ./for_gin
```

2, Run the code
```
python3 train_powerfulgnn_oneenc.py
```
for the momentum weight $\alpha = 0$ condition; or run the code

```
python3 train_powerfulgnn_twoenc.py
```
for other conditions.



###Note:

1, The default setting includes using the GPU.
2, To change model configurations, (e.g., set the epoch numbers of training as NNUMBER), add config `--epochs NUMBER`.
# Label-Contrastive-Coding-based-Graph-Neural-Network-for-Graph-Classification-
