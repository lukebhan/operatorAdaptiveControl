# operatorAdaptiveControl

This repo contains the code for the paper titled Operater Learning for Nonlinear Adaptive Control. 
We provide three examples in this code. 

### Linear 
For the Linear example, run ```generate.py``` in the **estimator** folder to generate the dataset.
Then, copy the **.dat** files to the desired model directory (ie FNO for the FNO model). Then
there is a jupyter-notebook in the directory which can be used to train each model type. If one wants to 
compare the models, copy all the models to the compare directory and run ```compare.py```. 

### Aircraft
For the Linear example, run ```generate.py``` in the **sol** folder to generate the dataset.
Then, copy the **.dat** files to the desired model directory (ie FNO for the FNO model). Then
there is a jupyter-notebook in the directory which can be used to train each model type. If one wants to 
compare the models, copy all the models to the compare directory and run ```compare.py```. To run the models
with the controller, use the ```control.py``` file. 

### PDE
###### Dataset
To generate the dataset, one must run the matlab file named ```generate2.m``` in the **dataGeneration** folder.

###### Estimator
Copy the ```.mat``` files to the **estimator** folder to create each model in a jupyter notebook. 

###### Gain
Copy the ```.mat``` files to the **gain** folder to create each model in a jupyter notebook. 

###### Comparison and Running in Closed Feedback Loop
Copy the trained models from both the **estimator** and **gain** folders for each type of model into the 
both folder. Then one first needs to run the ```generateControlData.py``` to prepare the estimation and 
gain solutions for Matlab. Then go to the Matlab folder and run ```KortewegSimuSuccApp.m``` to create solutions
using the trained estimator and gain models. You will need to run this file for every model you have
changing the filenames to read from and save to for each model. These can then be visualized by running the  
```control.py``` and ```compare.py``` files. (```Control.py``` produces the control outputs for one sample. 
```compare.py``` considers an entire dataset of samples and gives errors). 
