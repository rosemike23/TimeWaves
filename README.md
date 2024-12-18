# TimeWaves
The code of TimeWaves model for dynamical forecasting


## Download datasets
The data used in this project can be accessed via this [link](https://drive.google.com/drive/folders/1xZso6MUk00Ly_o0gs3VVCF2Odr7sR6KY?usp=drive_link).

Please download them and drag the data folders into ./dataset/.



## Preparation
TimeWaves model requires a custoum "_cwtline.py" to function.

'_cwtline.py' file can be found in the same folder as this README.txt.

To achieve this function, please proceed with the following steps:

1. Install Python 3.9. For convenience, execute the following command:
pip install -r requirements.txt

2. Modify the PyWavelets(pywt) pack:
- First, drag '_cwtline.py' into the "pywt" pack folder.
  
![alt text](image.png)

- Then, add a line of code (from ._cwtline import *) in line 25 of '__init__.py'.
  
![alt text](image-1.png)

3. Check function
- Go to the models folder and find 'TimesWave_boost.py'
- Locate line 41, and the code should recognize 'pywt.cwtline'.
- You are good to go！


## Run the program
Now we are ready to train and evaluate our model:
- We provide the experiment scripts for tested benchmarks under the folder ./scripts/.
- You can run the experiment results as the following examples（eg. long-term forecast - rocket）:
  
```
# Rocket forecast
bash ./scripts/long_term_forecast/Rocket_script/TimeWaves.sh
```

## Reference 

```
@inproceedings{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Learning Representations},
  year={2023},
}
```
