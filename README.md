# DL_based_ECG_project
The dissertation project of MSc in CityU

# Run models
## 1. Run two DL_based models
### For STA_CRNN model
After cloning this repo, you can run the model by:
```
cd DL_based_ECG_project
cd STA_CRNN
python cpsc2018_torch.py
```
### For VAN model
```
cd DL_based_ECG_project
cd VAN
python train.py
```

## 2. Run ACGAN model
```
cd DL_based_ECG_project
cd ECG_GAN
python trainer.py
```
### 2.1 Quantitative evaluation comparison between two ACGAN models
* Dynamic time warping (DTW)
* Fourier coefficients
* Visualization
```
cd DL_based_ECG_project
cd ECG_GAN
python KL_divergency.py
```

## 3. Run two self-supervised learning models
### 3.1 CPC
```
cd DL_based_ECG_project
cd CPC
./run.sh
```
### 3.2 MoCo
```
cd DL_based_ECG_project
cd MOCO
./run.sh
```

# Show results
## 1. Show heatmap of confusion matrices
```
cd DL_based_ECG_project
cd Metrics
python save_heatmap.py
```
## 2. Show the ROC curves
```
cd DL_based_ECG_project
cd Metrics
python Roc_GAN_before_after.py
```
