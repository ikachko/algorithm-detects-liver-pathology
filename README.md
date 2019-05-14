# Texture Analysis of Medical Ultrasound images

This is a my Bachelor Diploma work at Igor Sikorsky Kyiv Polytechnic Institute.

I have trained models on XGBoost for each disease images.

Available diseases images:

| Disease           | Image Count   |
| -----------       | ------------  |
| Dyscholia         | 5             |
| Hepatitis B       | 9             |
| Hepatitis C       | 11            |
| Wilson's Disease  | 6             |
| Autoimmune Hepatitis       | 8             |
| Norma        | 68             |

As you see, amount of disease images are pretty low, to build some models, but I tried anyway.
If you have some Ultrasound Images to share with, please contact me.


When you input some image, pre-trained models predicts output with GLRLM features and show percentage output for each 
### Feature Extracting Techniques
- [x] GLCM
- [x] GLRLM

GLCM Accuracy:

| Dataset   | Predictor  | Accuracy |
| --------- | ---------- | -------- |
| GLCM_0    | XDGBoost   | 50-60%      |

GLRLM Accuracy on Binary Classification:

| Dataset    | Predictor   | Accuracy |
| ---------  | ----------  | -------- |
| GLRLM_0    | XDGBoost    | 88-96%   |
| GLRLM_0    | LogisticReg | 85-95%   |
| GLRLM_0    | SVM         | 86-93%   |
| GLRLM_0    | KNN         | 86-93%   |


## Usage

```python
python3 main.py image.png
```

## Examples

These are examples of Gradient Boosting model probability prediction

#### Norma
```python
python3 main.py data/images/png/norm/norma0.png
```
```
Norma: 98%
Autoimmune Hepatitis: 0%
Dyscholia: 0%
Hepatitis B: 0%
Hepatitis C: 0%
Wilson's Disease: 0%
```

#### Dyscholia
```python
python3 main.py data/images/png/dsh/4.png
```
```
Norma: 2%
Autoimmune Hepatitis: 3%
Dyscholia: 85%
Hepatitis B: 2%
Hepatitis C: 2%
Wilson's Disease: 2%
```

#### Autoimmune Hepatitis

```python
python3 main.py data/images/png/auh/6.png
```
```
Norma: 1%
Autoimmune Hepatitis: 92%
Dyscholia: 1%
Hepatitis B: 0%
Hepatitis C: 1%
Wilson's Disease: 1%
```

#### Hepatitis B

```python
python3 main.py data/images/png/hpb/1.png
```

```
Norma: 15%
Autoimmune Hepatitis: 0%
Dyscholia: 0%
Hepatitis B: 83%
Hepatitis C: 0%
Wilson's Disease: 0%
```

#### Hepatitis C

```python
python3 main.py data/images/png/hpc/1.png
```
```
Norma: 0%
Autoimmune Hepatitis: 2%
Dyscholia: 0%
Hepatitis B: 0%
Hepatitis C: 94%
Wilson's Disease: 3%
```

#### Wilson's Disease

```python
python3 main.py data/images/png/wls/8.png
```
```
Norma: 5%
Autoimmune Hepatitis: 1%
Dyscholia: 0%
Hepatitis B: 0%
Hepatitis C: 1%
Wilson's Disease: 90%
```