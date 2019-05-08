# Texture Analysis of Medical Ultrasound images

This is a my Bachelor Diploma work at Igor Sikorsky Kyiv Polytechnic Institute.

I have trained models on SVM for each disease images.

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

```python
python3 main.py images/auh/2.png
```

```
Dyscholia: True 92%, False 7%
Hepatitis B: True 88%, False 11%
Hepatitis C: True 90%, False 9%
Wilson's Disease: True 93%, False 6%
Autoimmune Hepatitis: True 95%, False 4%
```