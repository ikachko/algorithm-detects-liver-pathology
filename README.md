# Texture Analysis of Medical Ultrasound images

This is a my Bachelor Diploma work at Igor Sikorsky Kyiv Polytechnic Institute.

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
