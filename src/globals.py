
X_FEATURES = [
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_GrayLevelVariance',
    'original_glrlm_LongRunHighGrayLevelEmphasis',
    'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_RunEntropy',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_ShortRunEmphasis'
]

IMPORTANT_FEATURES = [
    'original_glrlm_GrayLevelNonUniformity',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_GrayLevelVariance',
    'original_glrlm_LongRunHighGrayLevelEmphasis',
    'original_glrlm_LongRunLowGrayLevelEmphasis',
    'original_glrlm_RunEntropy',
    'original_glrlm_RunLengthNonUniformity',
    'original_glrlm_ShortRunEmphasis',
    'diagnosis_code'
]

MODEL_FILES = [
    './data/models/svm_norma_dsh.pkl',
    './data/models/svm_norma_gpb.pkl',
    './data/models/svm_norma_gpc.pkl',
    './data/models/svm_norma_vls.pkl',
    './data/models/svm_norma_auh.pkl'
]

SVM_MODEL_FILES = [
    './data/models/svm_norma_dsh.sav',
    './data/models/svm_norma_gpb.sav',
    './data/models/svm_norma_gpc.sav',
    './data/models/svm_norma_vls.sav',
    './data/models/svm_norma_auh.sav'
]

XGB_MODEL_FILES = [
    './data/models/xgb_glrlm.pickle.dat'
]

DISEASES = [
    'Norma',
    'Autoimmune Hepatitis',
    'Dyscholia',
    'Hepatitis B',
    'Hepatitis C',
    'Wilson\'s Disease'
]