import sys
try:
    import pandas
    import sklearn
    import xgboost
    import lightgbm
    import shap
    print('IMPORTS_OK')
except Exception as e:
    print('IMPORT_ERROR:', e)
    sys.exit(1)
