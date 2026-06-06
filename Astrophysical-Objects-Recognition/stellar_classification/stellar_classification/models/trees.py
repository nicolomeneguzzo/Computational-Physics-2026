"""Tree-based models for stellar classification."""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

class SimpleRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=200, max_depth=15,
                 min_samples_split=2, min_samples_leaf=1, 
                 max_features='sqrt', max_samples=None,  
                 class_weight='balanced', random_state=42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_split=min_samples_split,  
                         min_samples_leaf=min_samples_leaf,    
                         class_weight=class_weight, random_state=random_state, **kwargs)

class SimpleExtraTrees(ExtraTreesClassifier):
    def __init__(self, n_estimators=200, max_depth=15,
                 min_samples_split=2, min_samples_leaf=1,
                 max_features='sqrt', class_weight='balanced', random_state=42, **kwargs):
        super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_split=min_samples_split,  
                         min_samples_leaf=min_samples_leaf,    
                         class_weight=class_weight, random_state=random_state, **kwargs)