import matplotlib.pyplot as plt
import mlflow
import shap
import warnings
from sklearn.pipeline import Pipeline

def log_shap_summary(estimator, x_test, model_name: str) -> None:
    """Log SHAP summary plot for the model.
    
    Parameters
    ----------
    estimator : Pipeline
        The trained scikit-learn pipeline.
    x_test : pandas.DataFrame
        Test data.
    model_name : str
        Name of the model.
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # We use shap.Explainer on the predict_proba method for efficiency
    try:
        # Preprocess x_test
        preprocessor = estimator.named_steps["preprocessor"]
        clf = estimator.named_steps["clf"]
        x_transformed = preprocessor.transform(x_test)
        
        # Use TreeExplainer if possible
        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(x_transformed)
        else:
            explainer = shap.Explainer(clf.predict, x_transformed)
            shap_values = explainer(x_transformed)
        
        # Get feature names from preprocessor
        feature_names = preprocessor.get_feature_names_out()
        
        fig = plt.figure()
        
        if isinstance(shap_values, list):
            # For multi-class or some tree explainers
            shap.summary_plot(shap_values[1], x_transformed, feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, x_transformed, feature_names=feature_names, show=False)
            
        mlflow.log_figure(fig, "shap_summary.png")
        plt.close(fig)
    except Exception as e:
        print(f"Failed to generate SHAP plot for {model_name}: {e}")
