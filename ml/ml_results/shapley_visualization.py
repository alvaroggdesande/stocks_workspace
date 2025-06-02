from typing import Optional

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml.ml_results.base_visualization import VisualizationObject
from ml.model_result import Result
from Utils.logger_config import logger


class ShapleyVisualization(VisualizationObject):
    """Class for SHAP visualizations."""

    def __init__(self,
                 result: Result,
                 data_type="X_pred",
                 data_type_y="y_pred",
                 use_inv=True,
                 allow_downsampling=True,
                 explainer: Optional[shap.TreeExplainer] = None,
                 **kwargs
                 ) -> None:
        if result.pipeline_data is None:
            raise ValueError(
                "`PipelineData` from `Result` is None. Cannot create SHAP visualizations.")
        # setup data
        self.data_type = data_type
        self.allow_downsampling = allow_downsampling
        self._set_data(data_type, data_type_y, result)

        # setup SHAP
        if explainer is None:
            logger.debug("No explainer provided. Creating a new explainer.")
            self.explainer = shap.TreeExplainer(
                model=result.model_handler.model,
                feature_names=self.feature_names,
                data=self.X,
                link="logit",
                #model_output='probability',
                feature_perturbation="interventional")
        self.shap_values_explainer = self.explainer(self.X)
        if use_inv:
            self.shap_values_explainer.data = self.X_inv
        shap.initjs()

    def bar_plot(self, max_display=10, show=False, clf=True, **kwargs):
        """Generate and display a bar plot."""
        # Global Plot
        if clf:
            plt.clf()
        return shap.plots.bar(
            self.shap_values_explainer,
            max_display=max_display,
            show=show,
            **kwargs)

    def waterfall_plot(self, idx, max_display=10, show=False, clf=True, **kwargs):
        """Generate and display a waterfall plot."""
        # Local Plot
        if clf:
            plt.clf()
        return shap.plots.waterfall(
            self.shap_values_explainer[idx],
            max_display=max_display,
            show=show,
            **kwargs)

    def force_plot(self, idx, max_display=10, show=False, matplotlib=True, clf=True, **kwargs):
        """Generate and display a force plot."""
        # Local Plot
        if clf:
            plt.clf()
        top_n_idx = self._get_top_n_features(n=max_display)
        f_names = [self.feature_names[i] for i in top_n_idx]
        return shap.plots.force(
            base_value=self.explainer.expected_value,
            shap_values=self.shap_values_explainer.values[idx, top_n_idx],  # type: ignore
            features=self.X_inv[idx, top_n_idx],
            feature_names=f_names,
            matplotlib=matplotlib,
            show=show,
            **kwargs)

    def violine_plot(self, max_display=10, show=True, clf=True, **kwargs):
        """Generate and display a violine plot."""
        # Local Plot
        if clf:
            plt.clf()
        return shap.plots.violin(
            self.shap_values_explainer,
            feature_names=self.feature_names,
            max_display=max_display,
            show=show,
            **kwargs)

    def beeswarm(self, max_display=10, show=False, clf=True, **kwargs):
        """Generate and display a beeswarm plot."""
        # Global Plot
        if clf:
            plt.clf()
        return shap.plots.beeswarm(
            self.shap_values_explainer,
            max_display=max_display,
            show=show,
            **kwargs)

    def beeswarm_plot(self, max_display=10, show=False, clf=True, **kwargs):
        return self.beeswarm(max_display=max_display, show=show, clf=clf, **kwargs)

    def dependency_plot(self, ind="rank(1)", show=False, clf=True, **kwargs):
        """Generates a SHAP dependence plot for a given feature or the top features,
        illustrating the relationship between the features and the model's predictions.
        This plot can help in understanding how the value of a single feature affects the
        model's output.

        Parameters:
        - ind (str, optional): Specifies the feature to plot. The default value "rank(1)"
            plots the most important feature. You can specify a feature name or its index
            in the feature list.
        - show (bool, optional): If True, the plot is displayed immediately. Defaults to
            False, which means the plot is created but not shown; it must be explicitly
            displayed by the caller.
        - clf (bool, optional): If True, the current figure is cleared before plotting.
            This is useful when generating multiple plots in the same figure. Defaults to
            True.

        Keyword Arguments:
        - interaction_index (str or int, optional): Specifies a feature for coloring
            points to show possible interactions. By default, it is "auto", which lets
            SHAP choose an interaction feature automatically. Can be set to a feature name,
            index, or None to disable coloring.
        - x_jitter (float, optional): Adds jitter to the feature values on the x-axis to
            make it easier to see dense points. Defaults to 0 (no jitter).
        - alpha (float, optional): Controls the transparency of the points in the plot.
            Useful for visualizing the density of points. Defaults to 1 (opaque).
        - display_features (DataFrame or array-like, optional): Specifies the display
            values for features. By default, uses the inverse transformed feature values
            `self.X_inv` if available.
        """
        # Global Plot
        if clf:
            plt.clf()
        return shap.dependence_plot(
            ind=ind,
            shap_values=self.shap_values_explainer.values,
            features=self.X_inv,
            feature_names=self.feature_names,
            display_features=self.X_inv,
            show=show,
            **kwargs)

    def score_comparison(self, n=2, clf=True):
        """Generate and display force plots for score comparison.
        The method randomly selects n data points from the top and bottom 10% of the
        predictions to create a more representative sample.

        Args:
            n (int, optional): The number of data points to compare. Default is 2.
        """
        if self.data_type != "X_pred":
            raise ValueError("score_comparison() only works with X_pred")
        if clf:
            plt.clf()
        head_idx, tail_idx = self.get_head_tail_indices(n=2)
        for idx in head_idx+tail_idx:
            self.force_plot(idx, show=False)
        plt.show()

    def default_output(self):
        """Plot the default output for SHAP visualizations."""
        plt.clf()
        self.beeswarm(show=True, clf=False)
        self.score_comparison(clf=False)

    def get_head_tail_indices(self, n=2):
        """Returns the indices of data points in the top and bottom 10% of predictions.

        Args:
            n (int, optional): The number of data points to sample from each group.
            Default is 2.

        Returns:
            tuple of lists: A tuple containing lists of indices for the top and bottom
            data points.
        """
        sorted_indices = np.argsort(self.y)  # type: ignore
        # get top/bottom 10% indices
        pred_len = len(sorted_indices)  # type: ignore
        head_10_percent = sorted_indices[-int(pred_len*0.1):]
        tail_10_percent = sorted_indices[:int(pred_len*0.1)]
        # get random indices from top/bottom 10% to create a more representative sample
        head_indices = np.random.choice(head_10_percent, size=n, replace=False)
        tail_indices = np.random.choice(tail_10_percent, size=n, replace=False)
        return list(head_indices), list(tail_indices)

    def _get_top_n_features(self, n=10):
        """Returns the indices of the top n features based on their SHAP values.
        Easily filter dataset to only include top n features. Example: X[:, top_idx]
        (Needed for force_plot())

        Parameters:
            n : int, optional
                The number of top features to return. Default is 10.

        Returns:
            top_idx : list of int
                The indices of the top n features based on their SHAP values.
        """
        vals = np.abs(self.shap_values_explainer.values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.feature_names, vals)),
                                          columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(
            by=['feature_importance_vals'], ascending=False, inplace=True)
        top_features = feature_importance["col_name"][:n].to_list()
        top_idx = [self.feature_names.index(i) for i in top_features]
        return top_idx

    def _set_data(self, type_X, type_y, result):
        """Set the input data based on the specified data type."""
        if type_X not in ["X_pred", "X_train", "X_test"]:
            raise ValueError(
                "type_ must be one of 'X_pred', 'X_train', 'X_test'")
        if type_y not in ["y_pred", "y_train", "y_test"]:
            raise ValueError(
                "type_ must be one of 'y_pred', 'y_train', 'y_test'")
        X = result.pipeline_data.__getattribute__(type_X)
        y = result.pipeline_data.__getattribute__(type_y)
        if (X.shape[0] > 10_000) and self.allow_downsampling:
            logger.warning(
                f"Data size is large {X.shape}. Sampling to 10000 data points.")
            sample_idx = np.random.choice(X.shape[0], 10000, replace=False)
            X = X[sample_idx, :]
            y = y.iloc[sample_idx]

        self.X = X
        self.y = y
        self.X_inv = result.preprocessor.inverse_transform(self.X)
        self.feature_names = result.preprocessor.feature_names or []