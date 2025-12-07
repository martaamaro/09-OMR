import numpy as np
from imbami.utils.base_relevance_function import BaseRelevanceFunction



class DensityRatioRelevance(BaseRelevanceFunction):
    """
    Relevance function based on density ratios (empirical / relevance).
    """
    def __init__(self):
        """
        Initializes DensityRatioRelevance.
        Inherits from BaseRelevanceFunction. No arguments are required.
        """
        super().__init__()
    def fit(self,
            emp_data: np.ndarray, 
            rel_data: np.ndarray | None,
            emp_bandwidth_type: str = 'silverman',
            rel_bandwidth_type: str = 'uniform',
            rel_bandwidth_factor: float = 1.0):
        """
        Fits KDEs to both empirical and relevance datasets.

        Parameters:
            emp_data (np.ndarray): Empirical data array.
            emp_bandwidth_type (str): Bandwidth selection method for empirical KDE ('silverman' or 'ISJ').
            rel_data (np.ndarray or None): Relevance data array, required if rel_bandwidth_type is not 'uniform'.
            rel_bandwidth_type (str): Bandwidth type for relevance KDE ('silverman', 'ISJ', or 'uniform').
            rel_bandwidth_factor (float): Scaling factor for relevance KDE bandwidth.
        """
        self.fit_to_data(emp_data, emp_bandwidth_type, rel_data, rel_bandwidth_type, rel_bandwidth_factor)

    def eval(self, y: np.ndarray) -> np.ndarray:
        """
        Evaluate relevance scores.

        Parameters:
            y (np.ndarray): Input values.

        Returns:
            np.ndarray: Relevance scores.
        """
        lamb = self.get_ratio(y)
        relevance = 1/ lamb
        return relevance

    def get_ratio(self, y: np.ndarray) -> np.ndarray:
        emp_density, rel_density = self.get_densities(y)
        lamb = np.divide(emp_density, rel_density)
        return lamb