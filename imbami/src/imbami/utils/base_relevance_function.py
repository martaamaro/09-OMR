import numpy as np
import KDEpy
from KDEpy.bw_selection import improved_sheather_jones
import logging
import bisect


class BaseRelevanceFunction:
    """
    Base class for relevance functions based on density estimation.
    Provides shared functionality for empirical and relevance KDE fitting and evaluation.
    """
    def __init__(self):
        self.usable_emp_bandwidth_type = ['silverman', 'ISJ']
        self.usable_rel_bandwidth_type = ['silverman', 'ISJ', 'uniform']
        self._bisect_kde_score = bisect_kde_score
        self.get_kernel = get_kernel

    def fit_to_data(
        self,
        emp_data: np.ndarray,
        emp_bandwidth_type: str,
        rel_data: np.ndarray | None,
        rel_bandwidth_type: str,
        rel_bandwidth_factor: float
        ) -> None:
        """
        Fits KDEs to both empirical and relevance datasets.

        Parameters:
            emp_data (np.ndarray): Empirical data array.
            emp_bandwidth_type (str): Bandwidth selection method for empirical KDE ('silverman' or 'ISJ').
            rel_data (np.ndarray or None): Relevance data array, required if rel_bandwidth_type is not 'uniform'.
            rel_bandwidth_type (str): Bandwidth type for relevance KDE ('silverman', 'ISJ', or 'uniform').
            rel_bandwidth_factor (float): Scaling factor for relevance KDE bandwidth.
        """
        if rel_bandwidth_type not in self.usable_rel_bandwidth_type:
            raise ValueError(f"{rel_bandwidth_type=} is not an acceptable value: {self.usable_rel_bandwidth_type}")
        if emp_bandwidth_type not in self.usable_emp_bandwidth_type:
            raise ValueError(f"{emp_bandwidth_type=} is not an acceptable value: {self.usable_emp_bandwidth_type}")
                
        # Empirical Data
        self.emp_data = emp_data
        self.emp_bandwidth_type = emp_bandwidth_type
        self.empirical_kernel = self.get_kernel(data = self.emp_data,
                                           bandwidth_type= self.emp_bandwidth_type,
                                           bandwidth_factor= 1)
        self.emp_grid_x, self.emp_grid_y = self.empirical_kernel.evaluate(2**14)

        # Relevance Data            
        self.rel_bandwidth_type = rel_bandwidth_type
        self.rel_bandwidth_factor = rel_bandwidth_factor
        if self.rel_bandwidth_type == 'uniform':
            self.rel_kernel = Uniform_kernel(data = self.emp_data)
            self.rel_grid_x = np.linspace(start = self.emp_data.min(), stop = self.emp_data.max(), num= 2**14)
            self.rel_grid_y = self.rel_kernel(self.rel_grid_x)
        else:
            if rel_data is None or not isinstance(rel_data, np.ndarray):
                raise ValueError("If the rel_bandwidth_type is not 'uniform', rel_data must be provided as a np.ndarray.")
            self.rel_data = rel_data
            self.rel_kernel = self.get_kernel(data = self.rel_data,
                                            bandwidth_type= self.rel_bandwidth_type,
                                            bandwidth_factor= self.rel_bandwidth_factor)
            self.rel_grid_x, self.rel_grid_y = self.rel_kernel.evaluate(2**14)

    def get_densities(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate the density values for the input y from both empirical and relevance KDEs.

        Parameters:
            y (np.ndarray): Data points for which to compute densities.

        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                - emp_density: KDE values based on the empirical distribution.
                - rel_density: KDE values based on the relevance distribution.
        """
        emp_density = map(lambda i : self._bisect_kde_score(i, self.emp_grid_x, self.emp_grid_y), y)        
        emp_density = np.fromiter(emp_density, dtype=np.float64)
        if isinstance(self.rel_kernel, Uniform_kernel):
            rel_density = self.rel_kernel(y)
        else:
            rel_density = map(lambda i : self._bisect_kde_score(i, self.rel_grid_x, self.rel_grid_y), y)        
            rel_density = np.fromiter(rel_density, dtype=np.float64)
        return emp_density, rel_density


    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Alias to eval for calling the object directly with new input."""
        return self.eval(y)

class Uniform_kernel:
    """
    Simple uniform kernel for relevance estimation. 
    Returns constant density within the range of the training data.
    """
    def __init__(self, data: np.ndarray):
        self.rel = 1 / (max(data) - min(data))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return np.full(shape = y.shape, fill_value = self.rel)


def get_kernel(data: np.ndarray,
               bandwidth_type: str,
               bandwidth_factor: float) -> KDEpy.FFTKDE:
    """
    Returns a Gaussian KDE kernel using specified bandwidth type.

    Parameters:
        data (np.ndarray): Data array to fit the KDE.
        bandwidth_type (str): 'ISJ' for Improved Sheather-Jones or 'silverman'.
        bandwidth_factor (float): Factor to scale the bandwidth.

    Returns:
        KDEpy.FFTKDE: The fitted kernel density estimator.
    """
    match bandwidth_type:
        case 'ISJ':
            try:
                bandwidth = improved_sheather_jones(data.reshape(-1,1))
            except Exception as error:
                logging.warning('Failed to compute ISJ Bandwidth. Fall back to Silverman bandwidth as default.\n' + f'Exception error: {error}')
                bandwidth = (4*data.std(ddof=1)**5 / 3 / len(data))**(1/5)
        case 'silverman':
            bandwidth = (4*data.std(ddof=1)**5 / 3 / len(data))**(1/5)
        case _:
            raise ValueError(f"Unsupported bandwidth_type: {bandwidth_type}")
        
    bandwidth = bandwidth * bandwidth_factor
    kernel = KDEpy.FFTKDE(bw = bandwidth, kernel = 'gaussian').fit(data) # type: ignore
    return kernel
    
def bisect_kde_score(y, grid_x, grid_y) -> float:
    """
    Estimate KDE value for a point y using precomputed grid.

    Parameters:
        y (float): Point for density evaluation.
        grid_x (np.ndarray): KDE x-axis grid.
        grid_y (np.ndarray): KDE y-axis (density) values.

    Returns:
        float: Estimated density value at y.
    """
    idx = bisect.bisect(grid_x, y)
    idx = min(max(idx, 0), len(grid_x) - 1)
    return grid_y[idx]