import numpy as np

class ObjectPriors:
    """
    Manages size priors for 3D object detection based on the provided research paper.
    This class stores mean and standard deviation for object dimensions (L, W, H)
    and provides a method to generate realistic proxy object sizes, with special
    handling for 'Car' and 'Pedestrian' subcategories.
    """
    
    # Overall class statistics (Length, Width, Height) from the table
    CLASS_MEAN_SIZE = {
        'Car': np.array([4.35, 1.84, 1.56]), # L, H, W
        'Pedestrian': np.array([0.70, 1.62, 0.70]),
        'Cyclist': np.array([1.80, 1.73, 0.64]),
        'Bus': np.array([15.75, 2.50, 3.60]),
        'Trailer': np.array([6.00, 2.00, 2.00]),
        'Traffic_Cone': np.array([0.70, 0.50, 0.50]),
        'Barrier': np.array([1.50, 0.40, 0.60]),
    }

    CLASS_STD_SIZE = {
        'Car': np.array([0.55, 0.08, 0.12]),
        'Pedestrian': np.array([0.20, 0.06, 0.20]),
        'Cyclist': np.array([0.20, 0.09, 0.20]),
        'Bus': np.array([0.20, 0.20, 0.20]),
        'Trailer': np.array([1.50, 0.20, 0.20]),
        'Traffic_Cone': np.array([0.20, 0.20, 0.20]),
        'Barrier': np.array([0.20, 0.20, 0.20]),
    }

    # Subcategory data used exclusively for generating realistic proxy objects
    SUBCATEGORIES = {
        'Car': {
            # Format is L, H, W
            'City car':       {'mean': np.array([3.07, 1.76, 1.51]), 'std': np.array([0.72, 0.63, 0.03])},
            'Small car':      {'mean': np.array([3.07, 1.76, 1.51]), 'std': np.array([0.11, 0.03, 0.04])},
            'Compact car':    {'mean': np.array([4.22, 1.80, 1.50]), 'std': np.array([0.08, 0.02, 0.04])},
            'Family car':     {'mean': np.array([4.19, 2.10, 1.45]), 'std': np.array([0.85, 0.87, 0.04])},
            'Executive car':  {'mean': np.array([4.79, 1.85, 1.44]), 'std': np.array([0.05, 0.02, 0.03])},
            'Luxury car':     {'mean': np.array([4.99, 1.92, 1.42]), 'std': np.array([0.07, 0.02, 0.03])},
            'Sports car':     {'mean': np.array([4.31, 1.85, 1.28]), 'std': np.array([0.22, 0.08, 0.08])},
            'MPV':            {'mean': np.array([4.63, 1.81, 1.48]), 'std': np.array([0.04, 0.02, 0.03])},
            'Small SUV':      {'mean': np.array([4.70, 1.87, 1.76]), 'std': np.array([0.35, 0.07, 0.17])},
            'Compact SUV':    {'mean': np.array([3.93, 1.71, 1.53]), 'std': np.array([0.16, 0.06, 0.04])},
            'Mid-size SUV':   {'mean': np.array([4.23, 1.80, 1.56]), 'std': np.array([0.03, 0.01, 0.04])},
            'Large SUV':      {'mean': np.array([4.54, 1.83, 1.63]), 'std': np.array([0.03, 0.02, 0.03])},
            'Pick-up':        {'mean': np.array([4.73, 1.91, 1.68]), 'std': np.array([0.02, 0.04, 0.09])},
            'Passenger Van':  {'mean': np.array([5.35, 1.86, 1.83]), 'std': np.array([0.12, 0.03, 0.04])},
            'Estate car':     {'mean': np.array([4.43, 1.84, 1.84]), 'std': np.array([0.05, 0.02, 0.03])},
        },
        'Pedestrian': {
            'Pedestrian adult': {'mean': np.array([0.70, 1.73, 0.70]), 'std': np.array([0.20, 0.09, 0.20])},
            'Pedestrian kid':   {'mean': np.array([0.70, 1.52, 0.70]), 'std': np.array([0.20, 0.05, 0.20])},
        }
    }

    @classmethod
    def generate_proxy_size(cls, class_name: str) -> np.ndarray:
        """
        Generates a realistic object size (L, H, W) based on size priors.
        """
        if class_name in cls.SUBCATEGORIES:
            subcat_name = np.random.choice(list(cls.SUBCATEGORIES[class_name].keys()))
            stats = cls.SUBCATEGORIES[class_name][subcat_name]
            mean, std = stats['mean'], stats['std']
        elif class_name in cls.CLASS_MEAN_SIZE:
            mean = cls.CLASS_MEAN_SIZE[class_name]
            std = cls.CLASS_STD_SIZE[class_name]
        else:
            raise ValueError(f"Unknown class name for proxy generation: {class_name}")

        return np.maximum(0.1, np.random.normal(mean, std))