import numpy as np
import scipy.stats as stats
# Code for significance testing of normality of a dataset

def test_normality(data, alpha=0.05):
    """
    Runs normality tests on the given data.
    
    Parameters:
    data (array-like): The input histogram values.
    alpha (float): Significance level for hypothesis testing.
    
    Returns:
    dict: Test results including p-values and decision on normality.
    """
    results = {}
    
    # Shapiro-Wilk Test
    shapiro_stat, shapiro_p = stats.shapiro(data)
    results["Shapiro-Wilk"] = {
        "statistic": shapiro_stat,
        "p-value": shapiro_p,
        "normal": shapiro_p > alpha
    }
    
    # Kolmogorov-Smirnov Test (using normal distribution with sample mean and std)
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))
    results["Kolmogorov-Smirnov"] = {
        "statistic": ks_stat,
        "p-value": ks_p,
        "normal": ks_p > alpha
    }
    
    return results

if __name__ == "__main__":
    # Example: Generate a sample normal dataset
    sample_data = np.random.normal(loc=0, scale=1, size=1000)
    
    # Run the normality tests
    normality_results = test_normality(sample_data)
    
    # Print results
    for test, result in normality_results.items():
        print(f"{test} Test:")
        print(f"  Statistic: {result['statistic']:.4f}")
        print(f"  p-value: {result['p-value']:.4f}")
        print(f"  Data follows normal distribution: {result['normal']}")
        print("-")