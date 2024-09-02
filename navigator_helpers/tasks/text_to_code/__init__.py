try:
    import warnings

    import google.colab

    IN_COLAB = True
    warnings.simplefilter(action="ignore", category=FutureWarning)

except ImportError:
    IN_COLAB = False
