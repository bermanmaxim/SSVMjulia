# SSVM-julia

Cutting-plane Structured SVM in Julia

This implementation has the following advantages over SVM-struct-matlab:
* Constraint caching (reuse full inference results)
* Parallelization of the inference step (one process per example)
* Additional positivity constraints on arbitrary coordinates of w

The two first points are very beneficial in settings were the inference task takes most of the time.

See (SSVM-julia-tutorial.ipynb) for a tutorial in IJulia notebook.

## Reference
Joachims, T., Finley, T., & Yu, C. N. J. (2009). Cutting-plane training of structural SVMs. Machine Learning, 77(1), 27-59.
