# PyLib Finder Report

**Query:** function to locate knee in a PCA plot  
**Intent:** Find the optimal number of principal components by detecting the elbow/knee point in a PCA explained variance (scree) plot.  
**Date:** 2026-04-12_153516  

---

## Results (6 candidates, 2 excluded)

### 1. `kneed` — ⭐ RECOMMENDED
**Fit score:** 9.0/10  
**Safety:** ✅  
**Sources:** hints, stackoverflow, reddit, llm  

Automatically detects the knee/elbow point in a curve, directly applicable to finding the optimal number of components from a PCA scree plot. | functions: KneeLocator

> Directly and automatically detects the knee/elbow point in a curve, making it the most targeted solution for finding the optimal number of PCA components from a scree plot; minor deduction for 0.x pre-release status.

**Key functions/classes:** `KneeLocator`  
**Health:** v0.8.6 | updated recently  
**PyPI:** https://pypi.org/project/kneed/  
**GitHub:** https://github.com/arvkevi/kneed  

**Notes:**
- ℹ No license declared
- ℹ Pre-release version (0.8.6) — API may be unstable

---

### 2. `yellowbrick` — #2
**Fit score:** 6.0/10  
**Safety:** ✅  
**Sources:** hints, llm  

Machine learning visualization library that includes a KElbowVisualizer and PCA visualizer for determining optimal components or clusters. | functions: yellowbrick.features.PCA, KElbowVisualizer

> KElbowVisualizer automates elbow detection for clustering and its PCA visualizer helps with component analysis, but elbow detection is primarily designed for k-means rather than PCA scree plots; penalized for staleness (>730 days since update).

**Key functions/classes:** `KElbowVisualizer`, `yellowbrick.features.PCA`  
**Health:** v1.5 | updated 3y ago | 593,876 dl/month | Apache 2  
**PyPI:** https://pypi.org/project/yellowbrick/  
**GitHub:** https://github.com/DistrictDataLabs/yellowbrick/tarball/v1.5  
**Docs:** https://pythonhosted.org/yellowbrick/  

**Notes:**
- ⚠ Not updated in 3+ year(s)

---

### 3. `factor-analyzer` — #3
**Fit score:** 5.5/10  
**Safety:** ✅  
**Sources:** llm  

Provides parallel analysis and scree plot utilities for determining the number of factors/components to retain. | functions: FactorAnalyzer, calculate_bartlett_sphericity, calculate_kmo

> Provides parallel analysis and scree plot utilities for factor retention decisions which is related but not directly elbow detection on PCA explained variance; penalized for 0.x pre-release and >730 days since last update.

**Key functions/classes:** `FactorAnalyzer`, `calculate_kmo`, `calculate_bartlett_sphericity`  
**Health:** v0.5.1 | updated 2y ago | 48,665 dl/month  
**PyPI:** https://pypi.org/project/factor-analyzer/  
**GitHub:** https://github.com/EducationalTestingService/factor_analyzer  
**Docs:** https://github.com/EducationalTestingService/factor_analyzer  

**Notes:**
- ℹ No license declared
- ⚠ Not updated in 2+ year(s)
- ℹ Pre-release version (0.5.1) — API may be unstable

---

### 4. `prince` — #4
**Fit score:** 3.5/10  
**Safety:** ✅  
**Sources:** llm  

A library for multivariate exploratory data analysis (PCA, MCA, etc.) that provides eigenvalue-based diagnostics useful for component selection. | functions: PCA

> Offers PCA with eigenvalue diagnostics that could inform component selection, but does not provide automatic elbow/knee detection; penalized slightly for 0.x pre-release status.

**Key functions/classes:** `PCA`  
**Health:** v0.17.0 | updated recently  
**PyPI:** https://pypi.org/project/prince/  

**Notes:**
- ℹ No license declared
- ℹ No project URLs listed (no homepage or source link)
- ℹ Pre-release version (0.17.0) — API may be unstable

---

### 5. `plotly` — #5
**Fit score:** 3.0/10  
**Safety:** ✅  
**Sources:** llm  

Interactive plotting library that can create interactive scree plots, making it easier to visually identify the elbow point. | functions: plotly.express.line, plotly.graph_objects.Scatter

> Can create interactive scree plots for visual inspection of the elbow point, but provides no automatic elbow detection algorithm, serving only as a visualization aid.

**Key functions/classes:** `plotly.express.line`, `plotly.graph_objects.Scatter`  
**Health:** v6.7.0 | updated recently | 55.5M dl/month  
**PyPI:** https://pypi.org/project/plotly/  
**GitHub:** https://github.com/plotly/plotly.py/blob/main/CHANGELOG.md  

**Notes:**
- ℹ No license declared

---

### 6. `matplotlib` — #6
**Fit score:** 2.5/10  
**Safety:** ✅  
**Sources:** llm  

Essential for plotting the scree plot of explained variance to visually inspect the elbow point. | functions: matplotlib.pyplot.plot, matplotlib.pyplot.bar

> Essential for plotting the scree plot to visually inspect the elbow, but provides no automatic elbow/knee detection capability whatsoever.

**Key functions/classes:** `matplotlib.pyplot.plot`, `matplotlib.pyplot.bar`  
**Health:** v3.10.8 | updated 122d ago | 193.7M dl/month | License agreement for matplotlib versions 1.3.0 and later
  …  
**PyPI:** https://pypi.org/project/matplotlib/  
**GitHub:** https://github.com/matplotlib/matplotlib/issues  

---


## Comparison

**kneed** is purpose-built for exactly this task: you pass it the component indices and their explained variances, and it returns the elbow point automatically with minimal code. **yellowbrick** offers polished visualization and has broad adoption (~594k monthly downloads), but its elbow detector targets clustering (k-means) rather than PCA scree plots, and it hasn't been updated in over three years. **factor-analyzer** provides related scree/parallel-analysis tools from the factor-analysis tradition, yet it also lacks direct elbow detection and is similarly stale. The key trade-off is specificity versus ecosystem breadth: kneed does one thing well and is actively maintained, while the alternatives offer richer visualization or statistical context but don't squarely solve the stated problem. Use **kneed** — it's the most direct, lightweight, and up-to-date solution for programmatically finding the optimal number of principal components from a scree curve.


## Excluded Candidates (2)

| Package | Reason |
|---------|--------|
| `scikit-learn` | Known CVEs detected |
| `numpy` | Known CVEs detected |

**`scikit-learn`**
- CVE context: These three CVEs all relate to **persistent cross-site scripting (XSS) vulnerabilities in scikit-learn's HTML representation of estimators** (i.e., the `_repr_html_` output displayed in Jupyter notebooks), not in the core computational API. They do not affect typical programmatic use such as importing `sklearn.decomposition.PCA`, calling `.fit()`, or accessing `.explained_variance_ratio_`; they are only relevant if you display untrusted, attacker-crafted estimator objects in a Jupyter/HTML context.
- AST risk: This is a **false positive**. Scikit-learn is a widely trusted, established open-source machine learning library; its `os.environ` access in `__init__.py` is used for legitimate configuration purposes (e.g., controlling threading backends, build settings, and runtime behavior), not for malicious activity.

**`numpy`**
- CVE context: These CVEs are unlikely to affect typical programmatic use of numpy's numerical functions like `cumsum`, `diff`, and `gradient`. GHSA-2fc2-6r4j-p65h and GHSA-5545-2q6w-2gh6 involve tempfile race conditions and path traversal in `f2py` (a Fortran-to-Python compilation tool, not the core array API), while GHSA-6p56-wp2h-9hxr is a buffer overflow triggered by crafted `.f` files also in `f2py`—all scoped to optional build/compilation features that standard numerical computing workflows never invoke.
- AST risk: **False positive.** The real `numpy` package legitimately accesses `os.environ` for configuration purposes (e.g., controlling thread counts, build paths, and runtime settings like `NPY_PROMOTION_STATE`). This is entirely normal for a major scientific computing library and is not malicious.