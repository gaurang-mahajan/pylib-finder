# PyLib Finder Report

**Query:** find python function to identify knee in PCA plot  
**Intent:** Find a Python function to automatically detect the elbow/knee point in a PCA scree plot (explained variance vs. number of components).  
**Date:** 2026-02-27_225701  

---

## Results (4 candidates, 5 excluded)

### 1. `kneed` — ⭐ RECOMMENDED
**Fit score:** 9.5/10  
**Safety:** ✅  
**Sources:** llm  

A Python library specifically designed to detect knee/elbow points in curves, perfect for finding the optimal number of components in a PCA scree plot. | functions: KneeLocator

> The kneed library is purpose-built for detecting knee/elbow points in curves, making it an excellent fit for identifying the optimal number of PCA components from a scree plot of explained variance.

**Key functions/classes:** `KneeLocator`  
**Health:** v0.8.5 | updated 2y ago | 888,564 dl/month  
**PyPI:** https://pypi.org/project/kneed/  
**GitHub:** https://github.com/arvkevi/kneed  

**Notes:**
- ⚠ Not updated in 2+ year(s)

---

### 2. `pca` — #2
**Fit score:** 9.0/10  
**Safety:** ✅  
**Sources:** llm  

A dedicated PCA library that wraps scikit-learn and includes built-in scree plot generation with automatic elbow detection functionality. | functions: pca.fit_transform, pca.plot

> The `pca` library directly addresses the user's need by wrapping scikit-learn's PCA with built-in scree plot generation and automatic elbow/knee point detection, making it an excellent fit for the task.

**Key functions/classes:** `pca.fit_transform`, `pca.plot`  
**Health:** v2.10.2 | updated recently  
**PyPI:** https://pypi.org/project/pca/  
**GitHub:** https://github.com/erdogant/pca/archive/{version}.tar.gz  

---

### 3. `yellowbrick` — #3
**Fit score:** 4.0/10  
**Safety:** ✅  
**Sources:** llm  

A machine learning visualization library that includes a KElbowVisualizer for detecting optimal clusters/components via the elbow method. | functions: KElbowVisualizer

> Yellowbrick's KElbowVisualizer is designed for detecting the optimal number of clusters in clustering algorithms (e.g., KMeans), not for detecting the elbow/knee point in a PCA scree plot of explained variance, so while conceptually related to elbow detection, it does not directly address the user's specific PCA use case.

**Key functions/classes:** `KElbowVisualizer`  
**Health:** v1.5 | updated 3y ago | Apache 2  
**PyPI:** https://pypi.org/project/yellowbrick/  
**GitHub:** https://github.com/DistrictDataLabs/yellowbrick/tarball/v1.5  
**Docs:** https://pythonhosted.org/yellowbrick/  

**Notes:**
- ⚠ Not updated in 3+ year(s)

---

### 4. `ruptures` — #4
**Fit score:** 3.5/10  
**Safety:** ✅  
**Sources:** llm  

A change point detection library that can be repurposed to detect the elbow point in a scree plot by identifying where the explained variance curve changes slope. | functions: Pelt, Binseg

> While ruptures can technically detect change points in a 1D signal like a scree plot's explained variance curve, it is designed for time-series change point detection rather than elbow/knee detection specifically, making it an indirect and non-ideal solution compared to purpose-built libraries like kneed.

**Key functions/classes:** `Pelt`, `Binseg`  
**Health:** v1.1.10 | updated 170d ago | 977,145 dl/month | BSD-2-Clause  
**PyPI:** https://pypi.org/project/ruptures/  
**GitHub:** https://github.com/deepcharles/ruptures/issues/  
**Docs:** https://github.com/deepcharles/ruptures/  

---


## Excluded Candidates (5)

| Package | Reason |
|---------|--------|
| `scikit-learn` | Known CVEs detected |
| `numpy` | Known CVEs detected |
| `salah_analysis_using_mediapipe_and_opencv_with_creative_and_user_friendly_gui` | Not found on PyPI |
| `pose_detection_using_mediapipe` | Not found on PyPI |
| `automated_salah_pose_tracker_using_mediapipe_and_user_friendly_interface` | Not found on PyPI |
