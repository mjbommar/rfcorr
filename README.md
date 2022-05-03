<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<div align="center">
<h3 align="center">rfcorr - Random Forest-based "Correlation" measures</h3>
  <p align="center">
    This library records an open research agenda related to alternative conceptions of correlation based on tree-based ensembles, i.e., "random forests."     
    <br />
    <br />
    <strong>Author:</strong> <a href="https://www.linkedin.com/in/bommarito/">Michael Bommarito</a>
    <br />
    <strong>Project Homepage:</strong> <a href="https://github.com/mjbommar/rfcorr">GitHub</a>
    <br />
    <strong><a href="https://www.linkedin.com/posts/bommarito_github-mjbommarrfcorr-random-forest-based-activity-6899361292889460736-HhKp">Original Announcement</a></strong>
    <br />
   <strong><a href="https://pypi.org/project/rfcorr/">PyPI</a></strong>
  </p>
</div>

## INSTALL
```bash
$ pip install rfcorr
```

## USE
```python
import rfcorr.random_forest

# df = pandas.DataFrame of data with features/variables in columns
rfcorr.random_forest.get_pairwise_corr(df.values,
                                       num_trees=100, # number of trees in forest - bigger => tighter estimates
                                       lag=0, # lag feature-target variable => allows for asymmetric R(x,y) != R(y,x)
                                       method="regression", # estimate using regression or classification task
                                       use_permutation=True # permutation- or impurity-based importance estimates
)
```

## WHY?
Countless tasks rely on conceptions and formalizations of "correlation."  But in two decades of working in areas that utilize correlation
as key to their everyday operation, e.g., finance, I have found that few are measuring what their words reveal is their intuition.

While others have offered alternative measures of correlation or dependence generally, this is my contribution - an approach based on tree-based ensembles that natively supports lagged correlation.  
Tree-based ensembles have a number of highly-favorable properties:

 * Support for categorical features or mixed feature sets
 * Intrinsic uncertainty estimation and intervals through ensemble construction
 * Support for overfitting protection
 * High degree of flexibility
 * Estimation produces inferential models that can be used to make predictions

Additionally, this library supports lagging inputs against targets in the supervised training process, enabling asymmetric correlation estimates 
from the same data.

There are, however, downsides:

 * Slower estimation than other correlation methods
 * Stochastic estimates (though this library supports fixing RNG)
 * Question around interpretation of signedness or directionality
 * More complex interpretation than linear correlation measures
 * Scaling permutation-based estimates
 * Estimating covariance in asymmetric contexts 

## FUNCTIONALITY

 * Random Forest (`rfcorr.random_forest`)
   - `get_corr_classification`: Correlation from classification task
   - `get_corr_regression`: Correlation from regression task
   - `get_corr`: Convenience handler including lag support for (x, y)
   - `get_pairwise_corr`: Convenience handler including lag support for full matrix X
   - Support for impurity-based or permutation-based importances (`use_permutation=True`)
 * Extra Trees (`rfcorr.extra_trees`)
   - `get_corr_classification`: Correlation from classification task
   - `get_corr_regression`: Correlation from regression task
   - `get_corr`: Convenience handler including lag support for (x, y)
   - `get_pairwise_corr`: Convenience handler including lag support for full matrix X
   - Support for impurity-based or permutation-based importances (`use_permutation=True`)
 * CatBoost (`rfcorr.cat`) (WIP)
   - `get_corr_classification`: Correlation from classification task
   - `get_corr_regression`: Correlation from regression task
   - `get_corr`: Convenience handler including lag support for (x, y)
   - `get_pairwise_corr`: Convenience handler including lag support for full matrix X
   - Support for GPU training and limited subset of catboost training parameters
 * xgboost (`rfcorr.xgboost`) (WIP)
   - `get_corr_classification`: Correlation from classification task
   - `get_corr_regression`: Correlation from regression task
   - `get_corr`: Convenience handler including lag support for (x, y)
   - `get_pairwise_corr`: Convenience handler including lag support for full matrix X
 * TODO: Histogram-based Gradient Boosting Trees
 * TODO: Gradient-Boosting Trees
 * TODO: Support exposing intervals (std, range) from permutation-based estimates
  
## EXAMPLE USE

There are sample notebooks in the `notebooks/` directory, including:
* `notebooks/test_sector_etf.ipynb`: "Correlation" and eigenvalue/spectral representations for SPDR Sector ETFs and SPY 
* `notebooks/test_sector_etf_ts.ipynb`: Rolling "Correlation" time series for SPDR Sector ETFs and SPY
* `notebooks/test_periodic_pathological.ipynb`: Test of periodic (`sin(x)`) data with pathological results for Pearson/Spearman

Sample usage looks like this:
```python
import numpy
import pandas
import rfcorr.random_forest

# create sample data
x = numpy.arange(0, 8*numpy.pi, 0.1)
y1 = numpy.sqrt(x)
y2 = numpy.sin(x)

# fix random state/RNG
rs = numpy.random.RandomState(42)
pandas.DataFrame(rfcorr.random_forest.get_pairwise_corr(df.values, 
                                                      num_trees=1000,
                                                      lag=0,
                                                      method="regression", 
                                                      use_permutation=True,
                                                      random_state=rs),
                 columns=["x", "y1", "y2"],
                 index=["x", "y1", "y2"])
"""
x	y1	y2
x	1.000000	1.919737	0.001276
y1	1.965436	1.000000	0.003697
y2	0.649579	0.628396	1.000000
"""
#NB: ~0 correlation for x~y2 and y1~y2

# compare with pearson
df = pandas.DataFrame(zip(x, y1, y2), columns=["x", "y1", "y2"])
df.corr(method="pearson")

"""
	x	y1	y2
x	1.000000	0.978639	-0.194091
y1	0.978639	1.000000	-0.206973
y2	-0.194091	-0.206973	1.000000
"""

# compare with spearman
df.corr(method="spearman")
"""
x	y1	y2
x	1.000000	1.000000	-0.186751
y1	1.000000	1.000000	-0.186751
y2	-0.186751	-0.186751	1.000000
"""


```

## HISTORY
 * 0.1.0, 2022-02-22: Initial PyPI release
 * 0.1.1, 2022-05-02: catboost support (available on GH as of 2022-03-28)
 * 0.1.2, 2022-05-02: xgboost support

## LICENSE
Apache 2.0

## COLLABORATION

I'm currently working on a brief research note that should be on arxiv by March 2022.  I'd love to collaborate with anyone interested on the topic,
especially to bring broader perspective to backtesting, portfolio construction, and regime detection/timing applications.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/licensio/responsible-data-use-policy.svg?style=for-the-badge

[contributors-url]: https://github.com/licensio/responsible-data-use-policy/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/licensio/responsible-data-use-policy.svg?style=for-the-badge

[forks-url]: https://github.com/licensio/responsible-data-use-policy/network/members

[stars-shield]: https://img.shields.io/github/stars/licensio/responsible-data-use-policy.svg?style=for-the-badge

[stars-url]: https://github.com/licensio/responsible-data-use-policy/stargazers

[issues-shield]: https://img.shields.io/github/issues/licensio/responsible-data-use-policy.svg?style=for-the-badge

[issues-url]: https://github.com/licensio/responsible-data-use-policy/issues

[license-shield]: https://img.shields.io/github/license/licensio/responsible-data-use-policy.svg?style=for-the-badge

[license-url]: https://github.com/licensio/responsible-data-use-policy/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://linkedin.com/in/linkedin_username

[product-screenshot]: images/screenshot.png
