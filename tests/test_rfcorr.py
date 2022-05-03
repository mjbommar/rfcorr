# imports
import numpy

# project imports
import rfcorr.cat
import rfcorr.extra_trees
import rfcorr.random_forest
import rfcorr.xgboost
from rfcorr.sign import sign_without_zero, sign_with_zero
from rfcorr import __version__

# setup RNG
SEED = 13785
rs = numpy.random.RandomState(SEED)


def test_version():
    assert __version__ == "0.1.0"


def test_rf_classification():
    """
    Test basic random forest classification
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.random_forest.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        num_trees=10,
        lag=0,
        method="classification",
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_rf_regression():
    """
    Test basic random forest regression
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.random_forest.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        num_trees=10,
        lag=0,
        method="regression",
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_rf_classification_permu():
    """
    Test basic random forest classification
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.random_forest.get_pairwise_corr(
        numpy.vstack((sign_without_zero(x), sign_without_zero(y1))).T,
        num_trees=10,
        lag=0,
        method="classification",
        use_permutation=True,
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_rf_regression_permu():
    """
    Test basic random forest regression
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.random_forest.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        num_trees=10,
        lag=0,
        method="regression",
        use_permutation=True,
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_et_classification():
    """
    Test basic extra trees classification
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.extra_trees.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        num_trees=10,
        lag=0,
        method="classification",
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_et_regression():
    """
    Test basic extra trees regression
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.extra_trees.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        num_trees=10,
        lag=0,
        method="regression",
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_et_classification_permu():
    """
    Test basic extra trees classification
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.extra_trees.get_pairwise_corr(
        numpy.vstack((sign_without_zero(x), sign_without_zero(y1))).T,
        num_trees=10,
        lag=0,
        method="classification",
        use_permutation=True,
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_et_regression_permu():
    """
    Test basic extra trees regression
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    c = rfcorr.extra_trees.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        num_trees=10,
        lag=0,
        method="regression",
        use_permutation=True,
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_catboost_classification():
    """
    Test basic catboost classification
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(0, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(x)

    """
    TODO: file catboost issue for `Object of type RandomState is not JSON serializable`
    """
    c = rfcorr.cat.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        method="auto",
        # random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)


def test_xgboost_regression():
    """
    Test basic xgb classification
    :return:
    """
    # TODO: set these up as fixtures
    x = numpy.arange(-8 * numpy.pi, 8 * numpy.pi, 0.5)
    y1 = numpy.sqrt(numpy.abs(x))

    """
    TODO: file catboost issue for `Object of type RandomState is not JSON serializable`
    """
    c = rfcorr.xgboost.get_pairwise_corr(
        numpy.vstack((x, y1)).T,
        method="regression",
        random_state=rs,
    )

    assert isinstance(c, numpy.ndarray)
