""" unit tests for workexperience2024.core module (no external requirements)"""

import os

import pytest


def resource_unavailable() -> bool:
    """
    Checks whether these a resource is unavailable because the tests are
    being run on a CI pipeline runner.

    This is to support skipping specific tests where required, and works by
    checking for a `RUNNER` variable within the environment, which is set to
    `True` during CI environment instantiation.

    Returns:
        True if the job is running on a CI pipeline, or False otherwise.
    """
    return bool(os.environ.get("RUNNER"))


def platform_dependent() -> bool:
    """
    Checks whether the tests are being run on a platform, such as JupyterHub.

    This is to support skipping specific tests where required, and works by
    checking for a `PLATFORM ENVAR` variable within the environment, which
    is characteristic of a platform. This is obviously a dummy name.

    Returns:
        True if the job is running on a certain platform, or False otherwise.
    """
    return bool(os.environ.get("PLATFORM ENVVAR"))


def skipUnlessHasattr(obj, attr):
    if hasattr(obj, attr):
        return lambda func: func
    return pytest.mark.skip(
        reason="object {!r} doesn't have attribute {!r}".format(obj, attr)
    )


def skipIfLocal():
    if not resource_unavailable():
        return lambda func: func
    return pytest.mark.skip(reason="Running Locally, (decorated skip)")


def skipIfRunner():
    if not resource_unavailable():
        return lambda func: func
    return pytest.mark.skip(
        reason="On Runner, resource is unavailable (decorated skip)"
    )


def skipIfGoodReason():
    reason = []
    if not resource_unavailable():
        return lambda func: func
    else:
        reason.append("Resource Unavailable")

    if platform_dependent():
        return lambda func: func
    else:
        reason.append("Platform isn't right")

    return pytest.mark.skip(
        reason=f"There is a good reason for skipping this: {', '.join(reason)}"
    )


class TestMySkippingTestCasePytest:
    def setUp(self) -> None:
        self.attribute = True

    @pytest.mark.skip(reason="Demonstrating skipping.")
    def test_nothing(self):
        pytest.fail("Shouldn't happen")

    @pytest.mark.skipif(os.environ.get("RUNNER") is None, reason="Running locally.")
    def test_local(self):
        # Tests that work locally.
        pass

    @pytest.mark.skipif(
        os.environ.get("RUNNER") is not None, reason="Running on the remote runner."
    )
    def test_not_remote(self):
        # Tests that don't work remotely.
        pass

    @pytest.mark.skipif(
        os.environ.get("RUNNER") is None, reason="Not on the remote runner."
    )
    def test_remote(self):
        # Tests that work remotely.
        pass

    @pytest.mark.skipif(resource_unavailable(), reason="Resource not available.")
    def test_maybe_skipped(self):
        # test code that depends on the  resource.
        pass

    @pytest.mark.skipif(
        os.environ.get("PLATFORM ENVVAR") is not None,
        reason="Running on environment with constraints.",
    )
    def test_platform(self):
        # Tests that don't work on specific platform or environment.
        pass

    @pytest.mark.skipif(resource_unavailable(), reason="Resource not available.")
    @pytest.mark.skipif(
        platform_dependent(), reason="Resource not available on this platform."
    )
    def test_skipped_if_or(self):
        pass

    @skipIfGoodReason()
    def test_skipped_if_or_decorated(self):
        print("No good reason.")
        # Tests that don't work on specific platform or environment.
        pass

    @skipIfLocal()
    def test_remote_decorated(self):
        # Tests that work remotely.
        pass

    @skipIfRunner()
    def test_not_runner_decorated(self):
        # Tests that don't work remotely.
        pass

    @pytest.mark.xfail
    def test_fail(self):
        assert 1 == 0

    @skipUnlessHasattr(1, "FOO")
    def test_decorated(self):
        # Some tests that need obj.Foo
        pass
