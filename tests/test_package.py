# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

import hypertier


def test_version_is_defined():
    assert isinstance(hypertier.__version__, str)
    assert hypertier.__version__.count(".") == 2
