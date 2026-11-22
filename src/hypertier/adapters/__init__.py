# Copyright 2026 The HyperTier Authors
# SPDX-License-Identifier: Apache-2.0

"""Adapters that put models of other libraries on the streaming runtime, without rewriting
their classes. Each needs an optional extra; importing this package imports none of them.
"""

from hypertier.adapters.hf_transformers import StreamedModel, load_causal_lm, load_text_encoder

__all__ = ["StreamedModel", "load_causal_lm", "load_text_encoder"]
