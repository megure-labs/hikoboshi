from __future__ import annotations

import hikoboshi as hiko

from helpers import require


require(hiko.__version__ == "0.1.0", "unexpected package version")
require(callable(hiko.encode), "encode namespace must be callable")
require(callable(hiko.pairwise), "pairwise namespace must be callable")
require(callable(hiko.all_vs_all), "all_vs_all namespace must be callable")
require(callable(hiko.info), "info must be callable")
require(callable(hiko.version_info), "version_info must be callable")
require(hiko.version_info()["product_name"] == "Hikoboshi", "version_info product mismatch")
capabilities = hiko.info()["backend_capabilities"]
require(capabilities["cpu"]["scalar"]["runtime_available"], "scalar backend missing")
