# Python environment

The supported installation now uses one Python 3.11–3.13 environment for inference,
training, the local server and tests. Follow the [README installation](../README.md).
Dependencies and the tested Ultralytics version are in `pyproject.toml`.

During release preparation the existing local `.venv` (Python 3.12.2, Ultralytics
8.4.23, torch 2.10.0+cu128, rasterio 1.5.0) was retained, and test/build dependencies
and zstandard were added. The full suite, including real bundled-example inference,
was run in that environment. The separately requested anaconda test command is
also retained as an additional check; it skips tests requiring unavailable model
or raster dependencies. New installations do not need two interpreters.

A historical SQLite shared-library issue affected this machine's Conda-derived
Python. OpenDarkSTS's local server uses the standard HTTP server and JSON records;
a machine-specific SQLite preload is not part of the supported setup.
