# Third-party notices

Leaflet 1.9.4 is distributed under the BSD 2-Clause licence. Its copyright and
licence text are retained in `scripts/templates/static/vendor/leaflet/LICENSE`
in the installed package. Source: https://github.com/Leaflet/Leaflet/tree/v1.9.4

Leaflet.draw 1.0.4 is distributed under the MIT licence. Its copyright and
licence text are retained in `scripts/templates/static/vendor/leaflet-draw/LICENSE`
in the installed package. Source: https://github.com/Leaflet/Leaflet.draw/tree/v1.0.4

OpenStreetMap tiles carry attribution in each displayed map:
https://www.openstreetmap.org/copyright

IBM Plex Sans is distributed under the SIL Open Font License 1.1. The font files
and licence are retained in `scripts/templates/static/vendor/ibm-plex/` in the
installed package. Source: https://github.com/IBM/plex (obtained through Google Fonts).

The bundled example contains modified Copernicus Sentinel data 2020 and NOAA
Office for Coastal Management AIS records. Its README and provenance JSON identify
the transforms, source hashes, dates and terms. No satellite product identifier
or exact acquisition time was recovered from the historical source tile.

Ultralytics 8.4.23 is an external dependency under AGPL-3.0; OpenDarkSTS is
licensed under AGPL-3.0-only (see `LICENSE`). Python dependencies
remain governed by their respective licences. See repository `docs/licensing.md`.

The default `opendarksts_yolo11s_v4_best.pt` checkpoint is fine-tuned from
Ultralytics' COCO-pretrained YOLO11s on the balanced v4 dataset, derived from the
Madhulika/Roboflow version-7 source set and manually reviewed Sentinel-2 imagery.
Source metadata declares Public Domain, and the source-set author has agreed to
publication. The model card and weight recreation document record
provenance, test-split limitations and the release asset. The later
clean-split YOLO26n checkpoint is an optional alternative, not the package default.
