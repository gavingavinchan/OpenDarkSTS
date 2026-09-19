# Licence and data rights review

The release plan proposes **AGPL-3.0-only** for the program. This proposal is not yet
an agreed project licence: contributor consent must be recorded before publication.
Ultralytics is distributed under AGPL-3.0 or a separate commercial agreement; its
[licensing page](https://www.ultralytics.com/license) and the
[AGPL text](https://www.gnu.org/licenses/agpl-3.0.html) are the primary references.
The package deliberately omits a licence declaration until agreement is recorded.

The small example includes modified Copernicus Sentinel data 2020 and NOAA AIS.
The [example provenance](../opendarksts/example/README.md) records the sources,
transformations, hashes and gaps. Access to Sentinel data is covered by the
[Copernicus terms](https://dataspace.copernicus.eu/terms-and-conditions) and the
linked Sentinel legal notice. NOAA records are described in its
[AIS metadata](https://www.fisheries.noaa.gov/inport/item/77594).

Leaflet 1.9.4 is vendored for the map under its BSD 2-Clause licence, retained at
`scripts/templates/static/vendor/leaflet/LICENSE`. Map tiles are fetched from
OpenStreetMap with visible attribution; local image review and numeric area entry
remain usable without map tiles. Other Python dependencies retain their own licences.

Training inputs include Roboflow project `fleetdatascience/data-science-dark-fleet`,
version 7, whose local metadata declares Public Domain, plus manually reviewed
satellite images. A declaration in an export does not establish the provenance of
every upstream image or annotation. The full annotation/split release requires a
source-rights review before it is published. The model card records this gap.
