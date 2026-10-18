# Project page

`index.html` is the page; `assets/` holds its styles, scripts and pictures. Static files, no build step.

```
python tools/make_data.py <repository root> assets/js/data.js     # every number the charts draw, from results/runs
python tools/make_scene.py .                                       # the Sentinel-2 scene of the bands section (needs rasterio)
python tools/fetch_images.py <empty folder>                        # a pool of sample photos from the Hugging Face dataset server
python tools/make_images.py <that folder> assets/img
```

`make_data.py` recomputes every paired difference and interval from the saved runs and stops if one of the numbers printed in the
paper no longer comes out the same.

## After the decision

1. In `index.html`, remove the `noindex` line.
2. On the `html` tag change `data-review="blind"` to `data-review="open"`. That hides the "Anonymous for now" notes and shows the block with id `authors`.
3. Fill that block: the author list and the links to the paper, the code and the supplement.

## Credits

Aerial scenes are samples of NWPU-RESISC45 (Cheng, Han and Lu, 2017) and leaves of PlantVillage (Hughes and Salathé, 2015). The scene in the bands
section is Sentinel-2B, tile 32TNQ, 11 July 2024: contains modified Copernicus Sentinel data (2024). RESISC45 is distributed for research
use. Fonts: IBM Plex Sans, Source Serif 4, IBM Plex Mono and Nanum Pen Script, from Google Fonts under the SIL Open Font License.
