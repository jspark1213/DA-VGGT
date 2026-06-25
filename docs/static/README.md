# Project-page assets

Drop the media here; the page (`docs/index.html`) already references these paths.

## Figures (`static/images/`)
- `figure1.png` — PDF Figure 1 (motivation)
- `figure2.png` — PDF Figure 2 (method overview)
- `pipeline_poster.png` — *(optional)* poster frame shown before the pipeline video plays

Export the figures from the paper PDF as PNG (transparent or white background).
Then in `index.html`, replace each `placeholder-box` in the **Main Contribution**
section with:

```html
<figure class="fig"><img src="static/images/figure1.png" alt="Figure 1"></figure>
```

## Videos (`static/videos/`)
- `pipeline.mp4` — animated walkthrough of PDF Figure 3 (H.264 MP4 recommended)

Then in `index.html`, in the **Pipeline** section, delete the `placeholder-box`
and uncomment the `<video>` block right below it.

## Point clouds (`static/pointclouds/`)
For the interactive viewer. One **ours** + one **baseline** `.ply` per scene:
- `scene1_ours.ply`, `scene1_vggt.ply`
- `scene2_ours.ply`, `scene2_vggt.ply`
- … add more and edit the `SCENES` array near the bottom of `index.html`

Export as binary or ASCII `.ply` with per-point XYZ (and RGB if available — the
viewer auto-detects vertex colors). Keep point counts reasonable (≈200k–1M) so the
page stays smooth in-browser; downsample dense clouds before exporting.

The viewer (Three.js + OrbitControls) loads these live: drag to rotate, scroll to
zoom, right-drag to pan. Until a file exists it shows a "Point cloud not found"
message instead of breaking.
