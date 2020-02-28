# Visualizing results
Here are some example scripts to illustrate how to do basic visualiztion of Cholla output.

You will likely develop more customized, robust, and flexible scripts for your own usage.
These simple scripts here are intended to help you understand the basics of the generated data from Cholla.

## Merging HDF5 files
Multi-processor runs generate HDF5 files per-timestep per-processor.
To treat each timestep together we want to merge those per-processor HDF5 files.

| Script | Concatenate |
| ------ | ----------- |
`cat_dset_3d.py`    | 3D HDF5 datasets
`cat_projection.py` | The on-axis projection data created when the -DPROJECTION flag is turned on
`cat_rotated_projection.py` | The rotated projection data created when the -DROTATED_PROJECTION flag is turned on
`cat_slice.py` | The on-axis slice data created when the -DSLICES flag is turned on

## Plotting data
We here present simple Python matplotlib-based scripts to plot density, velocity, energy, and pressure.

| Script | Description |
| ------ | ----------- |
`plot_sod.py` | Plot 1D Sod Shock Tube test

Plot ranges are hard-coded to keep all plots on the same scale, but different problems will need completely different ranges.

## Movies
Making plots and movies of simulations is a key part in exploring and sharing the results of your simulations.  There are entire suites and choices of visualization software.  Here are just some simple prescriptions to get basic movies out of plots.

### Make a movie of a set of PNG files.

```
ffmpeg -r 10 -s 1800x1800 -i %d.png -crf 25 -pix_fmt yuv420p test_1d_blast.mp4
```

| Option | Description |
| ------ | ----------- |
-r | frame rate per second
-s | size in PixelxPixel.  The default figure of the plot_1d_blast.py is 6" at 300 dpi which makes an 1800x1800 image.  We keep the full resolution for the movie.
-i | format string describing input PNG filenames.  This is globbed by `ffmpeg`, not expanded on the command line.
-pix_fmt | yuv420p is backward compatible with more viewers.  yuv444p is the default if you don't specify.
<output filename> | Output name of the MP4 file.
