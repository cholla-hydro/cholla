# Visualizing results
Here are some example scripts and notebooks to illustrate how to perform basic visualization of Cholla output.

You will likely develop more customized, robust, and flexible scripts for your own usage.
These simple scripts here are intended to help you understand the basics of the generated data from Cholla.

## Using the `cholla_utils` python package

In this subsection we highlight a Jupyter notebook that uses the `cholla_utils` python package for plotting a 3D run.

| Notebook | Description |
| ------ | ----------- |
| `Projection_Slice_Tutorial.ipynb` | demonstrates how to analyze data with `cholla_utils` |

We **strongly** encourage you to make use of the `cholla_utils`, because it's compatible with concatenated datasets (the new format and old format), and distributed datasets. Furthermore, any changes introduced into Cholla's file-format will also be introduced into the functions in this module. In other words, the `cholla_utils` functions aim to be forward and backwards compatible.

At the time of writing `cholla_utils` doesn't support 1D or 2D datasets. We provide some guidance down below.

## Manually Plotting 1D Data
We here present simple Python matplotlib-based scripts to plot density, velocity, energy, and pressure for 1D datasets.

At the time of writing, the `cholla_utils` python package doesn't understand 1D or 2D datasets. So the included script illustrates how to directly use `h5py`

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
