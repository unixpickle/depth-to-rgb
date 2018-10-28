# depth-to-rgb

The problem is simple: encode a 16-bit depth image as a 24-bit RGB image, such that applying compression to the RGB image doesn't totally destroy the 16-bit depth reconstructions.

# Results

Here are the results of running various algorithms on the test image(s):

| Name | 10% quality | 50% quality | 100% quality |
|:----:|:-----------:|:-----------:|:------------:|
| grayscale | 914.77 | 292.61 | 106.24 |
| wrapbit | 248.71 | 90.67 | 43.85 |
| halftone-cheat | 12.97 | 3.36 | 2.20 |
