<!--
 Copyright 2022 Victor I. Afolabi

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# U-Net: Convolutional Neural Networks for Image Segmentation

The [U-Net] is a convolutional network architecture for fast and precise segmentation of images.

![U-Net Architecture](res/u-net-architecture.png)

## Image segmentation?

In an image classification task, the network assigns a label (or class) to each input image. However, suppose you want to know the shape of the object, which pixel belongs to which object, etc. In this case, you need to assign a class to each pixel of the image -- this task is known as **segmentation**. A segmentation model returns much more detailed information about the image. Image segmentation has many applications in medical imaging, self-driving cars and satallite imaging, just to name a few.

We use the [Oxford-IIIT Pet Dataset] ([Parkhi et al, 2012]). The dataset consists of images of 37 pet breeds, with 200 images per breed (~100 each in the training and test splits). Each image includes the corresponding labels, and pixel-wise masks. The masks are class-labels for each pixel. Each pixel is given one of three categories.

- Class 1: Pixel belonging to the pet.
- Class 2: Pixel bordering the pet.
- Class 3: None of the above/a surrounding pixel.


## Citation

```txt
@InProceedings{RFB15a,
  author       = "O. Ronneberger and P.Fischer and T. Brox",
  title        = "U-Net: Convolutional Networks for Biomedical Image Segmentation",
  booktitle    = "Medical Image Computing and Computer-Assisted Intervention (MICCAI)",
  series       = "LNCS",
  volume       = "9351",
  pages        = "234--241",
  year         = "2015",
  publisher    = "Springer",
  note         = "(available on arXiv:1505.04597 [cs.CV])",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2015/RFB15a"
}
```

[U-Net]: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
[Oxford-IIIT Pet Dataset]: https://www.robots.ox.ac.uk/%7Evgg/data/pets/
[Parkhi et al, 2012]: https://www.robots.ox.ac.uk/%7Evgg/publications/2012/parkhi12a/parkhi12a.pdf