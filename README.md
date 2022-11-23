# napari_svetlana

[![License](https://img.shields.io/pypi/l/napari_svetlana.svg?color=green)](https://bitbucket.org/koopa31/napari_svetlana/src/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari_svetlana.svg?color=green)](https://pypi.org/project/napari_svetlana)
[![Python Version](https://img.shields.io/pypi/pyversions/napari_svetlana.svg?color=green)](https://python.org)
[![tests](https://bitbucket.org/koopa31/napari_svetlana/workflows/tests/badge.svg)](https://bitbucket.org/koopa31/napari_svetlana/actions)
[![codecov](https://codecov.io/gh/koopa31/napari_svetlana/branch/main/graph/badge.svg)](https://codecov.io/gh/koopa31/napari_svetlana)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-svetlana)](https://napari-hub.org/plugins/napari-svetlana)
[![Documentation](https://readthedocs.org/projects/svetlana-documentation/badge/?version=latest)](https://svetlana-documentation.readthedocs.io/en/latest/)

The aim of this plugin is to classify the output of a segmentation algorithm.
The inputs are :
<ul>
  <li>A folder of raw images</li>
  <li>Their segmentation masks where each ROI has its own label.</li>
</ul>

Svetlana can process 2D, 3D and multichannel image. If you want to use it to work on cell images, we strongly
recommend the use of [Cellpose](https://www.cellpose.org) for the segmentation part, as it provides excellent quality results and a standard output format
accepted by Svetlana (labels masks). 

If you use this plugin please cite the paper: 

```bibtex
@InProceedings{2022_cazorla801,
	author = "Clément Cazorla and Pierre Weiss and Renaud Morin",
	title = "SVETLANA: UN CLASSIFIEUR DE SEGMENTATION POUR NAPARI",
	booktitle = "28° Colloque sur le traitement du signal et des images",
	year = "2022",
	publisher = "GRETSI - Groupe de Recherche en Traitement du Signal et des Images",
	number = "001-0194",
	pages = "p. 777-780",
	month = "Sep # 6--9",
	address = "Nancy",
	doi = "",
	pdf = "2022_cazorla801.pdf",
}
```


![](https://bitbucket.org/koopa31/napari_svetlana/raw/bb1010b99e03cdbd94d1cd70cc63f93deb63a58e/images/svetlagif.gif)


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

First install Napari in a Python 3.9 Conda environment following the instructions provided
in the official [documentation](https://napari.org/stable/tutorials/fundamentals/installation.html).

You can install `napari_svetlana` via [pip], or directly from the Napari plugin manager (see Napari documentation):
```bash
pip install napari_svetlana
```
WARNING:

If you have a Cuda compatible GPU on your computer, some computations may be accelerated
using [Cupy](https://pypi.org/project/cupy/). Unfortunately, Cupy needs Cudatoolkit to be installed. This library can only be installed via 
Conda while the plugin is a pip plugin, so it must be installed manually for the moment:
```bash
conda install cudatoolkit=10.2 
```
Also note that the library ([Cucim](https://pypi.org/project/cucim/)) that we use to improve these performances, computing morphological operations on GPU
is unfortunately only available for Linux systems. Hence, if you are a Windows user, this installation is not necessary.

## Tutorial

To learn more about the features of
Svetlana and how to use it, please check our [Youtube tutorial](https://youtube.com) and
our [documentation](https://svetlana-documentation.readthedocs.io/en/latest/).
A folder in this repository called "[Demo images](https://bitbucket.org/koopa31/napari_svetlana/src/main/Demo%20images/)", contains two demo images, similar to the ones
of the Youtube tutorial. Feel free to use them to test all the features of Sevtlana.

## The data augmentation 

It is possible to perform all the complex data augmentations proposed in the Albumentations
library. To do so, please refer to the [documentation](https://albumentations.ai/docs/getting_started/transforms_and_targets/),
and add all the needed parameters to the JSON configuration file.

**Example:**

Gaussian blurring in documentation :

```python
GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5)
```

Equivalent in JSON configuration file:
```json
"GaussianBlur": {
      "apply": "False",
      "blur_limit": "(3, 7)",
      "sigma_limit": "0", 
      "p": "0.5"
  }
```

where _apply_ means you want this data augmentation to be applied or not.

## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"napari_svetlana" is free and open source software

## Acknowledgements

The method was developed by [Clément Cazorla](https://koopa31.github.io/), [Renaud Morin](https://www.linkedin.com/in/renaud-morin-6a42665b/?originalSubdomain=fr) and [Pierre Weiss](https://www.math.univ-toulouse.fr/~weiss/). And the plugin was written by
Clément Cazorla. The project is co-funded by [Imactiv-3D](https://www.imactiv-3d.com/) and [CNRS](https://www.cnrs.fr/fr).

## Issues

If you encounter any problems, please [file an issue](https://bitbucket.org/koopa31/napari_svetlana/issues?status=new&status=open) along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
