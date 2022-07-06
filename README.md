# napari_svetlana

[![License](https://img.shields.io/pypi/l/napari_svetlana.svg?color=green)](https://bitbucket.org/koopa31/napari_svetlana/src/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari_svetlana.svg?color=green)](https://pypi.org/project/napari_svetlana)
[![Python Version](https://img.shields.io/pypi/pyversions/napari_svetlana.svg?color=green)](https://python.org)
[![tests](https://bitbucket.org/koopa31/napari_svetlana/workflows/tests/badge.svg)](https://bitbucket.org/koopa31/napari_svetlana/actions)
[![codecov](https://codecov.io/gh/koopa31/napari_svetlana/branch/main/graph/badge.svg)](https://codecov.io/gh/koopa31/napari_svetlana)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari_svetlana)](https://napari-hub.org/plugins/napari_svetlana)

The aim of this plugin is to classify the output of a segmentation algorithm.
The inputs are :
<ul>
  <li>A folder of raw image</li>
  <li>Their segmentation masks where each ROI has its own label.</li>
</ul>
Svetlana can process 2D, 3D and multichannel image.

If you use this plugin please cite the paper: 

  @article{cazorla2022Svetlana,<br/>
      title={Svetlana blabla},<br/>
      author={Cazorla, Morin, Weiss},<br/>
      journal={Nature Communication},<br/>
      volume={18},<br/>
      number={1},<br/>
      pages={100--106},<br/>
      year={2022},<br/>
      publisher={Nature Publishing Group}
      }


----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

You can install `napari_svetlana` via [pip]:

    pip install napari_svetlana


## Tutorial

To learn more about the features of
Svetlana and how to use it, please check our [Youtube tutorial](https://youtube.com).


## Contributing

Contributions are very welcome.

## License

Distributed under the terms of the [BSD-3] license,
"napari_svetlana" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

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
