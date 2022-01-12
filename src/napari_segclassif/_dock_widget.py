"""
This module is an example of a barebones QWidget plugin for napari

It implements the ``napari_experimental_provide_dock_widget`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory
from qtpy.QtWidgets import *
from qtpy.QtGui import *
from qtpy.QtCore import *


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        btn = QPushButton("Click me!")
        btn.clicked.connect(self._on_click)

        self.setLayout(QHBoxLayout())
        self.layout().addWidget(btn)

    def _on_click(self):
        print("napari has", len(self.viewer.layers), "layers")


class Window(QMainWindow):
    """
    This class is the "main" i.e. it implements the application frame containing all the widgets
    """
    def __init__(self, napari_viewer):
        super().__init__()

        self.setWindowTitle("Cells classification GUI")
        self.setWindowIcon(QIcon("intro-brain.png"))
        self.setGeometry(1000, 800, 1000, 800)

        self.viewer = napari_viewer
        self.model_sb = QStatusBar(self)
        self.model_sb.showMessage("Waiting for an action")

        self.gridLayout = QGridLayout(self)
        self.gridLayout.setMargin(80)
        self.Buttons_layout = QGridLayout(self)
        self.viewer_layout = QVBoxLayout()

        self.gridLayout.addLayout(self.Buttons_layout, 0, 0)
        self.gridLayout.addLayout(self.viewer_layout, 0, 1)
        self.viewer_layout.addWidget(self.viewer)
        self.viewer_layout.addWidget(self.model_sb)

        self.setAcceptDrops(True)

        self.widget = QWidget()
        self.widget.setLayout(self.gridLayout)

        self.setCentralWidget(self.widget)
        self.__createFileMenu()
        self.image_counter = 0

        self.image_name_list = []
        self.labels_list = []
        self.LIST = []
        self.tmp_path = "/home/clement/Bureau/resultat_7janvier.xlsx"

    def __createFileMenu(self):
        """
        Creation of the toolbar containing a file and a help menus
        """
        actOpen = QAction(QIcon("icons/open.png"), "&Open", self)
        actOpen.setStatusTip("Open file")
        actOpen.triggered.connect(self.pick_new)

        actSave = QAction(QIcon("icons/save.png"), "&Save before end", self)
        actSave.setStatusTip("Save table")
        actSave.triggered.connect(self.save_before_end)

        actResume = QAction(QIcon("icons/save.png"), "&Resume annotation", self)
        actResume.setStatusTip("Resume annotation")
        actResume.triggered.connect(self.resume_annotation)

        menuBar = QMenuBar(self)
        file = menuBar.addMenu("&File")
        file.addAction(actOpen)
        file.addAction(actSave)
        file.addAction(actResume)

    def resume_annotation(self):

        dialog = QFileDialog()
        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            path = dialog.getOpenFileName(self, "Select folder", options=QFileDialog.DontUseNativeDialog)
        elif platform.system() == 'Windows':
            path = dialog.getOpenFileName(self, "Select folder")

        self.df2 = pd.read_excel(path[0])

        last_element = list(self.df2["image_name"])[-1]
        self.folder_path = os.path.split(list(self.df2["image_name"])[-1])[0]

        self.image_counter = len(self.df2["image_name"])

        self.imagettes = natsorted([f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f)) and
                                    os.path.splitext(f)[0].endswith("imagette")])
        self.contours_imagettes = natsorted(
            [f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f)) and
             os.path.splitext(f)[0].endswith("contours")])

        self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter]))
        self.viewer.update_image()
        print("ok")

    def pick_new(self):
        """
        shows a dialog to choose the image to label
        """
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)

        if platform.system() == 'Linux' or platform.system() == 'Darwin':
            folder_path = dialog.getExistingDirectory(self, "Select folder", QDir.currentPath(), options=QFileDialog.DontUseNativeDialog)
        elif platform.system() == 'Windows':
            folder_path = dialog.getExistingDirectory(self, "Select folder")
        self.folder_path = folder_path

        self.image_counter = 0
        self.imagettes = natsorted([f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f)) and
                          os.path.splitext(f)[0].endswith("imagette")])
        self.contours_imagettes = natsorted([f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f)) and
                                   os.path.splitext(f)[0].endswith("contours")])

        self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter]))
        self.viewer.update_image()
        print("ok")

    def keyPressEvent(self, event):
        """ When Alt pressed, note number will be shown in viewedList. """
        if self.image_counter < len(self.imagettes) - 1:
            if event.key() == 49:
                print("Label 1")
                self.image_name_list.append(join(self.folder_path, self.imagettes[self.image_counter]))
                self.labels_list.append(1)
                self.image_counter += 1
                self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter]))
                self.viewer.update_image()
                self.model_sb.showMessage("Image " + str(self.image_counter) + " over " + str(len(self.imagettes))
                                          + " processed")
                print(self.image_name_list)
                print(self.labels_list)
                self.LIST.append([self.image_name_list[-1], self.labels_list[-1]])
                print(self.LIST)
                if self.image_counter == 1:
                    self.results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                elif hasattr(self, "df2") and self.image_counter == len(self.df2) + 1:
                    self.results_df = self.df2
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                else:
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')

            elif event.key() == 50:
                print("Label 2")
                self.image_name_list.append(join(self.folder_path, self.imagettes[self.image_counter]))
                self.labels_list.append(2)
                self.image_counter += 1
                self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter]))
                self.viewer.update_image()
                print(self.image_name_list)
                print(self.labels_list)
                self.model_sb.showMessage("Image " + str(self.image_counter) + " over " + str(len(self.imagettes))
                                          + " processed")
                self.LIST.append([self.image_name_list[-1], self.labels_list[-1]])
                print(self.LIST)
                if self.image_counter == 1:
                    self.results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                elif hasattr(self, "df2") and self.image_counter == len(self.df2) + 1:
                    self.results_df = self.df2
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                else:
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')

            elif event.key() == 82:
                self.image_name_list.pop()
                self.labels_list.pop()
                self.image_counter -= 1
                self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter]))
                self.viewer.update_image()
                print(self.image_name_list)
                print(self.labels_list)
                self.model_sb.showMessage("Image " + str(self.image_counter) + " over " + str(len(self.imagettes))
                                          + " processed")
                self.LIST.pop()
                print(self.LIST)
                if self.image_counter == 1:
                    self.results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                elif hasattr(self, "df2") and self.image_counter == len(self.df2) + 1:
                    self.results_df = self.df2
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                else:
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')

        elif self.image_counter == len(self.imagettes) - 1:
            if event.key() == 49:
                print("Label 1")
                self.image_name_list.append(join(self.folder_path, self.imagettes[self.image_counter]))
                self.labels_list.append(1)
                self.image_counter += 1
                self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter - 1]))
                self.viewer.update_image()
                print(self.image_name_list)
                print(self.labels_list)
                self.model_sb.showMessage("Image " + str(self.image_counter) + " over " + str(len(self.imagettes))
                                          + " processed, press 1 or 2 to finish")
                self.LIST.append([self.image_name_list[-1], self.labels_list[-1]])
                if self.image_counter == 1:
                    self.results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                elif hasattr(self, "df2") and self.image_counter == len(self.df2) + 1:
                    self.results_df = self.df2
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                else:
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')

            elif event.key() == 50:
                print("Label 2")
                self.image_name_list.append(join(self.folder_path, self.imagettes[self.image_counter]))
                self.labels_list.append(2)
                self.image_counter += 1
                self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter - 1]))
                self.viewer.update_image()
                print(self.image_name_list)
                print(self.labels_list)
                self.model_sb.showMessage("Image " + str(self.image_counter) + " over " + str(len(self.imagettes))
                                          + " processed, press 1 or 2 to finish")
                self.LIST.append([self.image_name_list[-1], self.labels_list[-1]])
                if self.image_counter == 1:
                    self.results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                elif hasattr(self, "df2") and self.image_counter == len(self.df2) + 1:
                    self.results_df = self.df2
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')
                else:
                    self.results_df.loc[len(self.results_df)] = self.LIST[-1]
                    self.results_df.to_excel(self.tmp_path, index=False, engine='xlsxwriter')

        else:
            self.viewer.image = imread("gameover.jpg")
            self.viewer.update_image()
            print("done")
            results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
            path = self.save()
            if hasattr(self, 'df2'):
                frames = [self.df2[["image_name", "label"]], results_df[["image_name", "label"]]]
                res = pd.concat(frames)
                res.to_excel(path, index=False, engine='xlsxwriter')
            else:
                results_df.to_excel(path, index=False, engine='xlsxwriter')

    def save(self, path=False):
        """
        Saves the label image in order not to have to draw them one more time if you work on the same image
        """
        if path is False:
            if "PYCHARM_HOSTED" in os.environ:
                path = QFileDialog.getSaveFileName(
                    self, 'Save labels',
                    '/path/to/file/location',
                    options=QFileDialog.DontUseNativeDialog,
                )[0]
            else:
                path = QFileDialog.getSaveFileName(
                    self, 'Save labels',
                    '/path/to/file/location',
                )[0]
        if path.endswith(".xlsx") is False:
            path += ".xlsx"
        return path

    def save_before_end(self):
        results_df = pd.DataFrame(self.LIST, columns=["image_name", "label"])
        path = self.save()
        results_df.to_excel(path, engine='xlsxwriter')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            if platform.system() == 'Linux' or platform.system() == 'Darwin':
                folder_path = event.mimeData().urls()[0].url()[7:]
            elif platform.system() == 'Windows':
                folder_path = event.mimeData().urls()[0].url()[8:]
        self.folder_path = folder_path

        self.image_counter = 0
        self.imagettes = natsorted([f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f)) and
                          os.path.splitext(f)[0].endswith("imagette")])
        self.contours_imagettes = natsorted([f for f in listdir(self.folder_path) if isfile(join(self.folder_path, f)) and
                                   os.path.splitext(f)[0].endswith("contours")])

        self.viewer.image = imread(join(self.folder_path, self.contours_imagettes[self.image_counter]))
        self.viewer.update_image()
        print("ok")


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    # you can return either a single widget, or a sequence of widgets
    return [ExampleQWidget, Window, example_magic_widget]
