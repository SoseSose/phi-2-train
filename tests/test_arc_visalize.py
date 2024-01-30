import numpy as np
from settings_arc import CH_source, MAX_SIZE
from arc_visualize import plot_some, plot_task
import pytest
from pathlib import Path

class TestPlotSome:
    test_image = np.tile(np.arange(CH_source + 2), (2, 6, 1))

    def test_normal_plot(self):
        plot_some(self.test_image, "original", show_num=True)

    @pytest.mark.parametrize("fold", [2, 3, 4, 5, 6, 7, 8, 9, 10])
    def test_fold(self,fold):
        plot_some(self.test_image + 4, "original", show_num=True, fold=fold)

    def test_padded_plot(self):
        plot_some(self.test_image, "padded image", pad=(MAX_SIZE, MAX_SIZE))

    def test_one_hot_plot(self):
        def one_hot(x: np.ndarray, depth):
            return np.identity(depth)[x]

        one_hot_img = one_hot(self.test_image, CH_source + 2)
        plot_some(one_hot_img, "image", pad=(MAX_SIZE, MAX_SIZE))

    def test_save_path(self):
        file_name = "test.png"

        plot_some(
            self.test_image,
            "test save",
            pad=(MAX_SIZE, MAX_SIZE),
            save_file_name="test.png",
        )
        file_path = Path(file_name)
        if not file_path.exists():
            raise ValueError("file not saved")
        else:
            file_path.unlink()

def test_plot_task():

    test_image = np.tile(np.arange(CH_source + 2), (2, 6, 1))
    plot_task(
        train=[{"input": test_image, "output": test_image}],
        test=[{"input": test_image, "output": test_image}],
        fold=10,
    )