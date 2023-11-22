from ipywidgets import GridspecLayout

from .viewer import Plot, Settings, MeshShading, LineShading, PointShading


class PlotGrid:
    def __init__(
        self,
        rows: int = 1,
        cols: int = 1,
        *,
        settings: Settings = Settings(),
        mesh_shading: MeshShading = MeshShading(),
        line_shading: LineShading = LineShading(),
        point_shading: PointShading = PointShading(),
        bbox_shading: LineShading = LineShading(line_color="blue"),
    ):
        self.__layout = GridspecLayout(rows, cols, justify_items="center", align_items="center")
        self.__plots: list[list[Plot]] = []
        for i in range(rows):
            row: list[Plot] = []
            for j in range(cols):
                viewer = Plot(
                    settings=settings,
                    mesh_shading=mesh_shading,
                    line_shading=line_shading,
                    point_shading=point_shading,
                    bbox_shading=bbox_shading,
                )
                self.__layout[i, j] = viewer._renderer
                row.append(viewer)
            self.__plots.append(row)

    def __getitem__(self, indices: tuple[int, int]) -> Plot:
        return self.__plots[indices[0]][indices[1]]

    def _repr_mimebundle_(self, **kwargs):
        return self.__layout._repr_mimebundle_(**kwargs)
