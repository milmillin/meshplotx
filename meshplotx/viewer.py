import numpy as np
from ipywidgets import embed
import pythreejs as p3s
from IPython.display import display
import uuid
from typing import Literal, TypedDict, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path

from .utils import get_colors, gen_circle, gen_checkers


@dataclass
class Settings:
    width: int = 600
    height: int = 600
    antialias: bool = True
    scale: float = 1.5
    background: str = "#ffffff"
    fov: float = 30  # degrees


@dataclass
class MeshShading:
    flat: bool = True
    wireframe: bool = False
    wire_width: float = 0.03
    wire_color: str = "black"
    side: Literal["FrontSide", "BackSide", "DoubleSide"] = "DoubleSide"
    colormap: str = "viridis"
    v_range: Optional[tuple[float, float]] = None  # for (vmin, vmax) for color value
    bbox: bool = False
    roughness: float = 0.5
    metalness: float = 0.25
    reflectivity: float = 1.0


@dataclass
class LineShading:
    line_width: float = 1.0
    line_color: str = "black"


@dataclass
class PointShading:
    point_color: str = "red"
    point_size: float = 0.01
    point_shape: Literal["circle"] = "circle"
    colormap: str = "viridis"
    v_range: Optional[tuple[float, float]] = None  # for (vmin, vmax) for color value


@dataclass
class _LineObject:
    geometry: p3s.LineSegmentsGeometry
    mesh: p3s.LineSegments2
    material: p3s.LineMaterial
    max: np.ndarray  # (3,)
    min: np.ndarray  # (3,)
    shading: LineShading


@dataclass
class _PointObject:
    geometry: p3s.BufferGeometry
    mesh: p3s.Points
    material: p3s.PointsMaterial
    max: np.ndarray  # (3,)
    min: np.ndarray  # (3,)
    shading: PointShading
    v: np.ndarray


@dataclass
class _MeshObject:
    geometry: p3s.BufferGeometry
    mesh: p3s.Mesh
    material: p3s.MeshStandardMaterial
    max: np.ndarray  # (3,)
    min: np.ndarray  # (3,)
    shading: MeshShading
    coloring: Literal["VertexColors", "FaceColors"]
    v: np.ndarray
    f: np.ndarray
    c: np.ndarray
    wireframe: Optional[p3s.LineSegments]
    bbox: Optional[tuple[_LineObject, np.ndarray, np.ndarray]]


ObjectID = int


class Plot:
    def __init__(
        self,
        v: Optional[np.ndarray] = None,
        f: Optional[np.ndarray] = None,
        c: Optional[np.ndarray] = None,
        *,
        uv: Optional[np.ndarray] = None,
        n: Optional[np.ndarray] = None,
        texture_data: Optional[np.ndarray] = None,
        settings: Settings = Settings(),
        mesh_shading: MeshShading = MeshShading(),
        line_shading: LineShading = LineShading(),
        point_shading: PointShading = PointShading(),
        bbox_shading: LineShading = LineShading(line_color="blue"),
    ):
        """
        Create a mesh viewer.

        Plot Mesh
        - v: (n_V, 3) vertices
        - f: (n_F, 3) faces indexed into v
        - c: (3,), (n_V, 3), (n_F, 3), (n_V,), or (n_F,) colors
        - uv: TODO
        - n: (n_V, 3) normal
        - texture_data: TODO

        Plot Edges
        - v: (n_V, 3) vertices
        - f: (n_E, 2) edges indexed into v

        Plot Points
        - v: (n_V, 3) vertices
        - c: (3,), (n_V, 3), or (n_V,) colors
        """
        self.__s = settings
        self.__ms = mesh_shading
        self.__ls = line_shading
        self.__ps = point_shading
        self.__bs = bbox_shading

        self._light = p3s.DirectionalLight(color="white", position=[0, 0, 1], intensity=0.6)  # type: ignore
        self._light2 = p3s.AmbientLight(intensity=0.5)  # type: ignore
        self._cam = p3s.PerspectiveCamera(
            position=[0, 0, 1],
            lookAt=[0, 0, 0],
            fov=self.__s.fov,  # type: ignore
            aspect=self.__s.width / self.__s.height,  # type: ignore
            children=[self._light],
        )
        self._orbit = p3s.OrbitControls(controlling=self._cam)
        self._scene = p3s.Scene(children=[self._cam, self._light2], background=self.__s.background)  # "#4c4c80"
        self._renderer = p3s.Renderer(
            camera=self._cam,
            scene=self._scene,
            controls=[self._orbit],
            width=self.__s.width,
            height=self.__s.height,
            antialias=self.__s.antialias,
        )

        self.__count: int = 0
        self.__obj: dict[ObjectID, Union[_MeshObject, _LineObject, _PointObject]] = {}

        if v is not None:
            if f is None:
                self.add_points(v, c)
            elif len(f.shape) == 2 and f.shape[1] == 2:
                self.add_edges(v, f)
            else:
                self.add_mesh(v, f, c, uv, n, texture_data)

    def add_mesh(
        self,
        v: np.ndarray,
        f: np.ndarray,
        c: Optional[np.ndarray] = None,
        uv: Optional[np.ndarray] = None,
        n: Optional[np.ndarray] = None,
        texture_data: Optional[np.ndarray] = None,
        mesh_shading: Optional[MeshShading] = None,
        bbox_shading: Optional[LineShading] = None,
    ) -> ObjectID:
        """
        v: (n_V, 3)
        f: (n_F, 3)
        c: (3,), (n_V, 3), (n_F, 3), (n_V,), or (n_F,)
        """
        assert len(v.shape) == 2 and v.shape[-1] == 3, "v must have shape (n_V, 3)"
        assert len(f.shape) == 2 and f.shape[-1] == 3, "f must have shape (n_F, 3)"

        # Type adjustment vertices
        v = v.astype("float32", copy=False)

        if mesh_shading is None:
            mesh_shading = self.__ms

        # Color setup
        colors, coloring = self.__get_colors(v, f, c, colormap=mesh_shading.colormap, v_range=mesh_shading.v_range)

        # Type adjustment faces and colors
        c = colors.astype("float32", copy=False)

        # Material and geometry setup
        ba_dict = {"color": p3s.BufferAttribute(c)}
        if coloring == "FaceColors":
            verts = np.zeros((f.shape[0] * 3, 3), dtype="float32")
            for ii in range(f.shape[0]):
                # print(ii*3, f[ii])
                verts[ii * 3] = v[f[ii, 0]]
                verts[ii * 3 + 1] = v[f[ii, 1]]
                verts[ii * 3 + 2] = v[f[ii, 2]]
            v = verts
        else:
            f = f.astype("uint32", copy=False).ravel()
            ba_dict["index"] = p3s.BufferAttribute(f, normalized=False)

        ba_dict["position"] = p3s.BufferAttribute(v, normalized=False)

        if uv is not None:
            uv_: np.ndarray = (uv - np.min(uv)) / (np.max(uv) - np.min(uv))
            if texture_data is None:
                texture_data = gen_checkers(20, 20)
            tex = p3s.DataTexture(data=texture_data, format="RGBFormat", type="FloatType")
            material = p3s.MeshStandardMaterial(
                map=tex,
                reflectivity=mesh_shading.reflectivity,
                side=mesh_shading.side,
                roughness=mesh_shading.roughness,
                metalness=mesh_shading.metalness,
                flatShading=mesh_shading.flat,
                polygonOffset=True,
                polygonOffsetFactor=1,
                polygonOffsetUnits=5,
            )
            ba_dict["uv"] = p3s.BufferAttribute(uv_.astype("float32", copy=False))
        else:
            material = p3s.MeshStandardMaterial(
                vertexColors=coloring,
                reflectivity=mesh_shading.reflectivity,
                side=mesh_shading.side,
                roughness=mesh_shading.roughness,
                metalness=mesh_shading.metalness,
                flatShading=mesh_shading.flat,
                polygonOffset=True,
                polygonOffsetFactor=1,
                polygonOffsetUnits=5,
            )

        if n is not None and coloring == "VertexColors":  # TODO: properly handle normals for FaceColors as well
            ba_dict["normal"] = p3s.BufferAttribute(n.astype("float32", copy=False), normalized=True)

        geometry = p3s.BufferGeometry(attributes=ba_dict)

        if coloring == "VertexColors" and n is None:
            geometry.exec_three_obj_method("computeVertexNormals")
        elif coloring == "FaceColors" and n is None:
            geometry.exec_three_obj_method("computeFaceNormals")

        # Mesh setup
        mesh = p3s.Mesh(geometry=geometry, material=material)

        # Wireframe setup
        wireframe = None
        if mesh_shading.wireframe:
            wf_geometry = p3s.WireframeGeometry(mesh.geometry)  # WireframeGeometry
            wf_material = p3s.LineBasicMaterial(color=mesh_shading.wire_color, linewidth=mesh_shading.wire_width)
            wireframe = p3s.LineSegments(geometry=wf_geometry, material=wf_material)  # type: ignore
            mesh.add(wireframe)

        # Bounding box setup
        bbox = None
        if mesh_shading.bbox:
            v_box, e_box = self.__get_bbox(v)
            if bbox_shading is None:
                bbox_shading = self.__bs
            bbox = self.__compute_line_object(v_box[e_box], bbox_shading)
            mesh.add(bbox.mesh)
            bbox = (bbox, v_box, e_box)

        # Object setup
        mesh_obj = _MeshObject(
            geometry,
            mesh,
            material,
            np.max(v, axis=0),
            np.min(v, axis=0),
            mesh_shading,
            coloring,
            v,
            f,
            c,
            wireframe,
            bbox,
        )

        return self.__add_object(mesh_obj)

    def add_lines(
        self, beginning: np.ndarray, ending: np.ndarray, line_shading: Optional[LineShading] = None
    ) -> ObjectID:
        """
        beginning: (n_E, 3)
        ending: (n_E, 3)
        """
        assert len(beginning.shape) == 2 and beginning.shape[-1] == 3, "Invalid beginning shape."
        assert len(ending.shape) == 2 and ending.shape[-1] == 3, "Invalid ending shape"
        assert len(beginning) == len(ending), "Number of lines mismatched"

        if line_shading is None:
            line_shading = self.__ls

        lines = np.hstack([beginning, ending])
        lines = lines.reshape((-1, 3))
        return self.__add_object(self.__compute_line_object(lines, line_shading))

    def add_edges(
        self,
        vertices: np.ndarray,
        edges: np.ndarray,
        line_shading: Optional[LineShading] = None,
    ) -> ObjectID:
        """
        vertices: (n_V, 3)
        edges: (n_E, 2) indexed into vertices
        """
        assert len(vertices.shape) == 2 and vertices.shape[-1] == 3, "Invalid vertices shape"
        assert len(edges.shape) == 2 and edges.shape[-1] == 2, "Invalid edges shape"

        if line_shading is None:
            line_shading = self.__ls

        return self.__add_object(self.__compute_line_object(vertices[edges].reshape(-1, 3), line_shading))

    def add_points(
        self,
        points: np.ndarray,
        c: Optional[np.ndarray] = None,
        point_shading: Optional[PointShading] = None,
    ) -> ObjectID:
        assert len(points.shape) == 2 and points.shape[-1] == 3, "Invalid points shape"
        points = points.astype("float32", copy=False)

        if point_shading is None:
            point_shading = self.__ps

        mi = np.min(points, axis=0)
        ma = np.max(points, axis=0)

        g_attributes = {"position": p3s.BufferAttribute(points, normalized=False)}
        m_attributes: dict[str, Any] = {"size": point_shading.point_size}

        if point_shading.point_shape == "circle":  # Plot circles
            tex = p3s.DataTexture(data=gen_circle(16, 16), format="RGBAFormat", type="FloatType")
            m_attributes["map"] = tex
            m_attributes["alphaTest"] = 0.5
            m_attributes["transparency"] = True
        else:  # TODO: Plot squares
            pass

        colors, v_colors = self.__get_point_colors(points, c, point_shading)
        if v_colors:  # Colors per point
            m_attributes["vertexColors"] = "VertexColors"
            g_attributes["color"] = p3s.BufferAttribute(colors, normalized=False)

        else:  # Colors for all points
            m_attributes["color"] = colors

        material = p3s.PointsMaterial(**m_attributes)
        geometry = p3s.BufferGeometry(attributes=g_attributes)
        ppoints = p3s.Points(geometry=geometry, material=material)
        point_obj = _PointObject(geometry, ppoints, material, ma, mi, point_shading, points)

        return self.__add_object(point_obj)

    def remove_object(self, obj_id: ObjectID) -> bool:
        if obj_id not in self.__obj:
            print("Invalid object id. Valid ids are: ", list(self.__obj.keys()))
            return False
        obj = self.__obj[obj_id]
        self._scene.remove(obj.mesh)
        del self.__obj[obj_id]
        self.__update_view()
        return True

    def reset(self) -> None:
        for obj in self.__obj.values():
            self._scene.remove(obj.mesh)
        self.__obj = {}
        self.__count = 0
        self.__update_view()

    def update_mesh(
        self,
        oid: ObjectID = 0,
        vertices: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        faces: Optional[np.ndarray] = None,
    ) -> None:
        obj = self.__obj[oid]
        assert isinstance(obj, _MeshObject), f"Object {oid} is not a mesh"
        if vertices is not None:
            if obj.coloring == "FaceColors":
                f = faces if faces is not None else obj.f
                verts = np.zeros((f.shape[0] * 3, 3), dtype="float32")
                for ii in range(f.shape[0]):
                    # print(ii*3, f[ii])
                    verts[ii * 3] = vertices[f[ii, 0]]
                    verts[ii * 3 + 1] = vertices[f[ii, 1]]
                    verts[ii * 3 + 2] = vertices[f[ii, 2]]
                v = verts
            else:
                v = vertices.astype("float32", copy=False)
            obj.geometry.attributes["position"].array = v
            # self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj.geometry.attributes["position"].needsUpdate = True
            obj.v = vertices
            # obj["geometry"].exec_three_obj_method('computeVertexNormals')
        if faces is not None:
            f = faces.astype("uint32", copy=False).ravel()
            # print(obj["geometry"].attributes)
            obj.geometry.attributes["index"].array = f
            # self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj.geometry.attributes["index"].needsUpdate = True
            obj.f = faces
        if colors is not None:
            colors, coloring = self.__get_colors(
                obj.v, obj.f, colors, colormap=obj.shading.colormap, v_range=obj.shading.v_range
            )
            colors = colors.astype("float32", copy=False)
            obj.geometry.attributes["color"].array = colors
            obj.geometry.attributes["color"].needsUpdate = True
            obj.c = colors

    def update_points(
        self, oid: ObjectID, points: Optional[np.ndarray] = None, colors: Optional[np.ndarray] = None
    ) -> None:
        obj = self.__obj[oid]
        assert isinstance(obj, _PointObject), f"Object {oid} is not a point set"
        if points is not None:
            obj.geometry.attributes["position"].array = points
            # self.wireframe.attributes["position"].array = v # Wireframe updates?
            obj.geometry.attributes["position"].needsUpdate = True
            obj.v = points
        else:
            points = obj.v
        if colors is not None:
            new_colors, v_colors = self.__get_point_colors(points, colors, obj.shading)
            if v_colors:
                assert isinstance(new_colors, np.ndarray)
                colors = colors.astype("float32", copy=False)
                obj.geometry.attributes["color"].array = colors
                obj.geometry.attributes["color"].needsUpdate = True
            else:
                # TODO: test
                obj.material.color = new_colors
                obj.material.needsUpdate = True

    def update_edges(self, oid: ObjectID, vertices: np.ndarray, edges: np.ndarray) -> None:
        obj = self.__obj[oid]
        assert isinstance(obj, _LineObject), f"Object {oid} is not an edge set"
        obj.geometry.positions = vertices[edges].astype(np.dtype("float32"))

    def update_lines(self, oid: ObjectID, vertices: np.ndarray, edges: np.ndarray) -> None:
        obj = self.__obj[oid]
        assert isinstance(obj, _LineObject), f"Object {oid} is not an edge set"
        obj.geometry.positions = vertices[edges].astype(np.dtype("float32"))

    # --------------
    # Export

    def display(self) -> None:
        display(self._renderer)

    def to_html(self, imports: bool = True, html_frame: bool = True) -> str:
        # Bake positions (fixes centering bug in offline rendering)
        if len(self.__obj) == 0:
            return ""

        ma = np.zeros((len(self.__obj), 3))
        mi = np.zeros((len(self.__obj), 3))
        for r, obj in enumerate(self.__obj):
            ma[r] = self.__obj[obj].max
            mi[r] = self.__obj[obj].min
        ma = np.max(ma, axis=0)
        mi = np.min(mi, axis=0)
        diag = np.linalg.norm(ma - mi)
        mean = (ma - mi) / 2 + mi
        for _, obj in self.__obj.items():
            if isinstance(obj, _MeshObject):
                obj.geometry.attributes["position"].array -= mean
                if obj.bbox is not None:
                    bbox, _, _ = obj.bbox
                    bbox.geometry.positions -= mean
            elif isinstance(obj, _PointObject):
                obj.geometry.attributes["position"].array -= mean
            else:
                obj.geometry.positions -= mean

        scale = self.__s.scale * (diag)
        self._orbit.target = (0.0, 0.0, 0.0)
        self._cam.lookAt([0.0, 0.0, 0.0])
        self._cam.position = (0.0, 0.0, scale)
        self._light.position = (0.0, 0.0, scale)

        state = embed.dependency_state(self._renderer)

        # Somehow these entries are missing when the state is exported in python.
        # Exporting from the GUI works, so we are inserting the missing entries.
        for k in state:
            if state[k]["model_name"] == "OrbitControlsModel":
                state[k]["state"]["maxAzimuthAngle"] = "inf"
                state[k]["state"]["maxDistance"] = "inf"
                state[k]["state"]["maxZoom"] = "inf"
                state[k]["state"]["minAzimuthAngle"] = "-inf"

        tpl = embed.load_requirejs_template
        if not imports:
            embed.load_requirejs_template = ""

        s = embed.embed_snippet(self._renderer, state=state)
        # s = embed.embed_snippet(self.__w, state=state)
        embed.load_requirejs_template = tpl

        if html_frame:
            s = "<html>\n<body>\n" + s + "\n</body>\n</html>"

        # Revert changes
        for _, obj in self.__obj.items():
            if isinstance(obj, _MeshObject):
                obj.geometry.attributes["position"].array += mean
                if obj.bbox is not None:
                    bbox, _, _ = obj.bbox
                    bbox.geometry.positions += mean
            elif isinstance(obj, _PointObject):
                obj.geometry.attributes["position"].array += mean
            else:
                obj.geometry.positions += mean
        self.__update_view()

        return s

    def save(self, filename: Union[str, Path] = "") -> None:
        if filename == "":
            filename = str(uuid.uuid4()) + ".html"
        filename = Path(filename)
        filename.write_text(self.to_html())
        print(f"Plot saved to file {filename}.")

    # --------------
    # Internal Functions

    def __add_object(self, obj: Union[_MeshObject, _LineObject, _PointObject]) -> ObjectID:
        self._scene.add(obj.mesh)
        oid = self.__count
        self.__obj[oid] = obj
        self.__count += 1
        self.__update_view()
        return oid

    def __compute_line_object(self, lines: np.ndarray, line_shading: LineShading) -> _LineObject:
        lines = lines.astype("float32", copy=False)
        mi = np.min(lines, axis=0)
        ma = np.max(lines, axis=0)

        geometry = p3s.LineSegmentsGeometry(positions=lines.reshape((-1, 2, 3)))
        material = p3s.LineMaterial(linewidth=line_shading.line_width, color=line_shading.line_color)
        plines = p3s.LineSegments2(geometry=geometry, material=material)  # type: ignore
        return _LineObject(geometry, plines, material, ma, mi, line_shading)

    def __update_view(self):
        if len(self.__obj) == 0:
            return
        ma = np.zeros((len(self.__obj), 3))
        mi = np.zeros((len(self.__obj), 3))
        for r, (_, obj) in enumerate(self.__obj.items()):
            ma[r] = obj.max
            mi[r] = obj.min
        ma = np.max(ma, axis=0)
        mi = np.min(mi, axis=0)
        diag = np.linalg.norm(ma - mi)
        mean = ((ma - mi) / 2 + mi).tolist()
        scale = self.__s.scale * (diag)
        self._orbit.target = mean
        self._cam.lookAt(mean)
        self._cam.position = (mean[0], mean[1], mean[2] + scale)
        self._light.position = (mean[0], mean[1], mean[2] + scale)
        self._orbit.exec_three_obj_method("update")
        self._cam.exec_three_obj_method("updateProjectionMatrix")

    def _repr_mimebundle_(self, **kwargs):
        return self._renderer._repr_mimebundle_(**kwargs)

    # --------------
    # Helpers

    def __get_colors(
        self,
        v: np.ndarray,
        f: np.ndarray,
        c: Optional[np.ndarray],
        v_range: Optional[tuple[float, float]],
        colormap: str,
    ) -> tuple[np.ndarray, Literal["VertexColors", "FaceColors"]]:
        """
        v: (n_V, 3)
        f: (n_F, 3)
        c: (3,), (n_V, 3), (n_F, 3), (n_V,), or (n_F,)

        returns: ((n_V, 3), "VertexColors") or ((n_F * 3, 3), "FaceColors")
        """
        coloring = "VertexColors"
        if c is None:
            colors = np.ones_like(v)
            colors[:, 0] = 1.0
            colors[:, 1] = 0.874
            colors[:, 2] = 0.0
        else:
            if c.shape == (3,):  # Single color
                colors = np.ones_like(v)
                colors[:, 0] = c[0]
                colors[:, 1] = c[1]
                colors[:, 2] = c[2]
            elif len(c.shape) == 2 and c.shape[-1] == 3:  # Color values for
                if c.shape[0] == f.shape[0]:  # faces
                    colors = np.hstack([c, c, c]).reshape((-1, 3))
                    coloring = "FaceColors"
                elif c.shape[0] == v.shape[0]:  # vertices
                    colors = c
                else:  # Wrong size
                    raise ValueError(f"Invalid color shape: {c.shape}")
            elif c.shape == (f.shape[0],):  # Function values for faces
                cc = get_colors(c, colormap, v_range)
                colors = np.hstack([cc, cc, cc]).reshape((-1, 3))
                coloring = "FaceColors"
            elif c.shape == (v.shape[0],):  # Function values for vertices
                colors = get_colors(c, colormap, v_range)
            else:
                raise ValueError(f"Invalid color shape: {c.shape}")
        return colors, coloring

    def __get_point_colors(
        self, v: np.ndarray, c: Optional[np.ndarray], point_shading: PointShading
    ) -> tuple[Union[np.ndarray, str], bool]:
        v_color = True
        if type(c) == type(None):  # No color given, use global color
            # conv = mpl.colors.ColorConverter()
            colors = point_shading.point_color  # np.array(conv.to_rgb(sh["point_color"]))
            v_color = False
        elif type(c) == str:  # No color given, use global color
            # conv = mpl.colors.ColorConverter()
            colors = c  # np.array(conv.to_rgb(c))
            v_color = False
        elif (
            type(c) == np.ndarray and len(c.shape) == 2 and c.shape[1] == 3 and c.shape[0] == v.shape[0]
        ):  # Point color
            colors = c.astype("float32", copy=False)
        elif type(c) == np.ndarray and c.size == v.shape[0]:  # Function color
            colors = get_colors(c, point_shading.colormap, point_shading.v_range)
            colors = colors.astype("float32", copy=False)
        else:
            print("Invalid color array given! Supported are numpy arrays.", type(c))
            colors = point_shading.point_color
            v_color = False

        return colors, v_color

    def __get_bbox(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        v: (n_V, 3)
        returns: v_bbox: (8, 3), e_bbox: (12, 2)
        """
        m = np.min(v, axis=0)
        M = np.max(v, axis=0)

        # Corners of the bounding box
        v_box = np.array(
            [
                [m[0], m[1], m[2]],
                [M[0], m[1], m[2]],
                [M[0], M[1], m[2]],
                [m[0], M[1], m[2]],
                [m[0], m[1], M[2]],
                [M[0], m[1], M[2]],
                [M[0], M[1], M[2]],
                [m[0], M[1], M[2]],
            ]
        )

        f_box = np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [4, 5],
                [5, 6],
                [6, 7],
                [7, 4],
                [0, 4],
                [1, 5],
                [2, 6],
                [7, 3],
            ],
            dtype=np.uint32,
        )
        return v_box, f_box
