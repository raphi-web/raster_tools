import numpy as np
from shapely import geometry
import raster_tools as rt
import copy


class Tiles:
    def __init__(self, raster, bounds):
        self.bounds = bounds
        self.shape = raster.shape
        self.rows, self.cols = raster.shape
        self.left, self.bottom, self.right, self.top = bounds
        self.resolution = (self.right - self.left) / self.rows
        self.rectangle = Tiles._mk_rectangle(bounds)

        self.tiles, self.boundaries, self.origins = Tiles._raster_quaters(
            raster, bounds)

        self.tile_rectangles = []
        for bound in self.boundaries:
            self.tile_rectangles.append(Tiles._mk_rectangle(bound))

    def intersects(self, geom):
        if self.rectangle.intersects(geom):
            return True
        else:
            return False

    def recompose(self):
        rast = np.empty((self.rows, self.cols))

        for tile, origin in zip(self.tiles, self.origins):
            if not self.contains_array():
                tile = tile.recompose()

            t_rows, t_cols = tile.shape
            row_origin, col_origin = origin

            rast[row_origin: row_origin + t_rows,
                 col_origin:col_origin + t_cols] = tile

        return rast

    def tileception(self):
        tile_copy = copy.deepcopy(self)

        sub_tiles = []
        if not self.contains_array():
            for tile in tile_copy.tiles:
                sub_tiles.append(tile.tileception())

        else:
            for tile, geom in zip(tile_copy.tiles, tile_copy.boundaries):
                left, bottom, right, top = geom
                tile_of_tile = Tiles(tile, (left, bottom, right, top))
                sub_tiles.append(tile_of_tile)

        tile_copy.tiles = sub_tiles
        return tile_copy

    def vector_intersects(self, geom, value=1):

        if not self.contains_array():
            new_tiles = []
            for t in self.tiles:

                for g in geom:
                    does_intersect = False
                    if t.intersects(g):
                        does_intersect=True
                        break

                if does_intersect:
                    #clipped_geometries = [t.rectangle.intersection(g) for g in geom]        
                    new_tiles.append(t.vector_intersects(geom, value=value))
                else:
                    new_tiles.append(t)

        else:
            new_tiles = []
            for t, b, r in zip(self.tiles, self.boundaries, self.tile_rectangles):
                for g in geom:
                    if r.intersects(g):
                        for i, j in np.ndindex(t.shape):
                            if Tiles._pixel_as_polygon((i, j), b, t.shape).intersects(g):
                                t[i, j] = value

                new_tiles.append(t)

            self.tiles = new_tiles

    def contains_array(self):
        if isinstance(self.tiles[0], Tiles):
            return False
        else:
            return True

    @staticmethod
    def _raster_quaters(rast, bounds):
        rows, cols = rast.shape
        row_half, col_half = rows//2, cols//2

        left, bottom, right, top = bounds
        width = right - left
        resolution = width / cols

        h1 = rast[:row_half]
        h2 = rast[row_half:]

        h1_bottom, h1_top = top - resolution * h1.shape[0], top
        h2_bottom, h2_top = bottom, bottom + resolution * h2.shape[0]

        h1_origin = (0, 0)
        h2_origin = (h1.shape[0], 0)  # +1 ?

        quaters = []
        for r in [h1, h2]:
            q1 = r[:, :col_half]
            q2 = r[:, col_half:]

            quaters.append(q1)
            quaters.append(q2)

        q1_left, q1_right = left, left + resolution * quaters[0].shape[1]
        q1_bottom, q1_top = h1_bottom, h1_top

        q2_left, q2_right = right - resolution * quaters[1].shape[1], right
        q2_bottom, q2_top = h1_bottom, h1_top

        q3_left, q3_right = q1_left, q1_right
        q3_bottom, q3_top = h2_bottom, h2_top

        q4_left, q4_right = q2_left, q2_right
        q4_bottom, q4_top = h2_bottom, h2_top

        bounds = [
            (q1_left, q1_bottom, q1_right, q1_top),
            (q2_left, q2_bottom, q2_right, q2_top),
            (q3_left, q3_bottom, q3_right, q3_top),
            (q4_left, q4_bottom, q4_right, q4_top),
        ]

        q1_origin = h1_origin
        q2_origin = (0, q1.shape[1])

        q3_origin = (h2_origin[0], 0)
        q4_origin = (h2_origin[0], q1.shape[1])

        origins = [q1_origin, q2_origin, q3_origin, q4_origin]

        return quaters, bounds, origins

    @staticmethod
    def _mk_rectangle(bounds):
        left, bottom, right, top = bounds
        rectangle = geometry.Polygon([
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom)])

        return rectangle

    @staticmethod
    def _pixel_as_polygon(indexes, bounds, shape):

        row, col = indexes
        _, ncols = shape
        left, _, right, _ = bounds

        res = (right - left) / ncols

        x, y = rt.get_pxl_coors(row, col, bounds, shape)

        pxl_bounds = (
            x - res,  # left
            y - res,  # bottom
            x + res,  # right
            y + res)  # top

        pixel = Tiles._mk_rectangle(pxl_bounds)

        return pixel


if __name__ == '__main__':
    # Tests

    rast = np.arange(0, 6400).reshape((80, 80))
    bounds = (0, 0, 100, 100)

    rast_tiles = Tiles(rast, bounds)
    print("Type of first Raster:", type(rast_tiles.tiles[0]))
    print("Reconstruction first Raster:", rast_tiles.recompose().shape)

    rast_tiles = rast_tiles.tileception()
    print("Tileception 1:", type(rast_tiles.tiles[0]))
    print("Reconstruction Tileception 1", rast_tiles.recompose().shape)

    rast_tiles = rast_tiles.tileception()
    print("Tileception 2.0:", type(rast_tiles.tiles[0]))
    print("Tileception 2.1:", type(rast_tiles.tiles[0].tiles[0]))
    print("Tileception 2.2:", type(rast_tiles.tiles[0].tiles[0].tiles[0]))
    print("Reconstruction Tileception 2", rast_tiles.recompose().shape)

    """
    print(rast_tiles.recompose().shape)
    tile_ceptions = tileception(rast_tiles)
    print(tile_ceptions.recompose().shape)
    tile_ceptions = tileception(tile_ceptions)
    print(tile_ceptions.recompose().shape)
    """
