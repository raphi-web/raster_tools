import numpy as np
import geopandas as gpd
import rasterio
import os
from numba import njit


@njit
def get_pxl_coors(row, col, bounds, rast_shape):
    """
    Calculates geographic coordinates based on
    the bounds of a raster
    """
    left, bottom, right, top = bounds
    nrows, ncols = rast_shape

    resolution_x = (right - left) / ncols
    resolution_y = (top - bottom) / nrows

    X = left + (resolution_x / 2) + (col * resolution_x)
    Y = top - (resolution_y / 2) - (row * resolution_y)

    return X, Y


def get_pxl_coors_src(row, col, src):
    """
    Calculates geographic coordinates based on
    the bounds of a raster,

    to be used with rasterio src object
    """
    left, bottom, right, top = src.bounds
    nrows, ncols = src.height, src.width

    resolution_x = (right - left) / ncols
    resolution_y = (top - bottom) / nrows

    X = left + (resolution_x / 2) + (col * resolution_x)
    Y = top - (resolution_y / 2) - (row * resolution_y)

    return X, Y


@njit
def get_neighbours2D(row, col, ksize, arr):
    if ksize % 2 == 0:
        raise ValueError("ksize must be uneven")

    ksize = (ksize - 1) // 2
    arr_rows, arr_cols = np.shape(arr)

    row_max = min(row + ksize + 1, arr_rows)
    row_min = max(row - ksize, 0)
    col_max = min(col + ksize + 1, arr_cols)
    col_min = max(col - ksize, 0)

    return arr[row_min:row_max, col_min:col_max]


def get_neighbours3D(row, col, ksize, arr):
    if ksize % 2 == 0:
        raise ValueError("ksize must be uneven")

    ksize = (ksize - 1) // 2
    arr_rows, arr_cols = np.shape(arr)

    row_max = min(row + ksize + 1, arr_rows)
    row_min = max(row - ksize, 0)
    col_max = min(col + ksize + 1, arr_cols)
    col_min = max(col - ksize, 0)

    if len(arr.shape) == 2:
        return arr[row_min:row_max, col_min:col_max]

    if len(arr.shape) == 3:
        return arr[:, row_min:row_max, col_min:col_max]


@njit
def std_filter(rast, ksize,target_rast=None):
    return_target = False
    if target_rast == None:
        target_rast = np.empty(rast.shape, dtype=np.float64)
        return_target = True

    for i, j in np.ndindex(rast.shape):
        target_rast[i, j] = get_neighbours2D(i, j, ksize, rast).std()

    if return_target:
        return target_rast


@njit
def argmax(rast1,rast2,target_rast = None):
    return_target = False
    if target_rast == None:
        target_rast = np.empty(rast1.shape,dtype=np.float64)
        return_target = True
    
    for i,j in np.ndindex(rast1.shape):
        value = max(rast1[i,j], rast2[i,j])
        if np.isnan(value):
            target_rast[i,j] = rast1[i,j]
        else:
            target_rast[i,j] = value

    if return_target:
        return target_rast

@njit
def med_filter(rast, ksize, target_rast=None):
    return_target = False
    if target_rast == None:
        target_rast = np.empty(rast.shape, dtype=np.float64)
        return_target = True

    for i, j in np.ndindex(rast.shape):
        target_rast[i, j] = np.median(get_neighbours2D(i, j, ksize, rast))
    if return_target:
        return target_rast


@njit
def avg_filter(rast, ksize, target_rast=None):
    return_target = False
    if target_rast == None:
        target_rast = np.empty(rast.shape, dtype=np.float64)
        return_target = True

    for i, j in np.ndindex(rast.shape):
        target_rast[i, j] = get_neighbours2D(i, j, ksize, rast).ravel().mean()
    
    if return_target:
        return target_rast


@njit
def img2grayscale(rast, weights, target_rast= None):
    return_target = False
    if target_rast == None:
        target_rast = np.empty(rast.shape[1:],dtype=np.float64)
        return_target = True
    rw, gw, bw = weights
    for i, j in np.ndindex(rast.shape[1:]):
        pxl = rast[:,i,j]
        r,g,b = pxl
        target_rast[i,j] =  r * rw + g * gw + b * bw

    if return_target:
        return target_rast

@njit
def normalize(img, max_v=1):
    mins = np.nanmin(img)
    maxs = np.nanmax(img)
    norm = (img - mins)/(maxs - mins)
    new_rast = norm * max_v

    return new_rast


@njit
def normalzieBands(raster,target_rast=None ,max_v=1):
    return_target = False
    if target_rast == None:
        target_rast = np.zeros(raster.shape, dtype=np.float64)
        return_target = True
    for idx, band in enumerate(raster):
        target_rast[idx] = normalize(band, max_v)

    if return_target:
        return target_rast


def arr2idxs(arr):
    idx_list = [(r, c) for r, c in np.ndindex(arr.shape)]
    narr = np.empty(arr.shape[0] * arr.shape[1]).astype("object")
    narr[:] = idx_list
    narr = narr.reshape(arr.shape)
    return narr


def rast_intersects(x, y, arr, bounds):
    left, bottom, right, top = bounds
    if len(arr.shape) == 2:
        nrows, ncols = arr.shape
    if len(arr.shape) == 3:
        nrows, ncols = arr.shape[1:]

    width = right - left
    height = top - bottom

    x_shift = x - left

    col = int(np.floor(x_shift))
    row = int(np.floor(height - y))

    if col < 0 or col > ncols:
        return False, ()

    if row < 0 or row > nrows:
        return False, ()

    return True, (col, row)


def index_neighbours(arr):
    indx_arr = arr2idxs(arr)
    neighbour_list = []

    def tuple_delete(arr, i, j):
        for idx, (ii, jj) in enumerate(arr):
            if ii == i and jj == j:
                arr = np.delete(arr, idx)
        return arr

    for i, j in np.ndindex(indx_arr.shape):
        neighbours = get_neighbours3D(i, j, 3, indx_arr)
        neighbours = tuple_delete(neighbours, i, j)

        neighbour_list.append(
            neighbours
        )

    neighbour_arr = np.array(neighbour_list, dtype=object)

    return neighbour_arr.reshape(arr.shape)





def raster2pnts(src):
    rast = src.read()
    left, bottom, right, top = src.bounds
    _, rows, cols = rast.shape
    resolution = (top-bottom)/rows
    xi = np.linspace(left + resolution/2, right - resolution/2, cols)
    yi = np.linspace(bottom + resolution/2, top - resolution/2, rows)
    xx, yy = np.meshgrid(xi, yi)

    coors = [xx.ravel().tolist(), yy.ravel().tolist()]
    for band in rast:
        band = band.ravel()
        coors.append(band)

    data = np.column_stack(coors)

    return data


def raster2df(src):

    rast = src.read()
    left, bottom, right, top = src.bounds
    nbands, nrows, ncols = rast.shape
    resolution = (top-bottom)/nrows
    xi = np.linspace(left + resolution/2, right - resolution/2, ncols)
    yi = np.linspace(bottom + resolution/2, top - resolution/2, nrows)
    xx, yy = np.meshgrid(xi, yi)
    yy = np.flip(yy, axis=0)
    coors = [xx.ravel(), yy.ravel()]

    for band in rast:
        band = band.ravel()
        coors.append(band)

    raster_data = np.column_stack(coors)

    raster_df = gpd.GeoDataFrame(data=raster_data[:, 2:],
                                 geometry=gpd.points_from_xy(
                                     raster_data[:, 0], raster_data[:, 1]),
                                 columns=["Band-" + str(i) for i in range(1, nbands+1)])

    return raster_df


def sampleRaster(src, poly_file, rast=None):

    if type(rast) == type(None):
        rast = src.read()

    left, bottom, right, top = src.bounds
    nbands, nrows, ncols = rast.shape
    resolution = (top-bottom)/nrows
    xi = np.linspace(left + resolution/2, right - resolution/2, ncols)
    yi = np.linspace(bottom + resolution/2, top - resolution/2, nrows)
    xx, yy = np.meshgrid(xi, yi)
    yy = np.flip(yy, axis=0)
    coors = [xx.ravel(), yy.ravel()]

    for band in rast:
        band = band.ravel()
        coors.append(band)

    raster_data = np.column_stack(coors)

    raster_df = gpd.GeoDataFrame(
        data=raster_data[:, 2:],
        geometry=gpd.points_from_xy(raster_data[:, 0], raster_data[:, 1]),
        columns=["Band-" + str(i) for i in range(1, nbands+1)])

    raster_df = raster_df.set_crs(str(src.crs))
    poly_df = gpd.read_file(poly_file)

    rasters_with_poly = gpd.sjoin(raster_df, poly_df,
                                  how="inner",
                                  op="intersects")

    return rasters_with_poly.sample(frac=1)


def classify_img(clf, rast, nodata=0):
    bands, rows, cols = rast.shape
    X_rast = rast.reshape(bands, rows*cols).T
    y_prediction = clf.predict(X_rast)
    y_result_rast = y_prediction.T.reshape(rows, cols)
    np.putmask(y_result_rast, rast[0] == nodata, nodata)

    return y_result_rast


def pad(rast, size):
    result = np.empty((rast.shape[0],) + size)
    for idx, band in enumerate(rast):
        narr = np.zeros(size)
        narr[:band.shape[0], :band.shape[1]] = band
        result[idx] = narr

    return result


def readTiff(fname):
    src = rasterio.open(fname)
    rast = src.read()
    return src, rast


def vrt_pad(src, tile_height, tile_width):

    rast = src.read()
    _, rows, cols = rast.shape
    left, bottom, right, top = src.bounds

    resolution = (top - bottom) / rows

    if rows % tile_height == 0:
        new_rows = rows

    else:
        new_rows = (rows - rows % tile_height) + tile_height

    if cols % tile_width == 0:
        new_cols = cols

    else:
        new_cols = (cols - cols % tile_width) + tile_width

    new_bottom = bottom - (new_rows - rows) * resolution
    new_right = right + (new_cols - cols) * resolution

    profile = src.profile
    profile.update(
        bounds=(left, new_bottom, new_right, top),
        width=new_cols,
        height=new_rows
    )

    return pad(rast, (new_rows, new_cols)), profile


def tile_iter(rast, func, tile_shape, double_tiles=False):

    def iterator(rast):
        bands, rast_rows, rast_cols = rast.shape
        tile_rows, tile_cols = tile_shape

        num_tile_rows = rast_rows // tile_rows
        num_tile_cols = rast_cols // tile_cols

        result_rast = np.zeros((rast_rows, rast_cols))

        for r in range(num_tile_rows):
            r_start = r * tile_rows
            r_stop = r * tile_rows + tile_rows

            for c in range(num_tile_cols):
                c_start = c * tile_cols
                c_stop = c * tile_cols + tile_cols

                tile = rast[:, r_start:r_stop, c_start: c_stop]
                result = func(tile)
                result_rast[r_start:r_stop, c_start: c_stop] = result

        return result_rast

    if double_tiles:
        first_result_rast = iterator(rast)
        rows, cols = first_result_rast.shape

        drow = tile_shape[0] // 2
        dcol = tile_shape[1] // 2
        crop_rast = rast[:, drow:rows-drow, dcol:cols-dcol]

        second_result_rast = np.zeros(first_result_rast.shape)
        second_result_rast[drow:rows-drow,
                           dcol:cols-dcol] = iterator(crop_rast)

        return first_result_rast, second_result_rast

    else:
        return iterator(rast)


def padTileIterator(rast, func, tile_shape, output_channels=1, mean=False):
    """
    iterates over a raster in tiles of given shape, 
    developed for predicting with klassifiers and semantic segmentation
    this function first padds the raster with zeros than iterates over padded raster
    so that only the inner most pixels of the tile are saved in the output raster. 
    This is done to always use the center pixel results of a classifier prediction of a tile.
    Because detection of Objects along the edges of a tile proves difficult!

    """

    bands, rast_rows, rast_cols = rast.shape
    tile_rows, tile_cols = tile_shape

    # padding equals half of tile size
    pad_offset_row = tile_rows // 2
    pad_offset_col = tile_cols // 2

    # create empty raster with additional padding
    padded_raster = np.zeros(
        (bands, rast_rows + tile_shape[0], rast_cols + tile_shape[1]))

    # replace zeros with mean for experimental prediction results,
    # maby it works better with some classifyers
    if mean:
        for idx, band in enumerate(rast):
            padded_raster[idx, padded_raster[idx, :, :, ] == 0] = band.mean()

    # fill the padded raster with the input raster so that the
    # input raster is aliged in the center of the padded raster

    padded_raster[:, pad_offset_row: -pad_offset_row,
                  pad_offset_col: - pad_offset_col] = rast[:]

    # create empty array to store results
    result_rast = np.zeros((output_channels, rast_rows, rast_cols))

    # calculate the number of tiles needed tile shape of (128,128)
    # in padded raster will be cut down to (64,64) in result raster

    num_tile_rows = rast_rows // pad_offset_row
    num_tile_cols = rast_cols // pad_offset_col

    # iterate over tiles
    for r in range(num_tile_rows):
        row_idx = r * pad_offset_row  # start-row index of input raster
        # start-row index of result raster
        pad_row_idx = r * pad_offset_row + pad_offset_row // 2

        for c in range(num_tile_cols):
            col_idx = c * pad_offset_col
            pad_col_idx = c * pad_offset_col + pad_offset_col // 2

            # get tile from padded_raster, perform input function on it
            result = func(padded_raster[:, pad_row_idx:  pad_row_idx +
                          tile_rows, pad_col_idx: pad_col_idx + tile_rows])

            # keep only the half in the center and add it to result array
            # expl: result-shape (128,128) ---> (32:69,32:69) == (64,64)
            result_rast[:, row_idx:row_idx + pad_offset_row,
                        col_idx:col_idx + pad_offset_col] = result[:, pad_offset_row // 2: - pad_offset_row//2,
                                                                   pad_offset_col // 2: -pad_offset_col // 2]

    return result_rast


def intervalClassify(rast, clist):
    clist = [rast.min()] + clist + [rast.max() + 1]
    nrast = np.zeros(rast.shape)

    for i in range(len(clist) - 1):
        nrast[(rast >= clist[i]) & (rast < clist[i+1])] = clist[i]

    return nrast


def intervalClassifyRaster(fname, src_name, class_list, bandnumb=1, dtype=np.int64):
    src = rasterio.open(src_name)
    rast = src.read(bandnumb)

    result = intervalClassify(rast, class_list)

    profile = src.profile
    profile.update(dtype=dtype)
    with rasterio.open(fname, "w+", **profile) as dst:
        dst.write(result.astype(dtype), 1)



