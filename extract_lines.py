#! /usr/bin/env python

"""
to run this script:
- install kraken
- activate kraken environment
- run: 

 python3 extract_lines.py -f xml -fi [png|jpg] -o path/to/output/dir path/to/xml/files/*.xml
"""

import click


# from https://github.com/alix-tz/kraken/blob/6b1b76458a07a080cfe9de184c4440444297dbc7/kraken/lib/segmentation.py
# to execute CK_extract_polygons()
from PIL import Image
from typing import List, Tuple, Union, Dict, Any, Sequence, Optional
import numpy as np
from skimage.measure import approximate_polygon, subdivide_polygon, regionprops, label
import shapely.geometry as geom
from skimage import draw, filters
from skimage.transform import PiecewiseAffineTransform, SimilarityTransform, AffineTransform, warp

from kraken.lib.segmentation import _rotate


# CK stands for Custom Kraken
def CK_extract_polygons(im: Image.Image, bounds: Dict[str, Any]) -> Image.Image:
    """
    Yields the subimages of image im defined in the list of bounding polygons
    with baselines preserving order.
    Args:
        im: Input image
        bounds: A list of dicts in baseline::
                    {'type': 'baselines',
                     'lines': [{'baseline': [[x_0, y_0], ... [x_n, y_n]],
                                'boundary': [[x_0, y_0], ... [x_n, y_n]]},
                               ....]
                    }
                or bounding box format::
                    {'boxes': [[x_0, y_0, x_1, y_1], ...], 'text_direction': 'horizontal-lr'}
    Yields:
        The extracted subimage
    """
    if 'type' in bounds and bounds['type'] == 'baselines':
        # select proper interpolation scheme depending on shape
        if im.mode == '1':
            order = 0
            im = im.convert('L')
        else:
            order = 1
        im = np.array(im)

        # custom: create im_bg with same size as im
        # custom: but filled with white color
        im_bg = np.zeros(im.shape, dtype=im.dtype)
        im_bg[:] = 255

        for line in bounds['lines']:
            if line['boundary'] is None:
                raise KrakenInputException('No boundary given for line')
            pl = np.array(line['boundary'])
            baseline = np.array(line['baseline'])
            c_min, c_max = int(pl[:, 0].min()), int(pl[:, 0].max())
            r_min, r_max = int(pl[:, 1].min()), int(pl[:, 1].max())

            if (pl < 0).any() or (pl.max(axis=0)[::-1] >= im.shape[:2]).any():
                raise KrakenInputException('Line polygon outside of image bounds')
            if (baseline < 0).any() or (baseline.max(axis=0)[::-1] >= im.shape[:2]).any():
                raise KrakenInputException('Baseline outside of image bounds')

            # fast path for straight baselines requiring only rotation
            if len(baseline) == 2:
                baseline = baseline.astype(float)
                # calculate direction vector
                lengths = np.linalg.norm(np.diff(baseline.T), axis=0)
                p_dir = np.mean(np.diff(baseline.T) * lengths/lengths.sum(), axis=1)
                p_dir = (p_dir.T / np.sqrt(np.sum(p_dir**2, axis=-1)))
                angle = np.arctan2(p_dir[1], p_dir[0])
                patch = im[r_min:r_max+1, c_min:c_max+1].copy()
                offset_polygon = pl - (c_min, r_min)
                r, c = draw.polygon(offset_polygon[:, 1], offset_polygon[:, 0])
                mask = np.zeros(patch.shape[:2], dtype=bool)
                # custom: to understand the syntax: https://scikit-image.org/docs/stable/api/skimage.draw.html#skimage.draw.polygon
                mask[r, c] = True
                patch[mask != True] = 255 # custom: changed the value to 255 (original value is 0)
                extrema = offset_polygon[(0, -1), :]
                # scale line image to max 600 pixel width
                tform, rotated_patch = _rotate(patch, angle, center=extrema[0], scale=1.0, cval=0)
                i = Image.fromarray(rotated_patch.astype('uint8'))
            # normal slow path with piecewise affine transformation
            else:
                if len(pl) > 50:
                    pl = approximate_polygon(pl, 2)
                full_polygon = subdivide_polygon(pl, preserve_ends=True)
                pl = geom.MultiPoint(full_polygon)

                bl = zip(baseline[:-1:], baseline[1::])
                bl = [geom.LineString(x) for x in bl]
                cum_lens = np.cumsum([0] + [line.length for line in bl])
                # distance of intercept from start point and number of line segment
                control_pts = []
                for point in pl.geoms:
                    npoint = np.array(point.coords)[0]
                    line_idx, dist, intercept = min(((idx, line.project(point),
                                                      np.array(line.interpolate(line.project(point)).coords)) for idx, line in enumerate(bl)),
                                                    key=lambda x: np.linalg.norm(npoint-x[2]))
                    # absolute distance from start of line
                    line_dist = cum_lens[line_idx] + dist
                    intercept = np.array(intercept)
                    # side of line the point is at
                    side = np.linalg.det(np.array([[baseline[line_idx+1][0]-baseline[line_idx][0],
                                                    npoint[0]-baseline[line_idx][0]],
                                                   [baseline[line_idx+1][1]-baseline[line_idx][1],
                                                    npoint[1]-baseline[line_idx][1]]]))
                    side = np.sign(side)
                    # signed perpendicular distance from the rectified distance
                    per_dist = side * np.linalg.norm(npoint-intercept)
                    control_pts.append((line_dist, per_dist))
                # calculate baseline destination points
                bl_dst_pts = baseline[0] + np.dstack((cum_lens, np.zeros_like(cum_lens)))[0]
                # calculate bounding polygon destination points
                pol_dst_pts = np.array([baseline[0] + (line_dist, per_dist) for line_dist, per_dist in control_pts])
                # extract bounding box patch
                c_dst_min, c_dst_max = int(pol_dst_pts[:, 0].min()), int(pol_dst_pts[:, 0].max())
                r_dst_min, r_dst_max = int(pol_dst_pts[:, 1].min()), int(pol_dst_pts[:, 1].max())
                output_shape = np.around((r_dst_max - r_dst_min + 1, c_dst_max - c_dst_min + 1))
                patch = im[r_min:r_max+1, c_min:c_max+1].copy()
                # offset src points by patch shape
                offset_polygon = full_polygon - (c_min, r_min)
                offset_baseline = baseline - (c_min, r_min)
                # offset dst point by dst polygon shape
                offset_bl_dst_pts = bl_dst_pts - (c_dst_min, r_dst_min)
                offset_pol_dst_pts = pol_dst_pts - (c_dst_min, r_dst_min)
                # mask out points outside bounding polygon
                mask = np.zeros(patch.shape[:2], dtype=bool)
                r, c = draw.polygon(offset_polygon[:, 1], offset_polygon[:, 0])
                mask[r, c] = True
                patch[mask != True] = 255 # custom: changed the value to 255 (original value is 0)
                # estimate piecewise transform
                src_points = np.concatenate((offset_baseline, offset_polygon))
                dst_points = np.concatenate((offset_bl_dst_pts, offset_pol_dst_pts))
                tform = PiecewiseAffineTransform()
                tform.estimate(src_points, dst_points)
                o = warp(patch, tform.inverse, output_shape=output_shape, preserve_range=True, order=order)
                i = Image.fromarray(o.astype('uint8'))
            yield i.crop(i.getbbox()), line
    else:
        if bounds['text_direction'].startswith('vertical'):
            angle = 90
        else:
            angle = 0
        for box in bounds['boxes']:
            if isinstance(box, tuple):
                box = list(box)
            if (box < [0, 0, 0, 0] or box[::2] >= [im.size[0], im.size[0]] or
                    box[1::2] >= [im.size[1], im.size[1]]):
                logger.error('bbox {} is outside of image bounds {}'.format(box, im.size))
                raise KrakenInputException('Line outside of image bounds')
            yield im.crop(box).rotate(angle, expand=True), box




@click.command()
@click.option('-f', '--format-type', type=click.Choice(['xml', 'alto', 'page', 'binary']), default='xml',
              help='Sets the input document format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both baselines, polygons, and a '
              'link to source images.')
@click.option('-i', '--model', default=None, show_default=True, type=click.Path(exists=True),
              help='Baseline detection model to use. Overrides format type and expects image files as input.')
@click.option('--repolygonize/--no-repolygonize', show_default=True,
              default=False, help='Repolygonizes line data in ALTO/PageXML '
              'files. This ensures that the trained model is compatible with the '
              'segmenter in kraken even if the original image files either do '
              'not contain anything but transcriptions and baseline information '
              'or the polygon data was created using a different method. Will '
              'be ignored in `path` mode. Note, that this option will be slow '
              'and will not scale input images to the same size as the segmenter '
              'does.')
@click.option('-fi', '--format-image', type=click.Choice(['jpg', 'png']), default='png',
              help='Sets the output image format.')
@click.option('-o', '--output', type=click.Path(exists=False), default='./out',
                help='Sets the output directory. Defaults to ./out')
@click.argument('files', nargs=-1)
def cli(format_type, model, repolygonize, files, format_image, output):
    """
    A small script extracting rectified line polygons as defined in either ALTO or
    PageXML files or run a model to do the same.
    """
    if len(files) == 0:
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    from PIL import Image
    import os
    from os.path import splitext
    from kraken import blla
    from kraken.lib import segmentation, vgsl, xml
    import io
    import json
    import pyarrow as pa

    import numpy as np

    def _make_output_file_path(filepath, output):
        """
        Creates an output file path from an input file path and the output directory.
        """
        basename = os.path.basename(filepath)
        return os.path.join(output, basename)

    
    def _add_white_background(im):
        # heavy option: change all pixels [0,0,0] to [255,255,255]
        im = im.convert('RGB')
        data = np.array(im)
        red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
        mask = (red == 0) & (green == 0) & (blue == 0)
        data[:,:,:3][mask] = [255, 255, 255]
        im = Image.fromarray(data)
        return im

    output = os.path.abspath(output)
    if not os.path.exists(output):
            os.makedirs(output)

    if model is None:
        for doc in files: # [:1] added while debugging
            click.echo(f'Processing {doc} ', nl=False)
            if format_type != 'binary':
                data = xml.preparse_xml_data([doc], format_type, repolygonize=repolygonize)
                if len(data) > 0:
                    bounds = {'type': 'baselines', 'lines': [{'boundary': t['boundary'], 'baseline': t['baseline'], 'text': t['text']} for t in data]}
                    for idx, (im, box) in enumerate(CK_extract_polygons(Image.open(data[0]['image']), bounds)):
                        click.echo('.', nl=False)
                        # debug
                        #if idx == 5:
                        #    break
                        # end debug
                        # custom: adding white where we have pure black, which is norammlly the result of .crop() in CK_extract_polygons().
                        im = _add_white_background(im) #WARNING: this might erase some text... 
                        im.save(_make_output_file_path('{}.{}.{}'.format(splitext(data[0]['image'])[0], idx, format_image), output))
                        with open(_make_output_file_path('{}.{}.gt.txt'.format(splitext(data[0]['image'])[0], idx), output), 'w', encoding="utf8") as fp:
                            fp.write(box['text'])
            else:
                with pa.memory_map(doc, 'rb') as source:
                    ds_table = pa.ipc.open_file(source).read_all()
                    raw_metadata = ds_table.schema.metadata
                    if not raw_metadata or b'lines' not in raw_metadata:
                        raise ValueError(f'{doc} does not contain a valid metadata record.')
                    metadata = json.loads(raw_metadata[b'lines'])
                    for idx in range(metadata['counts']['all']):
                        sample = ds_table.column('lines')[idx].as_py()
                        im = Image.open(io.BytesIO(sample['im']))
                        im.save(_make_output_file_path('{}.{}.{}'.format(splitext(doc)[0], idx, format_image), output))
                        with open(_make_output_file_path('{}.{}.gt.txt'.format(splitext(doc)[0], idx), output), 'w', encoding="utf8") as fp:
                            fp.write(sample['text'])
            print("\n")
    else:
        net = vgsl.TorchVGSLModel.load_model(model)
        for doc in files:
            click.echo(f'Processing {doc} ', nl=False)
            full_im = Image.open(doc)
            bounds = blla.segment(full_im, model=net)
            for idx, (im, box) in enumerate(CK_extract_polygons(full_im, bounds)):
                click.echo('.', nl=False)
                im.save(_make_output_file_path('{}.{}.{}'.format(splitext(doc)[0], idx, format_image), output))
            print("\n")


if __name__ == '__main__':
    cli()
