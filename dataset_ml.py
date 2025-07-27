import hashlib
from pathlib import Path

import numpy as np
import skimage
import tifffile
import torch
import pickle
import matplotlib.pyplot as plt
from normalize import fit_normalizer, transform_image
import random

class TrainImage():
    def __init__(self, info_dict):
        # make a copy of the info_dict to avoid modifying the original
        self.info_dict = {**info_dict}

        self.out_channels = [c['name'] for c in self.info_dict['metadata']['channels'] if not c['name'].startswith('p_')]
        if self.info_dict["dataset"]["one_channel"]:
            self.out_channels = [self.out_channels[0]]
        self.labels = self.info_dict["dataset"]["labels"]
    
        self.in_channels = self.info_dict["dataset"]["channels"]

        self.regions = []
        
        if self.info_dict["dataset"]["use_regions"]:
            region_coords = [f['geometry']['coordinates'][0] for f in self.info_dict['features']if f['properties']['classification']['name'] == 'Region*']
        else:
            region_coords = [[[0, 0],[0, info_dict["metadata"]["width"]],[info_dict["metadata"]["width"], info_dict["metadata"]["height"]], [info_dict["metadata"]["height"],0], [0, 0]]]
        
        
        self.regions.extend(
            {
                "coord": region,
                "mask": None,
                "x": None,
                "y": None,
                "w": None,
                "h": None,
            }
            for region in region_coords
        )

        if "width" in info_dict["metadata"]:
            self.image_width = info_dict["metadata"]["width"]
            self.image_height = info_dict["metadata"]["height"]
        else:
            # use region size
            self.image_width = max(max(p[0] for p in region['coord']) for region in self.regions)
            self.image_height = max(max(p[1] for p in region['coord']) for region in self.regions)

        whole_image = self.load_whole_image()
        
        # get normalization method from config, default to q5_q95
        norm_method = self.info_dict["dataset"].get("normalization_method", "q5_q95")
        
        # fit normalizer and store settings
        norm_settings = fit_normalizer(whole_image, norm_method)
        self.info_dict["metadata"]["normalization_method"] = norm_method
        self.info_dict["metadata"]["normalization_settings"] = norm_settings
    


        self.polygons = {c: [f['geometry']['coordinates'] for f in self.info_dict['features'] if f['properties']['classification']['name'] == c] for c in self.labels}
        self.load_mask()

    def generate_hash(self, region):
        # use the hash of image path, region x, y, w, h to generate a filename
        # this way we can cache the region
        
        # get the hash of the image path
        filename_hash = hashlib.sha256(self.info_dict["path"].encode('utf-8')).hexdigest()
        
        # get the hash of the region
        filename_hash += hashlib.sha256(str(region['x']).encode('utf-8')).hexdigest()
        filename_hash += hashlib.sha256(str(region['y']).encode('utf-8')).hexdigest()
        filename_hash += hashlib.sha256(str(region['w']).encode('utf-8')).hexdigest()
        filename_hash += hashlib.sha256(str(region['h']).encode('utf-8')).hexdigest()

        # also include downscale
        filename_hash += hashlib.sha256(str(self.info_dict["dataset"]["downscale"]).encode('utf-8')).hexdigest()
        
        # make hash shorter to use it as filename using sha1
        filename_hash = hashlib.sha1(filename_hash.encode('utf-8')).hexdigest()
        
        return filename_hash

    def cache_region(self, region):
        filename_hash = self.generate_hash(region)

        # save region as pickle to ./cache/{hash}.pkl, overwrite anyway
        cache_path = Path("cache") / f"{filename_hash}.pkl"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # save the region
        with open(cache_path, 'wb') as f:
            pickle.dump(region, f)

    def load_region_cache(self, info_dict, region):
        return None
        filename_hash = self.generate_hash(region)

        # save region as pickle to ./cache/{hash}.pkl, overwrite anyway
        cache_path = Path("cache") / f"{filename_hash}.pkl"

        # check if cache exists
        if not cache_path.exists():
            return None
        
        # save the region
        with open(cache_path, 'rb') as f:
            region = pickle.load(f)

        return region
    
    def load_whole_image(self):
        # load the whole image from tif file
        with tifffile.TiffFile(self.info_dict["path"]) as tif:
            if self.info_dict["dataset"]["different_pages"]:
                # if labels correspond to each image channel
                image = tif.asarray().transpose(2, 0, 1)
            else:
                # if just one image channel or RGB
                image = tif.asarray().transpose(2, 0, 1)

        # downscale if needed
        if self.info_dict["dataset"]["downscale"] > 1:
            image = skimage.transform.rescale(
                image,
                (1, 1 / self.info_dict["dataset"]["downscale"], 1 / self.info_dict["dataset"]["downscale"]),
                preserve_range=True,
                anti_aliasing=True
            )

        return image

    def load_region(self, region):
        # try cached region first
        if l_cached := self.load_region_cache(self.info_dict, region) is not None:
            return l_cached

        with tifffile.TiffFile(self.info_dict["path"]) as tif:
            # if labels correspond to each image channel
            if self.info_dict["dataset"]["different_pages"]:
                for i, c in enumerate(self.labels):
                    region["image"][i, :, :] = tif.asarray()[i,:,:][region['y']:region['y']+region['h'], region['x']:region['x']+region['w']]
            # if just one image channel or RGB
            else:
                
                region["image"] = tif.asarray()[region['y']:region['y'] + region['h'], region['x']:region['x'] + region['w']]
                
                region["image"] = region["image"].transpose(2, 0, 1)
            # check if channels are in the correct order
            if region["image"].shape[0] != len(self.in_channels):
                #region["image"] = region["image"].transpose(2, 0, 1)
                pass
                

        # downscale if needed
        if self.info_dict["dataset"]["downscale"] > 1:
            # use bilinear resize for image, and nearest neighbor for mask
            region["image"] = skimage.transform.rescale(
                region["image"],
                (1, 1 / self.info_dict["dataset"]["downscale"], 1 / self.info_dict["dataset"]["downscale"]),
                preserve_range=True,
                anti_aliasing=True
            )

            region["mask"] = skimage.transform.rescale(
                region["mask"],
                (1, 1 / self.info_dict["dataset"]["downscale"], 1 / self.info_dict["dataset"]["downscale"]),
                order=0
            ).astype(np.bool_)

            region["h"] = int(region["h"] / self.info_dict["dataset"]["downscale"])
            region["w"] = int(region["w"] / self.info_dict["dataset"]["downscale"])
            region['x'] = int(region['x'] / self.info_dict["dataset"]["downscale"])
            region['y'] = int(region['y'] / self.info_dict["dataset"]["downscale"])

        # cache the region
        self.cache_region(region)

        return region

    def load_mask(self):
        # for each region, create a mask        
        full_mask = np.zeros((len(self.labels), self.image_height, self.image_width), dtype=np.bool_)
        
        # print(1, full_mask.shape, self.image_height, self.image_width)
        for i, c in enumerate(self.labels):
            for poly in self.polygons[c]:
                if len(poly) == 1:
                    full_mask[i, :, :] += skimage.draw.polygon2mask(full_mask[i, :, :].shape, [(p[1], p[0]) for p in poly[0]])
                if len(poly) > 1:
                    for polym in poly:
                        full_mask[i, :, :] ^= skimage.draw.polygon2mask(full_mask[i, :, :].shape, [(p[1], p[0]) for p in polym[0]])
                    # if there are multiple polygons, we need to combine them
                    # this is the case for example with Multipolygons
                    # we can use skimage.draw.polygon2mask to create a mask for each polygon
                    # and then combine them using XOR

                else:
                    try:
                        
                        full_mask[i, :, :] += skimage.draw.polygon2mask(full_mask[i, :, :].shape, [(p[1], p[0]) for p in poly[0]])
                    except ValueError as e:
                        # no Polygon here, for example does not work with Multipolygons
                        # maybe implement method here to work with other shapes
                        print(poly)
                        raise ValueError from e
        # make sure the mask is boolean
        #full_mask = full_mask*1
                    
        for region in self.regions:
            # height, width --> largest x - smallest x, largest y - smallest y
            
            region['x'] = min(p[0] for p in region['coord'])
            region['y'] = min(p[1] for p in region['coord'])
            region['w'] = max(p[0] for p in region['coord']) - region['x']
            region['h'] = max(p[1] for p in region['coord']) - region['y']
            
            region["mask"] = full_mask[:, region['y']:region['y']+region['h'], region['x']:region['x'] + region['w']]
            
            region["image"] = np.zeros((len(self.in_channels), region['h'], region['w']), dtype=np.int16)

            # load the region
            region = self.load_region(region)

    def sample_position(self, width, height):
        # get a random region
        region = np.random.choice(self.regions)
        
        # Find positive mask positions (any channel with True values)
        # Combine all mask channels to find any positive area

        if random.randint(0, 100) < 12:
            x = np.random.randint(region['x'], region['x']+region['w']-width)
            y = np.random.randint(region['y'], region['y']+region['h']-height)
        else:
            combined_mask = np.any(region["mask"], axis=0)
            
            # Get coordinates of all positive mask positions
            positive_coords = np.where(combined_mask)
            
            if len(positive_coords[0]) == 0:
                # If no positive mask found, fall back to random sampling
                x = np.random.randint(region['x'], region['x']+region['w']-width)
                y = np.random.randint(region['y'], region['y']+region['h']-height)
            else:
                # Select a random positive position
                idx = np.random.randint(len(positive_coords[0]))
                center_y, center_x = positive_coords[0][idx], positive_coords[1][idx]
                
                # Sample around the positive position (within 1/2 width and height)
                half_width = width
                half_height = height
                
                # Calculate sampling bounds around the center position
                min_x = max(region['x'], region['x'] + center_x - half_width)
                max_x = min(region['x'] + region['w'] - width, region['x'] + center_x + half_width - width)
                min_y = max(region['y'], region['y'] + center_y - half_height)
                max_y = min(region['y'] + region['h'] - height, region['y'] + center_y + half_height - height)
                
                # Ensure valid bounds
                if max_x < min_x:
                    max_x = min_x
                if max_y < min_y:
                    max_y = min_y
                
                # Sample position around the positive area
                # x = np.random.randint(min_x, max_x + 1) if max_x >= min_x else min_x
                # y = np.random.randint(min_y, max_y + 1) if max_y >= min_y else min_y
                # use a normal distribution around the center
                x = int(np.clip(np.random.normal(center_x, half_width / 2), min_x, max_x))
                y = int(np.clip(np.random.normal(center_y, half_height / 2), min_y, max_y))

        return (x, y, width, height), region

    def pad_if_needed(self, img, h, w):
        # pad to h, w if needed
        if img.shape[1] < h:
            img = np.pad(img, ((0, 0), (0, h - img.shape[1]), (0, 0)), mode='constant')

        if img.shape[2] < w:
            img = np.pad(img, ((0, 0), (0, 0), (0, w - img.shape[2])), mode='constant')
    
        return img
    
    def load_image(self, region, position):
        x, y, h, w = position
        
        # make sure we don't go out of bounds
        # _w = min(w, self.image_width - x)
        # _h = min(h, self.image_height - y)
        r_x = x - region['x']
        r_y = y - region['y']
        image = region["image"][:, r_y:r_y + h, r_x:r_x + w]
        image = self.pad_if_needed(image, h, w)

        # normalize image
        norm_method = self.info_dict["metadata"]["normalization_method"]
        norm_settings = self.info_dict["metadata"]["normalization_settings"]
        image = transform_image(image, norm_settings, norm_method)

        return image
    
    def get_mask(self, region, position):
        x, y, h, w = position
        dx = x - region['x']
        dy = y - region['y']
        mask = region["mask"][:, dy:dy + h, dx:dx + w]
        mask = self.pad_if_needed(mask, h, w)
        mask = mask.astype(np.uint8)

        return mask

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, size=(128, 128), transforms=None, augmentations=None, repeats=64):
        self.data_dict = data_dict
        # increase the size by 50% to allow for some padding
        # this will be removed after the augmentation
        # and will prevent transformations to have padding issues
        self.size = np.int_(np.array(size) * 1.5) 
        # aug and transforms are from albumentations
        self.transform = transforms
        self.augmentation = augmentations
        self.repeats = repeats

    def __len__(self):
        return len(self.data_dict) * self.repeats
    
    def get_weights(self, mask):
        import numpy as np
        from skimage.morphology import binary_dilation, binary_erosion, disk
        from scipy.ndimage import uniform_filter

        weights = np.zeros_like(mask, dtype=np.float32)

        for i in range(mask.shape[0]):
            weights[i] = mask[i] > 0.5
            outline = (binary_dilation(mask[i], footprint=disk(1)) ^ binary_erosion(mask[i], footprint=disk(3))).astype(np.float32)
            outline = outline * 0.5
            # apply blur
            outline = uniform_filter(outline, size=3, mode='constant', cval=0.0)
            weights[i] = 1- outline

        return weights
        
    def __getitem__(self, index):
        index = index % len(self.data_dict)
        pos, region = self.data_dict[index].sample_position(self.size[0], self.size[1])
        image = self.data_dict[index].load_image(region, pos)
        mask = self.data_dict[index].get_mask(region, pos)

        # expects channels last
        image = np.transpose(image, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.transform:
            sample = self.transform(image=image, mask=mask) 
            image, mask = sample['image'], sample['mask']

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        weight = self.get_weights(mask)
        return image, np.float32(mask), weight