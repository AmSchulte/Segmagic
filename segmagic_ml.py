import torch
from model_ml import Model
import pytorch_lightning as lit
import math
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob
import tifffile as tiff
from pathlib import Path
import cv2 
from shapely.geometry import Polygon
import shapely
import geojson
import pandas as pd
from normalize import fit_normalizer, transform_image
import ttach as tta        

class Segmagic():
    def __init__(self, settings):
        self.settings = settings.to_dict()
        self.model_folder = settings["data"]["model_folder"]
        self.kernel_size = settings["dataset"]["kernel_size"]
        self.ensemble = False
        self.model_name = '/best_model.pth'
    
    def train_model(self, data):        

        training_params = {
            "labels":data.labels,
            "model_path": self.settings["data"]["model_folder"] + self.model_name,
            "in_channels":len(data.train_data[0].in_channels),
            "classes":len(data.labels),
            "spe": len(data.train_dl),
            **self.settings
        }

        self.model = Model(
            training_params
        )

        trainer = lit.Trainer(
            accelerator="auto",
            max_epochs=self.settings["training"]["epochs"], 
            precision="16-mixed",
            # see if gradient clipping of 4 is better
            gradient_clip_val=1.0,
        )

        trainer.fit(
            self.model, 
            data.train_dl, data.valid_dl
        )

    def train_model_ensemble(self, model_n, data):
        for i in range(model_n):
            data.train_test_split(test_ratio=0.1, valid_ratio=0.2, random=None)
            # kernel_size and batch_size can be adjusted
            data.run_data_loader()
            self.model_name = f'/SCUnet_model_{str(i)}.pth'
            self.train_model(data)

        
    def load_model(self):
        filepaths = glob.glob(f"{self.model_folder}/*.pth")
        if len(filepaths) > 1:
            self.ensemble = True
            self.models = []
            print(f"Loading models for ensemble")

            for filepath in filepaths:
                
                if 'SCUnet_model' in filepath:
                    print(f"Loading model from {filepath}")

                    model = torch.load(filepath, weights_only=False)
                    model.eval()
                    model.cuda()
                    self.models.append(model)

            #self.fmodel, self.params, self.buffers = combine_state_for_ensemble(self.models)
        else:
            self.model = torch.load(filepaths[0], weights_only=False)
            self.model.eval()
            self.model.cuda()
            print(f"Loaded model from {filepaths[0]}")

    
    # from http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python-from-scratch/
    def dnorm(self, x, mu, sd):
        return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

    def gaussian_kernel(self, size, sigma=1, verbose=False):
        kernel_1D = np.linspace(-(size // 2), size // 2, size)
        for i in range(size):
            kernel_1D[i] = self.dnorm(kernel_1D[i], 0, sigma)
        kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    
        kernel_2D *= 1.0 / kernel_2D.max()
    
        if verbose:
            plt.imshow(kernel_2D, interpolation='none', cmap='gray')
            plt.title("Kernel ( {}X{} )".format(size, size))
            plt.show()
    
        return kernel_2D

    def weight_function(self):
        return self.gaussian_kernel(self.kernel_size, sigma=self.kernel_size/8)

    def predict_image(self, image_to_predict, labels, threshold=0.5, show=False):
        if self.ensemble:
            print(f"Using ensemble of {len(self.models)} models for prediction")
            tta_models = [tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean') for model in self.models]
        else:
            print(f"Using single model for prediction")
            tta_models = [tta.SegmentationTTAWrapper(self.model, tta.aliases.d4_transform(), merge_mode='mean')]

        STEP_SCALE = 0.5
        STEP_SIZE = int(self.kernel_size * STEP_SCALE)
        #image_to_predict = image_to_predict.transpose(2, 0, 1)
        image_width = image_to_predict.shape[2]
        image_height = image_to_predict.shape[1]

        #self.load_model()
        x_padding = STEP_SIZE - image_width % STEP_SIZE
        y_padding = STEP_SIZE - image_height % STEP_SIZE


        # fill predicted mask to a tile size
        predicted_mask = np.zeros((
            image_height + y_padding,
            image_width + x_padding,
            len(labels)
        ))

        predicted_weighting = np.zeros((
            image_height + y_padding,
            image_width + x_padding
        ))

        weight = self.weight_function()
        x_steps = image_width // STEP_SIZE
        y_steps = image_height // STEP_SIZE

        # reading image, adjust code here for new images
        #img = image_to_predict.load_image(image_to_predict.regions[0], (0,0,image_to_predict.image_height,image_to_predict.image_width))
        img = image_to_predict.copy()
        
        #padding image
        normalized_img = np.pad(img, ((0,0), (int(y_padding/2), math.ceil(y_padding/2)), (int(x_padding/2), math.ceil(x_padding/2))), mode="edge")

        # normalize image
        normalized_img = (normalized_img - 0.5) / 0.25

        pbar = tqdm(total=x_steps * y_steps)
        with torch.no_grad():
            for x in range(x_steps):
                for y in range(y_steps):
                    x0 = x * STEP_SIZE
                    y0 = y * STEP_SIZE
                    x1 = self.kernel_size
                    y1 = self.kernel_size
                    
                    img_tile = normalized_img[:,y0:y0+y1, x0:x0+x1]
                    
                
                    img_tile = torch.from_numpy(img_tile).unsqueeze(0)
                    
                    outputs = []
                    for e, model in enumerate(tta_models):
                        pred_m = model(img_tile.cuda())
                        outputs.append(pred_m)
                    
                    pred = sum(outputs) / len(outputs)                        

                    pred = pred.squeeze(0).sigmoid().cpu().numpy()
                    
                    pred = pred.transpose(1, 2, 0)
                    
                    predicted_mask[y0:y0+y1, x0:x0+x1,:] += pred*np.expand_dims(weight, axis=2)
                    predicted_weighting[y0:y0+y1, x0:x0+x1] += weight
                    pbar.update(1)

        pbar.close()

        predicted_mask /= predicted_weighting[..., None]

        # remove padding
        predicted_mask = predicted_mask[int(y_padding/2):image_height+math.ceil(y_padding/2), int(x_padding/2):image_width+math.ceil(x_padding/2)]

        # uncertainty measure
        # numpy array p_hat with dimension (N_uncertain, width, height, depth)
        
        p_hat = np.expand_dims(predicted_mask, axis=0)
        epistemic = np.mean(p_hat**2, axis=0) - np.mean(p_hat, axis=0)**2
        aleatoric = np.mean(p_hat*(1-p_hat), axis=0)
        # Add uncertainties
        uncertainty = epistemic + aleatoric
        # Scale to 1 max overall
        uncertainty /= 0.25

        predicted_mask = np.uint8((predicted_mask>threshold)*255)

        if show:
            # conver image to uint8 for visualization
            img = np.uint8(image_to_predict*255)
            for label in range(len(labels)):
                # make subfigures and also show the uncertainty

                fig, axs = plt.subplots(1,2, figsize=(10,5))
                axs[0].set_title('Prediction')
                axs[0].imshow(img[0,:,:], cmap='gray')
                axs[0].imshow(predicted_mask[..., label], alpha=0.4, cmap="inferno")

                axs[1].set_title('Uncertainty')
                axs[1].imshow(uncertainty[..., label], cmap="gray")
                plt.show()

        return predicted_mask, uncertainty
    
    def get_dice(self, img1, img2):
        img1 = np.asarray(img1).astype(bool)
        img2 = np.asarray(img2).astype(bool)

        if img1.shape != img2.shape:
            # we seem to have padded, so we need to crop
            min_shape = np.minimum(img1.shape, img2.shape)
            img1 = img1[:min_shape[0], :min_shape[1]]
            img2 = img2[:min_shape[0], :min_shape[1]]

        intersection = np.logical_and(img1, img2)
        return 2. * intersection.sum() / (img1.sum() + img2.sum())
            
    def test_images(self, data):
        Path(data.base_path+'/Testing/images').mkdir(parents=True, exist_ok=True)
        Path(data.base_path+'/Testing/masks_ann').mkdir(parents=True, exist_ok=True)
        Path(data.base_path+'/Testing/masks_pred').mkdir(parents=True, exist_ok=True)
        Path(data.base_path+'/Testing/masks_uncertainty').mkdir(parents=True, exist_ok=True)
        

        results = {'image_name':[], 'Dice_score':[], 'Uncertainty_score':[]}
        # load model
        self.load_model()
        
        for i in range(len(data.test_data)):
            image_to_predict = data.test_data[i]
            
            test_image = image_to_predict.load_image(image_to_predict.regions[0], (0,0,image_to_predict.image_height,image_to_predict.image_width))
            
            predicted_mask, uncertainty = self.predict_image(test_image, data.labels, show=False)
            
            img = image_to_predict.load_image(image_to_predict.regions[0], (0,0,image_to_predict.image_height,image_to_predict.image_width))
            mask = image_to_predict.get_mask(image_to_predict.regions[0], (0,0,image_to_predict.image_height,image_to_predict.image_width))

            name = data.test_data[i].info_dict['image_name']
            tiff.imwrite(data.base_path+'/Testing/images/'+ name, img)
            tiff.imwrite(data.base_path+'/Testing/masks_ann/'+ name, np.uint8(mask.transpose(1, 2, 0)*255))
            tiff.imwrite(data.base_path+'/Testing/masks_pred/'+ name, predicted_mask)
            tiff.imwrite(data.base_path+'/Testing/masks_uncertainty/'+ name, np.uint8(uncertainty*255))
            # divide by 255 if analyzing the uncertainty later!
            

            for label in range(len(data.labels)):
                results['image_name'].append(name+'_'+data.labels[label])
                dice = self.get_dice(mask[label,:,:], predicted_mask[..., label])
                uncertainty_score = np.mean(uncertainty[..., label][predicted_mask[..., label]>0])
                results['Dice_score'].append(dice)
                results['Uncertainty_score'].append(uncertainty_score)

                # conver image to uint8 for visualization
                img = np.uint8(img*255)

                fig, axs = plt.subplots(1,3, figsize=(15,5))

                axs[0].set_title('Annotation')
                axs[0].imshow(img[0,:,:], cmap='gray')
                axs[0].imshow(mask[label,:,:], alpha=0.4, cmap="inferno")

                axs[1].set_title('Prediction Dice: '+str(round(dice,3)))
                axs[1].imshow(img[0,:,:], cmap='gray')
                axs[1].imshow(predicted_mask[..., label], alpha=0.4, cmap="inferno")

                axs[2].set_title('Uncertainty: '+str(round(uncertainty_score,3)))
                axs[2].imshow(uncertainty[..., label], cmap="gray")
                plt.savefig(f"{data.base_path}/Testing/Test_result_{data.labels[label]}_"+str(i)+'.png')
                plt.show()
            
            results_df = pd.DataFrame(results)
            results_df.to_excel(data.base_path+'/Testing/test_results.xlsx')
    
    def predict_folder(self, folder_path, labels, show=False):
        folder_saveto = folder_path + '_pred/masks'
        folder_saveto_json = folder_path + '_pred/jsons'
        folder_saveto_uncertainty = folder_path + '_pred/uncertainty'

        Path(folder_path + '_pred').mkdir(parents=True, exist_ok=True)
        Path(folder_saveto).mkdir(parents=True, exist_ok=True)
        Path(folder_saveto_json).mkdir(parents=True, exist_ok=True)
        Path(folder_saveto_uncertainty).mkdir(parents=True, exist_ok=True)

        filepaths = glob.glob(f"{folder_path}/*.tif") + glob.glob(f"{folder_path}/*.tiff")

        self.load_model()

        for filepath in tqdm(filepaths):
            # ensure the filename ends with .tif of tiff
            filename = Path(filepath).name

            image_to_predict = tiff.imread(filepath)
            if image_to_predict.ndim == 2:
                image_to_predict = np.expand_dims(image_to_predict, axis=2)
            image_to_predict = image_to_predict.transpose(2, 0, 1)
            
            norm_method = self.settings["dataset"]["normalization_method"]
            norm_settings = self.settings["dataset"]["normalization_settings"]
            norm_settings = fit_normalizer(image_to_predict, norm_method)
            image_to_predict = transform_image(image_to_predict, norm_settings, norm_method)

            predicted_mask, uncertainty = self.predict_image(image_to_predict, labels, show=show)
            tiff.imwrite(f"{folder_saveto}/{filename}", predicted_mask, metadata={'labels': labels})
            tiff.imwrite(f"{folder_saveto_uncertainty}/{filename}", np.uint8(uncertainty*255), metadata={'labels': labels})

            self.save_geojson(predicted_mask, labels, f"{folder_saveto_json}/{filename}")

    
    def save_geojson(self, predicted_mask, labels, saveto):
        features = []

        for e, label in enumerate(labels):
            
            mask = predicted_mask[..., e]
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if hierarchy is not None:
                blob_idx = np.squeeze(np.where(hierarchy[0, :, 3] == -1))

                polys = []
                
                for b_idx in np.nditer(blob_idx):
                    this_poly = contours[b_idx]
                    this_poly = np.array(this_poly).squeeze(1)
                    holes = []

                    cnt_idx = np.squeeze(np.where(hierarchy[0, :, 3] == b_idx))
                    if cnt_idx.size > 0:
                        holes += [np.array(contours[c]).squeeze(1) for c in np.nditer(cnt_idx)]

                    if len(this_poly) > 5:
                        polys.append(Polygon(this_poly, holes=holes))
                
                # maybe limit popygon size here
                #polys = [p for p in polys if p.area > 200]
                #polys = [p for p in polys if p.area < 100000]

                features += [{
                    'type': 'Feature', 
                    'properties': {"classification": {"colorRGB": -2315298, "name": f"p_{label}"}, "isLocked": False, "measurements": []}, 
                    'id': 'PathAnnotationObject', 
                    'geometry': shapely.geometry.mapping(p)} 
                    for p in polys]
                        
        with open(f'{saveto}.geojson', 'w') as outfile:
            geojson.dump(features, outfile)
        
