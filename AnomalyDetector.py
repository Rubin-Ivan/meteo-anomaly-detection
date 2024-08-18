import numpy as np
import pandas as pd
from bamt.networks import HybridBN
from scipy.spatial import cKDTree
from PIL import Image
import os
import cv2
import torch
from torchvision import transforms
import os

class AnomalyDetector:
    def __init__(self, bn_name='models/bn.json', model_path='models/model.pt', image_size=(256, 256), threshold=0.25, transparency=128):
        self.bn_name = bn_name
        self.model_path = model_path
        self.image_size = image_size
        self.threshold = threshold
        self.transparency = transparency
        self.bn = HybridBN(has_logit=True)
        self.bn.load(bn_name)
        self.model = torch.jit.load(model_path)

    def interpolate_mask(self, small_lat, small_lon, large_lat, large_lon, land_water_mask, cluster_mask):
        large_lat_flat = large_lat.flatten()
        large_lon_flat = large_lon.flatten()

        large_coords = np.vstack((large_lat_flat, large_lon_flat)).T
        tree = cKDTree(large_coords)

        small_lat_flat = small_lat.flatten()
        small_lon_flat = small_lon.flatten()

        small_coords = np.vstack((small_lat_flat, small_lon_flat)).T
        dist, idx = tree.query(small_coords)

        land_water_flat = land_water_mask.flatten()
        cluster_flat = cluster_mask.flatten()

        small_land_water_mask = land_water_flat[idx].reshape(small_lat.shape)
        small_cluster_mask = cluster_flat[idx].reshape(small_lat.shape)

        return small_land_water_mask, small_cluster_mask

    def prob_mask_generation(self, data, land_mask, clusters):
        list_znach = []
        list_month = []
        list_cluster = []
        list_X = []
        list_Y = []

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if land_mask[y, x] == 0:
                    list_znach.append(data[y, x])
                    list_Y.append(y)
                    list_X.append(x)
                    list_month.append(1)  # Assuming month is constant for simplicity
                    list_cluster.append(clusters[y, x])

        df = pd.DataFrame({
            'VALUE': list_znach,
            'TIME': list_month,
            'CLUSTER': list_cluster,
            'x': list_X,
            'y': list_Y
        })

        df['CLUSTER'] = df['CLUSTER'].astype(int).astype(str)
        df['TIME'] = df['TIME'].astype(int).astype(str)
        df = df.round(2)
        df.loc[df['CLUSTER'] == '0', 'CLUSTER'] = '3'
        data_test2 = df.drop(columns=['x', 'y'])

        prob = self.probability_list(data_test2, ['VALUE', 'CLUSTER', 'TIME'], 'NORM')
        df['prob'] = prob

        max_x = data.shape[1]
        max_y = data.shape[0]
        prob_mask = np.zeros((max_y, max_x))
        for _, row in df.iterrows():
            prob_mask[row['y'], row['x']] = row['prob']

        prob_mask_int = (prob_mask >= self.threshold).astype(int)
        return prob_mask, prob_mask_int

    def probability_list(self, data, node_list, node_class):
        probability_list = []
        for i in range(data.shape[0]):
            probability_list.append(self.bn.get_dist(node_class, {
                node_list[0]: data[node_list[0]][i],
                node_list[1]: data[node_list[1]][i],
                node_list[2]: data[node_list[2]][i]
            })[0])
        return probability_list

    def create_ice_concentration_image(self, ice_data, land_water_mask, probability_mask):
        n, p = ice_data.shape
        ice_data_norm = np.uint8((ice_data / 100.0) * 255)

        ice_image = Image.fromarray(ice_data_norm).convert("L").convert("RGB")
        full_image = Image.new("RGB", self.image_size, (0, 255, 0))
        full_image.paste(ice_image, (0, 0))

        land_mask_rgb = np.zeros((n, p, 3), dtype=np.uint8)
        land_mask_rgb[land_water_mask == 1] = [0, 255, 0]
        land_mask_image = Image.fromarray(land_mask_rgb, "RGB")
        land_mask_alpha = Image.fromarray(np.uint8(land_water_mask * 255), "L")

        full_image.paste(land_mask_image, (0, 0), land_mask_alpha)

        probability_mask_norm = np.uint8(probability_mask * 255)
        probability_image = Image.new("RGBA", (p, n), (0, 0, 255, 0))
        alpha_channel = Image.fromarray(probability_mask_norm).convert("L")
        alpha_channel = alpha_channel.point(lambda x: x * (self.transparency / 255))
        probability_image.putalpha(alpha_channel)

        full_image = full_image.convert("RGBA")
        full_image.alpha_composite(probability_image, (0, 0))

        if not os.path.exists('QT/images'):
            os.makedirs('QT/images')
        full_image.save(f'QT/images/1.png')

    def deepl_pred(self):
        image = cv2.imread('QT/images/1.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Преобразование изображения
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        image = transform(image).unsqueeze(0) 
    
        # Прогоняем через модель
        with torch.no_grad():
            pred_mask = self.model(image)
            pred_mask = (pred_mask > 0.5).float()  
        
        
        pred_mask_np = pred_mask.squeeze().cpu().numpy()
    
        return pred_mask_np

    def extract_mask(self, image_with_mask, mask_size, mask_position=(0, 0)):
        x, y = mask_position
        mask_width, mask_height = mask_size
        extracted_mask = image_with_mask[y:y + mask_height, x:x + mask_width]
        return extracted_mask

    def get_anom_mask(self, data, small_lat, small_lon):
        large_lat = np.load('masks/lat_full.npy')
        large_lon = np.load('masks/lon_full.npy')
        land_water_mask = np.load('masks/land_mask_full.npy')
        cluster_mask = np.load('masks/clusters_full.npy')

        small_land_water_mask, small_cluster_mask = self.interpolate_mask(small_lat, small_lon, large_lat, large_lon, land_water_mask, cluster_mask)
        prob_mask, anomaly_mask = self.prob_mask_generation(data, small_land_water_mask, small_cluster_mask)
        self.create_ice_concentration_image(data, small_land_water_mask, prob_mask)
        predict = self.deepl_pred()
        mask = self.extract_mask(predict, mask_size=(small_land_water_mask.shape[1], small_land_water_mask.shape[0]), mask_position=(0, 0))

        file_path = 'QT/images/1.png'
        if os.path.exists(file_path):
            os.remove(file_path) 
        return mask, small_land_water_mask

