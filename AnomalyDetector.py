import numpy as np
import pandas as pd
from bamt.networks import HybridBN
from scipy.spatial import cKDTree
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, Dropout
from tensorflow.keras.models import Model

class AnomalyDetector:
    def __init__(self, bn_name='bn.json', model_path='model.keras', image_size=(256, 256), threshold=0.25, transparency=128):
        self.bn_name = bn_name
        self.model_path = model_path
        self.image_size = image_size
        self.threshold = threshold
        self.transparency = transparency
        self.bn = HybridBN(has_logit=True)
        self.bn.load(bn_name)
        self.model = tf.keras.models.load_model(model_path)

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
        img_dim = 256
        image_datagen = ImageDataGenerator(rescale=1./255)
        test_image_generator = image_datagen.flow_from_directory(
            'QT',
            target_size=(img_dim, img_dim),
            class_mode=None,
            classes=['images'],
            batch_size=32,
            seed=42,
            shuffle=False
        )

        predict = self.model.predict(test_image_generator)
        predict[predict >= 0.5] = 1
        predict[predict < 0.5] = 0
        return predict

    def extract_mask(self, image_with_mask, mask_size, mask_position=(0, 0)):
        x, y = mask_position
        mask_width, mask_height = mask_size
        extracted_mask = image_with_mask[y:y + mask_height, x:x + mask_width, 0]
        return extracted_mask

    def get_anom_mask(self, data, small_lat, small_lon):
        large_lat = np.load('lat_full.npy')
        large_lon = np.load('lon_full.npy')
        land_water_mask = np.load('land_mask_full.npy')
        cluster_mask = np.load('clusters_full.npy')

        small_land_water_mask, small_cluster_mask = self.interpolate_mask(small_lat, small_lon, large_lat, large_lon, land_water_mask, cluster_mask)
        prob_mask, anomaly_mask = self.prob_mask_generation(data, small_land_water_mask, small_cluster_mask)
        self.create_ice_concentration_image(data, small_land_water_mask, prob_mask)
        predict = self.deepl_pred()
        mask = self.extract_mask(predict[0], mask_size=(small_land_water_mask.shape[1], small_land_water_mask.shape[0]), mask_position=(0, 0))

        file_path = 'QT/images/1.png'
        if os.path.exists(file_path):
            os.remove(file_path) 
        return mask, small_land_water_mask

