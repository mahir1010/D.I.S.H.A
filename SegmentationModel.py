import numpy as np
import segmentation_models as sm
import tensorflow as tf
from PIL import Image


class SegmentationModel:

    def __init__(self, backbone, weights_path, image_size=256):
        self.model = sm.Unet(backbone, encoder_weights=None, input_shape=(image_size, image_size, 3))
        latest = tf.train.latest_checkpoint(weights_path)
        self.model.load_weights(latest).expect_partial()
        self.model.compile()
        self.image_size = image_size
        self.list_of_indices = []

    # def predict(self,images,batch_size=30):
    #     output_mask=[]
    #     self.list_of_indices.clear()
    #     dataset = None
    #     for image in images:
    #         patches = tf.image.extract_patches(np.expand_dims(self.add_padding(image),axis=0)/255.0,sizes=[1,self.image_size,self.image_size,1],strides=[1,self.image_size,self.image_size,1],rates=[1,1,1,1],padding='VALID')
    #         self.list_of_indices.append({'total':patches.shape[1]*patches.shape[2],'height':patches.shape[1],'width':patches.shape[2]})
    #         data = tf.data.Dataset.from_tensor_slices(tf.reshape(patches,(patches.shape[1]*patches.shape[2],self.image_size,self.image_size,3)))
    #         dataset = data if dataset is None else dataset.concatenate(data)
    #     dataset=dataset.batch(batch_size)
    #     output = self.model.predict(dataset)
    #     output = np.reshape(output,(output.shape[0],self.image_size,self.image_size))
    #     current_index = 0
    #     for idx,image in enumerate(images):
    #         mask=Image.fromarray(np.zeros((image.shape[0],image.shape[1])))
    #         data_struct = self.list_of_indices[idx]
    #         for height in range(data_struct['height']):
    #             for width in range(data_struct['width']):
    #                 sub_mask = Image.fromarray(output[current_index]*255)
    #                 coords = (width*self.image_size,height*self.image_size)
    #                 mask.paste(sub_mask,coords)#(*coords,coords[0]+self.image_size,coords[1]+self.image_size))
    #                 current_index+=1
    #         output_mask.append(mask)
    #     return output_mask

    def predict(self, image, batch_size=60):
        patches = tf.image.extract_patches(np.expand_dims(self.add_padding(image), axis=0) / 255.0,
                                           sizes=[1, self.image_size, self.image_size, 1],
                                           strides=[1, self.image_size, self.image_size, 1], rates=[1, 1, 1, 1],
                                           padding='VALID')
        self.list_of_indices.append(
            {'total': patches.shape[1] * patches.shape[2], 'height': patches.shape[1], 'width': patches.shape[2]})
        total_height_patches = patches.shape[1]
        total_width_patches = patches.shape[2]
        data = tf.data.Dataset.from_tensor_slices(
            tf.reshape(patches, (patches.shape[1] * patches.shape[2], self.image_size, self.image_size, 3)))
        data = data.batch(batch_size)
        output = self.model.predict(data)
        output = np.reshape(output, (output.shape[0], self.image_size, self.image_size))
        mask = Image.fromarray(np.zeros((image.shape[0], image.shape[1])))
        current_index = 0
        for height in range(total_height_patches):
            for width in range(total_width_patches):
                sub_mask = Image.fromarray(output[current_index] * 255)
                coords = (width * self.image_size, height * self.image_size)
                mask.paste(sub_mask, coords)
                current_index += 1
        return np.array(mask, dtype=np.uint8)

    def add_padding(self, image):
        img = Image.fromarray(image)
        width, height = img.size
        new_width = width + width % self.image_size
        new_height = height + height % self.image_size
        result = Image.new(img.mode, (new_width, new_height), (0, 0, 0))
        result.paste(img, (0, 0))
        return np.array(result)
