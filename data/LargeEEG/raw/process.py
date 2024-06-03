import numpy as np
import scipy.io as sio
import torch

thingsId2label = sio.loadmat('category_mat_manual.mat')['category_mat_manual']
thingsId2label = np.argmax(thingsId2label, axis=1)
categories = [
    'animal', 'bird', 'body part', 'clothing', 'clothing accessory', 'container',
    'dessert', 'drink', 'electronic device', 'food', 'fruit', 'furniture',
    'home decor', 'insect', 'kitchen appliance', 'kitchen tool', 'medical equipment',
    'musical instrument', 'office supply', 'part of car', 'plant', 'sports equipment',
    'tool', 'toy', 'vegetable', 'vehicle', 'weapon'
]
# Grouping similar categories
grouped_categories = {
    'Animal': ['animal', 'bird', 'insect'],
    'Human Body': ['body part'],
    'Clothing and Accessories': ['clothing', 'clothing accessory'],
    'Food': ['dessert', 'drink', 'food', 'fruit', 'vegetable'],
    'Home and Furniture': ['furniture', 'home decor'],
    'kitchen': ['kitchen appliance', 'kitchen tool'],
    'electronics': ['electronic device'],
    'medical equipment': ['medical equipment'],
    'office supply': ['office supply'],
    'tool': ['tool'],
    'musical instrument': ['musical instrument'],
    'vehicle': ['part of car', 'vehicle'],
    'sports equipment': ['sports equipment'],
    'toy': ['toy'],
    'weapon': ['weapon'],
    'Plant': ['plant'],
    'Container': ['container'],
}
new_categories = {
    'Animal': 0,
    'Human Body': 1,
    'Clothing and Accessories': 2,
    'Food': 3,
    'Home and Furniture': 4,
    'kitchen': 5,
    'electronics': 6,
    'medical equipment': 7,
    'office supply': 8,
    'tool': 9,
    'musical instrument': 10,
    'vehicle': 11,
    'sports equipment': 12,
    'toy': 13,
    'weapon': 14,
    'Plant': 15,
    'Container': 16,
}
reverse_grouped_categories = {v: k for k, values in grouped_categories.items() for v in values}
thingsId2labelv2 = [new_categories[reverse_grouped_categories[categories[label]]] for label in thingsId2label]

meta_data = np.load('../raw/image_metadata.npy', allow_pickle=True).item()
print(meta_data['train_img_files'][:10]) 
imageId2thingsId = [int(concept.split('_')[0])-1 for concept in meta_data['train_img_concepts_THINGS']] + \
                    [int(concept.split('_')[0])-1 for concept in meta_data['test_img_concepts_THINGS']]

path = '../raw/'
for i in range(1,11):
    data1 = np.load(path+f'sub-0{i}/preprocessed_eeg_training.npy', allow_pickle=True).item()
    data2 = np.load(path+f'sub-0{i}/preprocessed_eeg_test.npy', allow_pickle=True).item()
    
    # 1654 training object concepts × 10 images per concept | 16,540 training image conditions X 4 repetitions
    # 200 test object concepts × 1 image per concept | 200 test image conditions X 80 repetitions
    _, _, num_channels, num_samples = data1['preprocessed_eeg_data'].shape
    new_data = np.concatenate((data1['preprocessed_eeg_data'].reshape(-1, num_channels, num_samples), \
                                data2['preprocessed_eeg_data'].reshape(-1, num_channels, num_samples)), axis=0)  
    imageIds = np.array([imageId for imageId in range(16540) for _ in range(4)] + 
                            [imageId for imageId in range(16540, 16740) for _ in range(80)])
    thingsIds = np.array([imageId2thingsId[imageId] for imageId in imageIds])
    label_v1 = np.array([thingsId2label[thingsId] for thingsId in thingsIds])
    label_v2 = np.array([thingsId2labelv2[thingsId] for thingsId in thingsIds])
    
    print(f"new_data: {new_data.shape}, imageIds: {imageIds.shape}, thingsIds: {thingsIds.shape}, label_v1: {label_v1.shape}, label_v2: {label_v2.shape}")
    
    indices = np.arange(new_data.shape[0])
    np.random.seed(0)
    np.random.shuffle(indices)
    x, label_v1, label_v2 = new_data[indices], label_v1[indices], label_v2[indices]
    
    count1 = np.zeros(27)
    count2 = np.zeros(17)
    for j in range(27):
        count1[j] = np.sum(label_v1 == j)
    for j in range(17):
        count2[j] = np.sum(label_v2 == j)
    print(f"count1: {count1}, count2: {count2}")
    
    np.save(f'S{i}.npy', {"x": x, "label_v1": label_v1, "label_v2": label_v2})
    break
