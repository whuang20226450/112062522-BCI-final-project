import numpy as np
import matplotlib.pyplot as plt
import mne
import scipy.io as sio
import os

def get_label_mapping():
    # make labels
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
        'animal': ['animal', 'bird', 'insect'],
        'human body': ['body part'],
        'clothing and accessories': ['clothing', 'clothing accessory'],
        'food': ['dessert', 'drink', 'food', 'fruit', 'vegetable'],
        'home and furniture': ['furniture', 'home decor'],
        'kitchen': ['kitchen appliance', 'kitchen tool'],
        'electronics': ['electronic device'],
        'medical equipment': ['medical equipment'],
        'office supply': ['office supply'],
        'musical instrument': ['musical instrument'],
        'vehicle': ['part of car', 'vehicle'],
        'toy': ['toy'],
        'plant': ['plant'],        
        'other': ['tool', 'sports equipment', 'weapon', 'container'],
    }
    groupedCategories2label = {
        'animal': 0,
        'human body': 1,
        'clothing and accessories': 2,
        'food': 3,
        'home and furniture': 4,
        'kitchen': 5,
        'electronics': 6,
        'medical equipment': 7,
        'office supply': 8,
        'musical instrument': 9,
        'vehicle': 10,
        'toy': 11,
        'plant': 12,
        'other': 13,
    }
    reverse_grouped_categories = {v: k for k, values in grouped_categories.items() for v in values}
    thingsId2label = [groupedCategories2label[reverse_grouped_categories[categories[label]]] for label in thingsId2label]
    return thingsId2label

if __name__ == "__main__":
    
    dataset_id = '1130'
    if dataset_id[2] == '3':
        input_name = f"{dataset_id[:2] + '0' + dataset_id[3]}.set"
    else:
        input_name = f"{dataset_id}.set"
    output_name = f"../processed/{dataset_id}.npy"
    
    raw = mne.io.read_raw_eeglab(input_name, preload=True)
    events, event_id = mne.events_from_annotations(raw)
    event_desc = {v: k for k, v in event_id.items()}
    for i in range(events.shape[0]):
        events[i, 2] = event_desc[events[i, 2]]
        
    epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0), preload=True)
    del raw
    
    # run auto-reject for 'XX3X'
    if dataset_id[2] == '3' and not os.path.exists('rejected_epochs.txt'):
        from autoreject import AutoReject
        ar = AutoReject()
        epochs = ar.fit_transform(epochs) 

    # (83064, 63/17, 129)
    data = epochs.get_data()
    del epochs
    
    thingsId2label = get_label_mapping()        
    labels = []
    for i in range(events.shape[0]):
        labels.append(thingsId2label[events[i, 2] - 1])
    labels = np.array(labels)        
    print(len(labels))  
    
    # delete rejected epochs for autoreject
    if dataset_id[2] == '3':
        with open('rejected_epochs.txt', 'r') as file:
            content = file.read()
        rejected_epochs = [int(num.strip()) - 1 for num in content.split(',')]
        
        data = np.delete(data, rejected_epochs, axis=0)
        labels = np.delete(labels, rejected_epochs)
    
    # shuffle data
    idx = np.arange(len(labels))
    np.random.seed(0)
    np.random.shuffle(idx)
    data, labels = data[idx], labels[idx]
    
    # split to train/val/test
    size = len(labels)
    train_size, val_size, test_size = int(size * 0.7), int(size * 0.15), int(size * 0.15)
    train_data, train_labels = data[:train_size], labels[:train_size]
    val_data, val_labels = data[train_size:train_size+val_size], labels[train_size:train_size+val_size]
    test_data, test_labels = data[train_size+val_size:], labels[train_size+val_size:]
    
    # check label distribution
    print("label distribution")
    for labels in [train_labels, val_labels, test_labels]:
        label_count = np.zeros(14)
        for label in labels:
            label_count[label] += 1
        print(label_count / len(labels))
        
    # save data
    np.save(output_name, 
                {"train_x": train_data, 
                "train_y": train_labels, 
                "val_x": val_data, 
                "val_y": val_labels, 
                "test_x": test_data, 
                "test_y": test_labels})
    
    data = np.load(output_name, allow_pickle=True).item()
    print(data['train_x'].shape, data['train_y'].shape)
    print(data['val_x'].shape, data['val_y'].shape)
    print(data['test_x'].shape, data['test_y'].shape)