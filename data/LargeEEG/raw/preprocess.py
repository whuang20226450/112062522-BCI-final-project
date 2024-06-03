import csv
import os
import mne
import numpy as np
from sklearn.utils import shuffle
import numpy as np
import scipy.io as sio


# This script is from https://github.com/gifale95/eeg_encoding/blob/main/02_eeg_preprocessing
def preprocessing(config):
    
    ### Make mapping from event id to concept id ###
    meta_data = np.load('../raw/image_metadata.npy', allow_pickle=True).item()
    # ['00001_aardvark', '00001_aardvark', '00001_aardvark']
    # ['aardvark_01b.jpg', 'aardvark_02s.jpg', 'aardvark_03s.jpg']
    # ['00001_aardvark', '00001_aardvark', '00001_aardvark']
    # ['00001_aircraft_carrier', '00002_antelope', '00003_backscratcher']
    # ['aircraft_carrier_06s.jpg', 'antelope_01b.jpg', 'backscratcher_01b.jpg']
    # ['00010_aircraft_carrier', '00027_antelope', '00051_backscratcher']
    # print(meta_data['train_img_concepts'][:3]) 
    # print(meta_data['train_img_files'][:3]) 
    # print(meta_data['train_img_concepts_THINGS'][:3]) 
    # print(meta_data['test_img_concepts'][:3]) 
    # print(meta_data['test_img_files'][:3]) 
    # print(meta_data['test_img_concepts_THINGS'][:3]) 
    
    # train 16540 images, each image type has 10 instance, total 1654 types, from 27 categories
    # test 200 images, each image type has 1 instance, total 200 types
    # 16540 16540 16540
    # 200 200 200
    # print(len(meta_data['train_img_files']), len(meta_data['train_img_concepts']), len(meta_data['train_img_concepts_THINGS']))
    # print(len(meta_data['test_img_files']), len(meta_data['test_img_concepts']), len(meta_data['test_img_concepts_THINGS']))
    
    trainEventId2conceptId = [0] + [int(concept.split('_')[0]) for concept in meta_data['train_img_concepts_THINGS']]
    testEventId2conceptId = [0] + [int(concept.split('_')[0]) for concept in meta_data['test_img_concepts_THINGS']]
    
    
    
    ### Loop across data collection sessions ###
    eeg_datas = []
    for data_part in ['test', 'training']:
        for s in range(config["n_ses"]):
            eeg_dir = os.path.join(f"sub{config['sub']:02}", f"ses-{s+1:02}", f"raw_eeg_{data_part}.npy")
            eeg_data = np.load(eeg_dir, allow_pickle=True).item()
            ch_names = eeg_data['ch_names']
            sfreq = eeg_data['sfreq']
            ch_types = eeg_data['ch_types']
            eeg_data = eeg_data['raw_eeg_data']
            
            stim_index = ch_names.index('stim')
            # Reject the target trials (event 99999)
            eeg_data[stim_index] = np.where(eeg_data[stim_index] == 99999, 0, eeg_data[stim_index])
            if data_part == 'training':
                eeg_data[stim_index] = [trainEventId2conceptId[int(event_id)] for event_id in eeg_data[stim_index]]
            elif data_part == 'test':
                eeg_data[stim_index] = [testEventId2conceptId[int(event_id)] for event_id in eeg_data[stim_index]]
            eeg_datas.append(eeg_data)
            print(eeg_data.shape)
            del eeg_data
                
    eeg_datas = np.concatenate(eeg_datas, axis=1)
    info = mne.create_info(ch_names, sfreq, ch_types)
    info.set_montage("standard_1005")
    raw = mne.io.RawArray(eeg_datas, info)
    del eeg_datas



    ### Get events, drop unused channels and reject target trials ###
    events = mne.find_events(raw, stim_channel='stim')
    if config['drop_channel']:
        chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'], '^O *|^P *'))
        new_chans = [raw.info['ch_names'][c] for c in chan_idx]
        raw.pick_channels(new_chans)
    else:
        raw.drop_channels(['stim'])   
    my_annotations = mne.annotations_from_events(
        events=events, sfreq=raw.info['sfreq'], orig_time=None
    )
    raw.set_annotations(my_annotations)     
    raw.resample(config['sfreq'])  

    return raw


if __name__ == "__main__":

    config = {
        'sub': 1,
        'n_ses': 4,
        'sfreq': 128,
        'drop_channel': True,
    }    
    raw = preprocessing(config)         
    raw.export(f'1000.set', fmt='eeglab', overwrite=True)
    # # raw = mne.io.read_raw_eeglab('1000.set', preload=True)
    # # events, event_id = mne.events_from_annotations(raw)
    # # event_desc = {v: k for k, v in event_id.items()}
    # # print(events[:10])
    # # for event in events[:10]:
    # #     print(event_desc[event[2]], event[2])
    
    config['drop_channel'] = False    
    raw = preprocessing(config)         
    mne.export.export_raw(f'0000.set', raw, fmt='eeglab', overwrite=True)
    
    
    




    # thingsId2label = sio.loadmat('category_mat_manual.mat')['category_mat_manual']
    # thingsId2label = np.argmax(thingsId2label, axis=1)
    # categories = [
    #     'animal', 'bird', 'body part', 'clothing', 'clothing accessory', 'container',
    #     'dessert', 'drink', 'electronic device', 'food', 'fruit', 'furniture',
    #     'home decor', 'insect', 'kitchen appliance', 'kitchen tool', 'medical equipment',
    #     'musical instrument', 'office supply', 'part of car', 'plant', 'sports equipment',
    #     'tool', 'toy', 'vegetable', 'vehicle', 'weapon'
    # ]
    # # Grouping similar categories
    # grouped_categories = {
    #     'animal': ['animal', 'bird', 'insect'],
    #     'human body': ['body part'],
    #     'clothing and accessories': ['clothing', 'clothing accessory'],
    #     'food': ['dessert', 'drink', 'food', 'fruit', 'vegetable'],
    #     'home and furniture': ['furniture', 'home decor'],
    #     'kitchen': ['kitchen appliance', 'kitchen tool'],
    #     'electronics': ['electronic device'],
    #     'medical equipment': ['medical equipment'],
    #     'office supply': ['office supply'],
    #     'musical instrument': ['musical instrument'],
    #     'vehicle': ['part of car', 'vehicle'],
    #     'toy': ['toy'],
    #     'plant': ['plant'],        
    #     'other': ['tool', 'sports equipment', 'weapon', 'container'],
    # }
    # groupedCategories2label = {
    #     'animal': 0,
    #     'human body': 1,
    #     'clothing and accessories': 2,
    #     'food': 3,
    #     'home and furniture': 4,
    #     'kitchen': 5,
    #     'electronics': 6,
    #     'medical equipment': 7,
    #     'office supply': 8,
    #     'musical instrument': 9,
    #     'vehicle': 10,
    #     'toy': 11,
    #     'plant': 12,
    #     'other': 13,
    # }
    # reverse_grouped_categories = {v: k for k, values in grouped_categories.items() for v in values}
    # thingsId2label = [groupedCategories2label[reverse_grouped_categories[categories[label]]] for label in thingsId2label]
    

