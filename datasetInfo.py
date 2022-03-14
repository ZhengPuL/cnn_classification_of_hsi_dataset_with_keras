class DatasetInfo:
    info = {
        'indian':{
            'data_path':'./data/Indian/Indian_pines_corrected.mat',
            'label_path':'./data/Indian/Indian_pines_gt.mat',
            'data_key':'indian_pines_corrected',
            'label_key':'indian_pines_gt',
            'target_names': [
        'Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn',
        'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
        'Stone-Steel-Towers']
        },
        'ksc':{
            'data_path': './data/KSC/KSC.mat',
            'label_path': './data/KSC/KSC_gt.mat',
            'data_key': 'KSC',
            'label_key': 'KSC_gt',
            'target_names':[
                'Scrub', 'Willow swamp', 'Cabbage palm hammock',
                'Cabbage palm/oak hammock', 'Slash pine', 'Oak/broadleaf hammock',
                'Hardwood swamp', 'Graminoid marsh', 'Spartina marsh',
                'Cattail marsh', ' Salt marsh', ' Mud flats', 'Water'
            ]
        },
        'pavia':{
            'data_path':'./data/Pavia/Pavia.mat',
            'label_path':'./data/Pavia/Pavia_gt.mat',
            'data_key':'pavia',
            'label_key':'pavia_gt'
        },
        'salinas':{
            'data_path':'./data/Salinas/Salinas.mat',
            'label_path':'./data/Salinas/Salinas_gt.mat',
            'data_key':'salinas_corrected',
            'label_key':'salinas_gt',
            'target_names':[
                'Broccoli green weeds 1','Broccoli green weeds 22','Fallow',
                'Fallow rough plow','Fallow smooth','Stubble','Celery',
                'Grapes untrained','Soy vineyard develop','Corn senesced green weeds',
                'Lettuce romaine 4wk', ' Lettuce romaine 5wk','Lettuce romaine 6wk',
                ' Lettuce romaine 7wk', 'Vineyard untrained', 'Vineyard vertical trellis'
            ]
        }
    }
