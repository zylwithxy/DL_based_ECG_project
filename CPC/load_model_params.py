import torch
import os
from Model import G_enc, STA_Classifier

def load_model_params(checkpoint, GRU_hidden, gru_layer, timestep):

    conv_arch = [(2,64),(2,128),(3,256),(3,256),(3,256)]
    params = (timestep, 64, 15360, conv_arch, GRU_hidden, gru_layer)
    g_enc = G_enc(*params).cuda()

    for key, value in checkpoint['state_dict'].items():
        print(value.device)
        break

    g_enc.load_state_dict(checkpoint['state_dict'])
    g_enc.eval()
    print(g_enc.gru_layers)

    return g_enc


def load_ftmodel_params(GRU_hidden, index_ft, index_pre, random_state, 
                        timestep_state, timestep, classifier):
    """
    GRU_hidden: The hidden number of GRU units.
    index_ft: The index of fine tuning.
    index_pre: The index of pre-training models.
    random_state: bool. True means random init.
    timestep_state: bool. whether the timestep is 12.
    classifier: the name of classifier
    """
    temp = f'Fine_tuning_cdc_1_fold_GRU_{GRU_hidden}_{index_pre}_{index_ft}.pth'
    temp_time_step = temp if timestep_state else f'Timestep_{timestep}_' + temp
    fine_tuning_name  = temp_time_step if not random_state else 'Random_' + temp_time_step
    fine_tuning_name = fine_tuning_name if classifier == 'class_1' \
                        else classifier + '_' + fine_tuning_name

    """
    if random_state:
        fine_tuning_name = f'Random_Fine_tuning_cdc_1_fold_GRU_{GRU_hidden}_{index_pre}_{index_ft}.pth'
    else:
        fine_tuning_name = f'Fine_tuning_cdc_1_fold_GRU_{GRU_hidden}_{index_pre}_{index_ft}.pth'
    """

    checkpoint = torch.load(os.path.join(PATH, fine_tuning_name), 
                            map_location= lambda storage, loc: storage)

    print(checkpoint['best_F1'],
          checkpoint['validation_loss'] 
          )

    
def load_premodel_params(GRU_hidden, index, timestep_state, timestep):
    """
    index: The index of pre-training models.
    timestep_state: bool. whether the timestep is 12
    """
    filename = f'cdc_1_fold_GRU_{GRU_hidden}_{index}.pth' if timestep_state \
               else f'Timestep_{timestep}_'+f'cdc_1_fold_GRU_{GRU_hidden}_{index}.pth'
    checkpoint = torch.load(os.path.join(PATH, filename), map_location= lambda storage, loc: storage)
    print(checkpoint['validation_loss'],
          checkpoint['validation_loss'].device, 
          checkpoint['validation_acc'])

if __name__ == '__main__':
    
    PATH = '/home/alien/XUEYu/paper_code/Parameters/2018_CPC'
    timestep_state = True # whether the timestep is 12
    timestep = 24
    random_state = False
    index_ft = 21
    index_pre = 9
    GRU_hidden = 5
    classifier = 'class_3'
    gru_layer = 2

    filename = 'layers_gru_2_Timestep_24_cdc_1_fold_GRU_5_24.pth'
    checkpoint = torch.load(os.path.join(PATH, filename))
    # Load params on CPU
    
    g_enc = load_model_params(checkpoint, GRU_hidden, gru_layer, timestep) # Loading parameters to the model

    # load_premodel_params(GRU_hidden, index_pre, timestep_state, timestep)
    # print('Fine-tuning metrics')
    """
    load_ftmodel_params(GRU_hidden, index_ft, index_pre, random_state, 
                        timestep_state, timestep, classifier)
    """