import matplotlib.pyplot as plt
import numpy as np
import os

class ECG_Show():
    def __init__(self, real_ecg) -> None:   
        """
        real_ecg: shape: (1350, 5500) Real ECG data and the lead is confirmed.
        """
        self.path = '/home/alien/XUEYu/Draft/GAN_pic/generate_lead'
        assert len(real_ecg) == 1350
        self.real =  real_ecg

    def plot_real(self, real):
        pass

    def plot_fake(self, fake, leadnum, model_name):
        """
        fake: List [18, 5500].
        leadnum: 1-12. Represent the lead num.
        model_name: Denote
        """
        FIG_NUM = 2
        for i in range(FIG_NUM):
            fig_f, axes_f = plt.subplots(9, sharex=True, figsize=(15, 27))
            setattr(self, f"fig_f{i + 1}", fig_f)
            setattr(self, f"axes_f{i + 1}", axes_f)
            cur_axe = getattr(self, f"axes_f{i + 1}")
            for j in np.arange(9):
                cur_axe[j].grid(which='both', axis='both', linestyle='--')
                cur_axe[j].plot(fake[i*9 + j], color="salmon", label= 'Synthetic signal')
                cur_axe[j].plot(self.real[i * 5 + 10 + j*150], color="b", label= 'Real signal')
                cur_axe[j].set_title(f'Sample{i+1}_'+f'Type {j+1}')
                if j == 0:
                    cur_axe[j].legend(loc= 'upper right')
        
        for i in range(FIG_NUM):
            cur_fig = getattr(self, f"fig_f{i + 1}")
            cur_fig.tight_layout(pad= 1.08 * 1.5, h_pad= 2.5)
            top, bottom = cur_fig.subplotpars.top, cur_fig.subplotpars.bottom
            cur_fig.subplots_adjust(top= top - 0.025, bottom= bottom + 0.024)

            filename = 'lead' + str(leadnum)
            cur_path = os.path.join(self.path, filename)
            if not os.path.exists(cur_path):
                os.makedirs(cur_path)

            file_count = len(os.listdir(cur_path))
            cur_fig.savefig(os.path.join(cur_path, f'{file_count+1}_Sample_{i+1}_synthetic_'+ model_name +'.png'))
        
        plt.show()