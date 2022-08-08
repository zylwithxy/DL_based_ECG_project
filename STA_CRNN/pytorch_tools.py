
class EarlyStopping():
    """Early stops the training if F1 score doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
        self.net_parameters = None
        self.confuse_mat_return = None
    def __call__(self, F1_score, net_params, confuse_mat):
        """
        F1_score: F1_score of the validation set. For instance: 0.75
        """

        if self.best_score is None:
            self.best_score = F1_score
        elif F1_score < self.best_score - self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if F1_score > self.best_score:
                self.best_score = F1_score
                self.net_parameters = net_params
                self.confuse_mat_return = confuse_mat
                self.counter = 0