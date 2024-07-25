import torch


class F1Measure():
    def __init__(self) -> None:

        self.tp = 0
        self.fp = 0
        self.fn = 0


    def update(self, y_hat, y):

        y_hat = y_hat.to(torch.bool).flatten()
        y = y.to(torch.bool).flatten()
        self.tp += torch.logical_and(y_hat, y).sum()
        self.fp += torch.logical_and(y_hat, ~y).sum()
        self.fn += torch.logical_and(~y_hat, y).sum()

        return (2*self.tp)/((2*self.tp + self.fp + self.fn) + 1e-6)


class IoUMeasure():
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

        self.n = 0
        self.iou = 0

    def update(self, y_hat, y):
        y_hat = y_hat.to(torch.bool).flatten()
        y = y.to(torch.bool).flatten()
        self.tp += torch.logical_and(y_hat, y).sum()
        self.fp += torch.logical_and(y_hat, ~y).sum()
        self.fn += torch.logical_and(~y_hat, y).sum()
        
        iou = self.tp/(self.tp + self.fn + self.fp + 1e-6)
        
        self.n += 1
        self.iou += iou
        
        return iou  
    
    def get(self):
        return self.tp/(self.tp + self.fn + self.fp + 1e-6), self.iou/self.n 
