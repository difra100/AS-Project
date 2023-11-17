import pytorch_lightning as pl


class pl_Dataset_(pl.LightningDataModule):

    def __init__(self, train_loader, val_loader):

        self.train_loader = train_loader
        self.val_loader = val_loader


        

    def setup(self, stage=None):
        if stage == 'fit':
            print("")
        elif stage == 'test':
            print("")
            
    def train_dataloader(self, *args, **kwargs):
        return self.train_loader
    def val_dataloader(self, *args, **kwargs):
        return self.val_loader