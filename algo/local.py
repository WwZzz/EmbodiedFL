import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fmodule
import torch

class Server(fedavg.Server):
    pass

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.all_steps = 0

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            self.all_steps += 1
            if hasattr(self.gv.logger, 'writter'):
                self.gv.logger.writter.add_scalar('train/batch_loss', loss.item(), self.all_steps)
        return

