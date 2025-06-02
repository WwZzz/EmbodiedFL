import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fmodule
import torch
import numpy as np

class Server(fedavg.Server):
    def iterate(self):
        self.selected_clients = self.sample()
        res = self.communicate(self.selected_clients)
        self.model = self.aggregate(res['model'])
        mean_batch_loss = np.array(res['loss']).mean()
        if hasattr(self.gv.logger, 'writter'):
            self.gv.logger.write_var_into_output('train_loss', mean_batch_loss)
            self.gv.logger.writter.add_scalar('train_loss', mean_batch_loss, self.current_round)
        return

class Client(fedavg.Client):
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        loss = self.train(model)
        cpkg = self.pack(model, loss)
        return cpkg

    def pack(self, model, loss, *args, **kwargs):
        return {'model': model, 'loss': loss}

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        total_loss = 0.
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad > 0: torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            optimizer.step()
            total_loss += loss.item()
        total_loss /= self.num_steps
        return total_loss
