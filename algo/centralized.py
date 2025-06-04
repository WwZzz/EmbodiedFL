import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fmodule
import torch

class Server(fedavg.Server):
    def pack(self, client_id, mtype=0, *args, **kwargs):
        return {'model': self.model}

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        self.all_steps = 0
        self.model = self.server.model
        self.optimizer = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)

    @fmodule.with_multi_gpus
    def train(self, model):
        model.train()
        crt_batch_loss = 0.
        for iter in range(self.num_steps):
            # get a batch of data
            batch_data = self.get_batch_data()
            model.zero_grad()
            # calculate the loss of the model on batched dataset through task-specified calculator
            loss = self.calculator.compute_loss(model, batch_data)['loss']
            loss.backward()
            if self.clip_grad>0:torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)
            self.optimizer.step()
            self.all_steps += 1
            if hasattr(self.gv.logger, 'writter'):
                self.gv.logger.write_var_into_output('batch_loss', loss.item())
                self.gv.logger.writter.add_scalar('train/batch_loss', loss.item(), self.all_steps)
            crt_batch_loss += loss.item()
        crt_batch_loss/=self.num_steps
        if hasattr(self.gv.logger, 'writter'):
            self.gv.logger.write_var_into_output('train_epoch_loss', crt_batch_loss)
            self.gv.logger.writter.add_scalar('train/epoch_loss', crt_batch_loss, self.server.current_round)
        return

