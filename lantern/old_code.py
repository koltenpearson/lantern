#old code that might get repurposed

#TODO I probably want to delete these
def wrap_single(item) :
    """convert things to a tuple of length one, if it is not an iterable"""
    if isinstance(item, tuple) or isinstance(item, list) :
        return item

    return (item,)

def unwrap_single(item) :
    """unwrap a tuple of length 1, but otherwise pass the item through"""
    if len(item) == 1 :
        item = item[0]
    return item


#TODO make it so that the basic trainer can be used instead of having to always define your own training loop
def basic_trainer(model_step, dsets, batch_size, start_epoch, end_epoch, threads, logger, saver) :

    train_loader = DataLoader(dsets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=threads)
    if 'val' in dsets :
        val_loader = DataLoader(dsets['val'],
                batch_size=batch_size,
                shuffle=False,
                num_workers=threads)
    if 'test' in dsets :
        test_loader = DataLoader(dsets['test'],
                batch_size=batch_size,
                shuffle=False, 
                num_workers=threads)

    print("starting training")
    while last_epoch < target_epoch :
        last_epoch += 1

        for m in model : m.train()
        logger.log_epoch_start('train')
        model_step(train_loader, logger) 
        model_info.train(train_loader, unwrap_single(model), unwrap_single(loss), unwrap_single(optimizer), logger)

        for m in model : m.eval()
        if 'val' in dsets :
            logger.log_epoch_start('val')
            model_info.train(val_loader, unwrap_single(model), unwrap_single(loss), unwrap_single(optimizer), logger)

        if scheduler[0] is not None :
            for s in scheduler : s.step()

        saver.save(last_epoch)
        logger.log_completion()

        print(f"finished epoch {last_epoch}")


    if 'test' in dsets :

        logger.log_epoch_start('test')
        model_step(test_loader, unwrap_single(model), unwrap_single(loss), unwrap_single(optimizer), logger)
        logger.log_completion()


def basic_train(dataloader, model, loss, optimizer, logger) :
    volatile = not model.training

    for data in dataloader :

        inputs, labels = data

        inputs, labels = get_variables(inputs, labels, volatile, model.on_gpu)

        if model.training :
            optimizer.zero_grad()

        outputs = model(inputs)

        loss_out = loss(outputs, labels)

        if model.training :
            loss_out.backward()
            optimizer.step()

        
        logger.log_scalar_direct('loss', loss_out.data.mean())
        logger.log_batch(inputs, outputs, labels)



def gen_gradpool_train(iter_size) :
    return lambda *args : gradpool_train(iter_size, *args)

def gradpool_train(iter_size, dataloader, model, loss, optimizer, logger) :
    iter_counter = 0

    volatile = not model.training
    
    for batch_counter, data in enumerate(dataloader) :

        inputs, labels = data
        inputs, labels = get_variables(inputs, labels, volatile, model.on_gpu)
        
        if model.training and iter_counter == 0 :
            optimizer.zero_grad()

        outputs = model(inputs)
        loss_out = loss(outputs, labels)

        if model.training :
            loss_out = loss_out / iter_size
            loss_out.backward()
            iter_counter += 1
            if (iter_counter == iter_size or batch_counter == (len(dataloader)-1)) :
                optimizer.step()
                iter_counter = 0

        logger.log_scalar_direct('loss', loss_out.data.mean())
        logger.log_batch(inputs, outputs, labels)


class AdversarialTrainer :

    def __init__(self, loss_cutoff) :
        self.initialized = False
        self.loss_cutoff = loss_cutoff
        self.mode = 'd'

    def train_discrimintor(self) :

        disc_loss_sum = 0
        disc_loss_count = 0
        for data in self.dataloader :

            #real data
            real_inputs, original_labels = data
            fake_seeds = self.g_model.generator_input(real_inputs, original_labels)

            real_inputs, real_labels = get_variables(
                real_inputs, 
                torch.ones(real_inputs.shape[0]).long(), 
                not self.d_model.training, 
                self.d_model.on_gpu
            )

            if self.d_model.training :
                self.d_optim.zero_grad()

            real_outputs = self.d_model(real_inputs)
            real_loss_out = self.loss(real_outputs, real_labels)

            if self.d_model.training :
                disc_loss_sum += real_loss_out.data.sum() 
                disc_loss_count += real_loss_out.data.shape[0]
                real_loss_out.backward()

            fake_seeds, fake_labels = get_variables(
                fake_seeds, 
                torch.zeros(fake_seeds.shape[0]).long(), 
                not self.g_model.training, 
                self.g_model.on_gpu
            )
            fake_inputs = self.g_model(fake_seeds).detach() #we don't want to train the generator on these so detach and save some time
            fake_outputs = self.d_model(fake_inputs)
            fake_loss_out = self.loss(fake_outputs, fake_labels)

            if self.d_model.training :
                disc_loss_sum += fake_loss_out.data.sum()
                disc_loss_count += real_loss_out.data.shape[0]
                fake_loss_out.backward()
                self.d_optim.step()

            self.logger.log_scalar_direct('d_loss', torch.cat((fake_loss_out.data, real_loss_out.data)).mean())
            self.logger.log_batch(
                torch.cat((real_inputs,fake_inputs)), 
                torch.cat((real_outputs, fake_outputs)), 
                torch.cat((real_labels, fake_labels)), 
                prefix='d_'
            )

        return disc_loss_sum/disc_loss_count


    def train_generator(self) :
        gen_loss_sum = 0
        gen_loss_count = 0

        for data in self.dataloader :

            if self.g_model.training :
                self.g_optim.zero_grad()

            original_input, original_labels = data
            g_seed = self.g_model.generator_input(original_input, original_labels)
            g_seed, g_labels = get_variables(
                g_seed, 
                torch.ones(g_seed.shape[0]).long(),
                not self.g_model.training,
                self.g_model.on_gpu
            )

            g_inputs = self.g_model(g_seed)
            g_output = self.d_model(g_inputs)
            g_loss_out = self.loss(g_output, g_labels)

            if self.g_model.training :
                gen_loss_sum += g_loss_out.data.sum()
                gen_loss_count += g_loss_out.data.shape[0]
                g_loss_out.backward()
                self.g_optim.step()

            self.logger.log_scalar_direct('g_loss', g_loss_out.data.mean())
            self.logger.log_batch(g_seed, g_inputs, g_output, prefix='g_')

        return gen_loss_sum / gen_loss_count

    def __call__ (self, dataloader, models, loss, optimizers, logger) :
        #TODO maybe I should re think trainers instead of doing this?
        if not self.initialized :
            self.dataloader = dataloader
            self.g_model, self.d_model = models
            self.loss = loss
            self.g_optim, self.d_optim = optimizers
            self.logger = logger

        if self.mode == 'd' :
            print('training discriminator')
            disc_loss = self.train_discrimintor()
            if disc_loss < self.loss_cutoff :
                self.mode = 'g'
        elif self.mode == 'g' :
            print('training generator')
            gen_loss = self.train_generator()
            if gen_loss < self.loss_cutoff :
                self.mode = 'd'



