from util_libs import *


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args['bs'])
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args['device']), target.to(args['device'])
        output = net_g(data)
        log_probs = output[1] if isinstance(output, tuple) else output
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    return accuracy.numpy(), test_loss



class LocalUpdate(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay=self.weight_decay)


        prev_net = copy.deepcopy(net)

        batch_loss = []
        step_count = 0

        while(True):
          for batch_idx, (images, labels) in enumerate(self.ldr_train):
              images, labels = images.to(self.args['device']), labels.to(self.args['device'])
              if(self.use_data_augmentation == True):
                images = self.transform_train(images)
              net.zero_grad()
              output = net(images)
              log_probs = output[1] if isinstance(output, tuple) else output
              loss = self.loss_func(log_probs, labels)
              loss.backward()
                
              if(self.use_gradient_clipping ==True):
                torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)
            
              optimizer.step()
              batch_loss.append(loss.item())
              step_count=step_count+1
              if(step_count >= self.args['cp']):
                break
          if(step_count >= self.args['cp']):
            break

        with torch.no_grad():

                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                model_to_return = params_delta_vec
            
        return model_to_return

class LocalUpdate_scaffold(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        

        
    def train_and_sketch(self, net, idx, mem_mat, c):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay = self.weight_decay)


        prev_net = copy.deepcopy(net)
        
        eta = self.lr

        batch_loss = []
        step_count = 0

        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.use_data_augmentation == True):
                  images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)


                state_params_diff = c-mem_mat[idx]
                local_par_list = None
                for param in net.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
                
                loss_algo = torch.sum(local_par_list * state_params_diff)
                loss = loss + loss_algo



                loss.backward()

                if(self.use_gradient_clipping ==True):
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)

                optimizer.step()
                batch_loss.append(loss.item())
                step_count=step_count+1

            
            
                if(step_count >= self.args['cp']):
                    break

            if(step_count >= self.args['cp']):
              break

        with torch.no_grad():


                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                
                mem_mat[idx] = (mem_mat[idx]-c) - params_delta_vec/(step_count*eta)


                model_to_return = params_delta_vec
            
        return model_to_return

class LocalUpdate_fedprox(object):
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        # self.loss_func = nn.MSELoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.mu = args_hyperparameters['mu']
        self.transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),])
        
    def train_and_sketch(self, net):
        net.train()

        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum = 0, weight_decay = self.weight_decay)


        prev_net = copy.deepcopy(net)
        prev_net_vec = parameters_to_vector(prev_net.parameters())
        
        eta = self.lr
        mu = self.mu

        batch_loss = []
        step_count = 0

        while(True):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args['device']), labels.to(self.args['device'])
                if(self.use_data_augmentation == True):
                  images = self.transform_train(images)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)

                local_par_list = parameters_to_vector(net.parameters())
                loss_algo = torch.linalg.norm(local_par_list-prev_net_vec)**2
                loss = loss + mu*0.5*loss_algo



                loss.backward()

                if(self.use_gradient_clipping==True):
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=self.max_norm)

                optimizer.step()
                batch_loss.append(loss.item())
                step_count=step_count+1
                
               
            
            
                if(step_count >= self.args['cp']):
                    break
                    
                    
            if(step_count >= self.args['cp']):
                break

        with torch.no_grad():


                vec_curr = parameters_to_vector(net.parameters())
                vec_prev = parameters_to_vector(prev_net.parameters())
                params_delta_vec = vec_curr-vec_prev
                
                model_to_return = params_delta_vec
            
        return model_to_return

class LocalUpdate_dts(object):
    """
    Base DTS local update.
    Trains backbone + TempNet jointly with temperature-scaled cross-entropy.
    Returns the backbone parameter delta (same interface as LocalUpdate).
    The caller is responsible for passing in and receiving back the TempNet.
    """
    def __init__(self, args, args_hyperparameters, dataset=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.dataset = dataset
        self.ldr_train = DataLoader(dataset, batch_size=self.args['bs'], shuffle=True)
        self.lr = args_hyperparameters['eta_l']
        self.use_data_augmentation = args_hyperparameters['use_augmentation']
        self.use_gradient_clipping = args_hyperparameters['use_gradient_clipping']
        self.max_norm = args_hyperparameters['max_norm']
        self.weight_decay = args_hyperparameters['weight_decay']
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

    def train_and_sketch(self, net, tempnet):
        """
        Args:
            net     : backbone (copy of global model)
            tempnet : TempNet (copy of global TempNet)
        Returns:
            backbone_delta : parameter delta vector for the backbone
            tempnet        : updated TempNet (returned so the server can aggregate)
        """
        net.train()
        tempnet.train()

        backbone_opt = torch.optim.SGD(
            net.parameters(), lr=self.lr, momentum=0,
            weight_decay=self.weight_decay
        )
        # TempNet uses the same local LR; you can tune this separately if needed
        temp_opt = torch.optim.SGD(tempnet.parameters(), lr=self.lr, momentum=0)

        prev_net = copy.deepcopy(net)

        step_count = 0
        while True:
            for images, labels in self.ldr_train:
                images = images.to(self.args['device'])
                labels = labels.to(self.args['device'])

                if self.use_data_augmentation:
                    images = self.transform_train(images)

                backbone_opt.zero_grad()
                temp_opt.zero_grad()

                features, logits = net(images)               # backbone returns (features, logits)
                tau = tempnet(features.detach())              # τ predicted from features; detach to avoid
                                                              # double-backprop through backbone
                scaled_logits = logits / tau
                loss = self.loss_func(scaled_logits, labels)
                loss.backward()

                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_norm)

                backbone_opt.step()
                temp_opt.step()

                step_count += 1
                if step_count >= self.args['cp']:
                    break
            if step_count >= self.args['cp']:
                break

        with torch.no_grad():
            vec_curr = parameters_to_vector(net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            backbone_delta = vec_curr - vec_prev

        return backbone_delta, tempnet

def get_grad(net_glob, args, args_hyperparameters,  dataset, alg, idx, mem_mat, c, tempnet=None):

    if(alg=='fedadam' or alg == 'fedexp' or alg =='fedavg' or alg=='fedavgm' or alg=='fedavgm(exp)' or alg=='fedadam' or alg=='fedadagrad'):

        local = LocalUpdate(args, args_hyperparameters, dataset=dataset)

        grad = local.train_and_sketch(copy.deepcopy(net_glob))

        return grad

    elif(alg=='scaffold' or alg=='scaffold(exp)'):

         local = LocalUpdate_scaffold(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob),idx,mem_mat,c)

         return grad
    
    elif(alg=='fedprox' or alg=='fedprox(exp)'):            
        

         local = LocalUpdate_fedprox(args, args_hyperparameters, dataset=dataset)

         grad = local.train_and_sketch(copy.deepcopy(net_glob))

         return grad
    
    elif alg == 'dts':
        # NOTE: 'tempnet' must be added as an optional keyword argument to get_grad
        local = LocalUpdate_dts(args, args_hyperparameters, dataset=dataset)
        
        grad, updated_tempnet = local.train_and_sketch(copy.deepcopy(net_glob), tempnet)
        
        return grad, updated_tempnet

  
