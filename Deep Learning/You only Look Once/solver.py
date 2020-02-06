from model import Yolo
from torch.autograd import Variable
from torchvision.utils import save_image
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import datetime
import math

class Solver(object):
    """Solver for training and testing StarGAN."""

    def __init__(self, voc_loader, config):
        """Initialize configurations."""

        # Data loader.
        self.voc_loader = voc_loader
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        # Model configurations.
        self.c_dim = config.c_dim
        self.c2_dim = config.c2_dim
        self.image_size = config.image_size
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.g_repeat_num = config.g_repeat_num
        self.d_repeat_num = config.d_repeat_num
        self.lambda_cls = config.lambda_cls
        self.lambda_rec = config.lambda_rec
        self.lambda_gp = config.lambda_gp

        # Training configurations.
        self.dataset = config.dataset
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        self.num_iters_decay = config.num_iters_decay
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.n_critic = config.n_critic
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.resume_iters = config.resume_iters
        self.selected_attrs = config.selected_attrs

        # Test configurations.
        self.test_iters = config.test_iters

        # Miscellaneous.
        self.use_tensorboard = config.use_tensorboard
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Directories.
        self.log_dir = config.log_dir
        self.sample_dir = config.sample_dir
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.lr_update_step = config.lr_update_step

        # Build the model and tensorboard.
        self.build_model()
        if self.use_tensorboard:
            self.build_tensorboard()

    def build_model(self):
        """Create a generator and a discriminator."""
        self.Yolo = Yolo(3, 64)

        self.Yolo_optimizer = torch.optim.SGD(self.Yolo.parameters(), lr=self.g_lr, momentum=0.9, weight_decay=0.0005)
        self.print_network(self.Yolo, 'Yolo')
       
        self.Yolo.to(self.device)
    
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_iters):
        """Restore the trained generator and discriminator."""
        print('Loading the trained models from step {}...'.format(resume_iters))
        G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(resume_iters))
        self.Yolo.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))

    def build_tensorboard(self):
        """Build a tensorboard logger."""
        from logger import CsvLogger
        self.logger = CsvLogger(self.log_dir)

    def update_lr(self, g_lr):
        """Decay learning rates of the generator and discriminator."""
        for param_group in self.Yolo_optimizer.param_groups:
            param_group['lr'] = g_lr

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.Yolo_optimizer.zero_grad()

    def denorm(self, x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def create_labels(self, c_org, c_dim=5, dataset='CelebA', selected_attrs=None):
        """Generate target domain labels for debugging and testing."""
        # Get hair color indices.
        if dataset == 'CelebA':
            hair_color_indices = []
            for i, attr_name in enumerate(selected_attrs):
                if attr_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    hair_color_indices.append(i)

        c_trg_list = []
        for i in range(c_dim):
            if dataset == 'CelebA':
                c_trg = c_org.clone()
                if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                    c_trg[:, i] = 1
                    for j in hair_color_indices:
                        if j != i:
                            c_trg[:, j] = 0
                else:
                    c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
            elif dataset == 'RaFD':
                c_trg = self.label2onehot(torch.ones(c_org.size(0))*i, c_dim)

            c_trg_list.append(c_trg.to(self.device))
        return c_trg_list

    def classification_loss(self, logit, target, dataset='CelebA'):
        """Compute binary or softmax cross entropy loss."""
        if dataset == 'CelebA':
            return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)
        elif dataset == 'RaFD':
            return F.cross_entropy(logit, target)

    def getGroundTruthVec(self, x_min, x_max, y_min, y_max, W, H, class_name):
        """Get the YOLO ground truth vector"""
        b_corner = [x_min/W, x_max/W, y_min/H, y_max/H]
        b_center = []
        x_c = ((x_min + x_max)/2)/W
        y_c = ((y_min + y_max)/2)/H
        w_c = (x_max - x_min)/W
        h_c = (y_max - y_min)/H
        b_center = [x_c, y_c, W_c, h_c]

        b_yolo = []
        g_x = math.floor(7*x_c)
        g_y = math.floor(7*y_c)
        b_yolo = [7*x_c - g_x, 7*y_c - g_y, math.sqrt(w_c), math.sqrt(h_c)]

        p = [0]*20
        p[self.classes.index(class_name)] = 1

        return b_yolo, p, (g_x, g_y)

    def IOU(self, b, b1):
        """Calculate IOU between two rectangles
           b: ground truth b_yolo
           b1: predicted bounding box coordinates (center format)"""
        b_x = b[0]
        b_y = b[1]
        b_w = b[2]
        b_h = y[3]
        b_xmin = (2*b_x - b_w)/2
        b_xmax = (2*b_x + b_w)/2
        b_ymin = (2*b_y - b_h)/2
        b_ymax = (2*b_y + b_h)/2

        b1_x = b1[0]
        b1_y = b1[1]
        b1_w = b1[2]
        b1_h = b1[3]
        b1_xmin = (2*b1_x - b1_w)/2
        b1_xmax = (2*b1_x + b1_w)/2
        b1_ymin = (2*b1_y - b1_h)/2
        b1_ymax = (2*b1_y + b1_h)/2

        int_w = max(0, min(b_xmax, b1_xmax) - max(b_xmin, b1_xmin))
        int_h = max(0, min(b_ymax, b1_ymax) - max(b_ymin, b1_ymin))
        intersect = int_w * int_h
        union = b_w*b_h + b1_w*b1_h - intersect
        iou = intersect/union

        return iou
 
    def train(self):
        """Train StarGAN within a single dataset."""
        # Set data loader.
        data_loader = self.voc_loader
        lambda_noobj = 0.5
        lambda_coord = 5
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)
        image, labels = next(data_iter)
        #image = image.to(self.device)
        
        # Learning rate cache for decaying.
        g_lr = self.g_lr

        # Start training from scratch or resume training.
        start_iters = 0
        if self.resume_iters:
            start_iters = self.resume_iters
            self.restore_model(self.resume_iters)

        # Start training.
        print('Start training...')
        start_time = time.time()
        for i in range(1,136):
            for j in range(0, 11530):

                # =================================================================================== #
                #                             1. Preprocess input data                                #
                # =================================================================================== #

                # Fetch real images and labels.
                try:
                    image, label = next(data_iter)
                except:
                    data_iter = iter(data_loader)
                    image, label = next(data_iter)

                label = label[0]['object']
                imgSize = label[0]['size']
                orginal_b = []
                org_p = []
                obj_centers = []
                center_bbox = dict()
                b = []
                p = []
                w = imgSize['width']
                h = imgSize['height']
                for i in range(len(label)):
                    x_max = label[i]['bndbox']['xmax']
                    print(x_max)
                    x_min = label[i]['bndbox']['xmin']
                    y_min = label[i]['bndbox']['ymin']
                    y_max = label[i]['bndbox']['ymax']
                    class_name = label[i]['name']
                    b, p, objc = getGroundTruthVec(x_min, x_max, y_min, y_max, w, h, class_name)
                    if(i==0):
                        original_b = b
                        org_p = p
                        obj_centers = [objc]
                        center_bbox[objc] = (b[0],b[1],b[2],b[3],i)
                    else:
                        original_b.extend(b)
                        org_p.extend(p)
                        obj_centers.extend([objc])
                        center_bbox[objc] = (b[0],b[1],b[2],b[3],i)
                        
                image = image.to(self.device)           
                orignal_b = original_b.to(self.device)             
                org_p = org_p.to(self.device)
                obj_centers = obj_centers.to(self.device)
                center_bbox = center_bbox.to(self.device)
                # =================================================================================== #
                #                             2. Train the discriminator                              #
                # =================================================================================== #

                # Compute loss with real images.
                out = self.Yolo(image)
                className = out[0:979]
                c1c2 = out[980:1077]
                boxDims = out[1078:1469]
                outArray = np.zeros(7,7,30)
                for j in range(0,7):
                    for i in range(0,7):
                        outArray[i,j,0:19] = className[i*20*7+j*20:i*20*7+j*20+19]
                        outArray[i,j,20:21] = c1c2[i*2*7+j*2:i*2*7+j*2+1]
                        outArray[i,j,22:29] = boxDims[i*8*7+j*8:i*8*7+j*8+7] #x,y,w,h

                #Calculating the losses
                noObj_loss = 0
                xy_loss = 0
                wh_loss = 0
                c_loss = 0
                classprob_loss = 0
                for i in range(0,7):
                    for j in range(0,7):
                        if (i,j) in obj_centers: #object exists
                            gt = center_bbox[(i,j)]
                            b = [gt[0],gt[1],gt[2],gt[3]]
                            b1 = outArray[i,j,22:25]
                            b2 = outArray[i,j,26:29]
                            iou_b1 = IOU(b,b1)
                            iou_b2 = IOU(b,b2)
                            if(iou_b1 > iou_b2):
                                xy_loss = xy_loss + (math.pow((gt[0]-outArray[i,j,22]),2) + math.pow((gt[1]-outArray[i,j,23]),2))
                                wh_loss = wh_loss + (math.pow((gt[2]-outArray[i,j,24]),2) + math.pow((gt[3]-outArray[i,j,25]),2))
                                c_loss = c_loss + (math.pow((iou_b1-outArray[i,j,20]),2))
                                classprob_loss = classprob_loss + np.sum(math.pow((np.array(org_p[gt[4]*20:gt[4]*20+19])-outArray[i,j,0:19]),2))
                            else:
                                xy_loss = xy_loss + (math.pow((gt[0]-outArray[i,j,26]),2) + math.pow((gt[1]-outArray[i,j,27]),2))
                                wh_loss = wh_loss + (math.pow((gt[2]-outArray[i,j,28]),2) + math.pow((gt[3]-outArray[i,j,29]),2))
                                c_loss = c_loss + (math.pow((iou_b2-outArray[i,j,21]),2))
                                classprob_loss = classprob_loss + np.sum(math.pow((np.array(org_p[gt[4]*20:gt[4]*20+19])-outArray[i,j,0:19]),2))
                        else:  # No object
                            noObj_loss = noObj_loss + (math.pow(outArray[i,j,20],2) + math.pow(outArray[i,j,21],2))

                # Backward and optimize.
                loss = lambda_coord*xy_loss + lambda_coord*wh_loss +c_loss + lambda_noobj*noObj_loss + classprob_loss
                self.reset_grad()
                loss.backward()
                self.Yolo_optimizer.step()

                # Logging.
                loss = {}
                loss['xy_loss'] = xy_loss.item()
                loss['wh_loss'] = wh_loss.item()
                loss['c_loss'] = c_loss.item()
                loss['classProb_loss'] = classprob_loss.item()
                loss['noObj_loss'] = noObj_loss.item()
                

                # =================================================================================== #
                #                                 4. Miscellaneous                                    #
                # =================================================================================== #

                # Print out training information.
                if (i+1) % self.log_step == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                    for tag, value in loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)

                    if self.use_tensorboard:
                        for tag, value in loss.items():
                            self.CsvLogger.scalar_summary(tag, value, i+1)

                # Save model checkpoints.
                if (i+1) % self.model_save_step == 0:
                    G_path = os.path.join(self.model_save_dir, '{}-G.ckpt'.format(i+1))
                    torch.save(self.Yolo.state_dict(), G_path)
                    print('Saved model checkpoints into {}...'.format(self.model_save_dir))

                # Decay learning rates.
                if i > 75 and i < 105:
                    g_lr = 0.001
                    self.update_lr(g_lr)
                    print ('Decayed learning rates, g_lr: {}.'.format(g_lr))
                if i > 105:
                    g_lr = 0.0001
                    self.update_lr(g_lr)
                    print ('Decayed learning rates, g_lr: {}.'.format(g_lr))

    
    def test(self):
        """Translate images using StarGAN trained on a single dataset."""
        # Load the trained generator.
        self.restore_model(self.test_iters)
        
        # Set data loader.
        if self.dataset == 'CelebA':
            data_loader = self.celeba_loader
        elif self.dataset == 'RaFD':
            data_loader = self.rafd_loader
        
        with torch.no_grad():
            for i, (x_real, c_org) in enumerate(data_loader):

                # Prepare input images and target domain labels.
                x_real = x_real.to(self.device)
                c_trg_list = self.create_labels(c_org, self.c_dim, self.dataset, self.selected_attrs)

                # Translate images.
                x_fake_list = [x_real]
                for c_trg in c_trg_list:
                    x_fake_list.append(self.G(x_real, c_trg))

                # Save the translated images.
                x_concat = torch.cat(x_fake_list, dim=3)
                result_path = os.path.join(self.result_dir, '{}-images.jpg'.format(i+1))
                save_image(self.denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
                print('Saved real and fake images into {}...'.format(result_path))
