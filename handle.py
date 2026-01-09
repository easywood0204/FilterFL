import copy
import os
import logging
import random
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets
from torchvision.transforms import transforms
from collections import defaultdict
from models.ResNet8 import ResNet8

from device import device
logger = logging.getLogger("logger")

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Handle:
    def __init__(self, current_time, params, name):

        self.current_time = current_time
        self.model = None

        self.train_dataset = None
        self.test_dataset = None
        self.test_dataset_poisoned = None

        self.train_data = None
        self.clients_data_num = None
        self.test_data = None

        self.poisoned_test_data = None
        self.test_un_target_label_data = None
        self.test_target_label_data = None

        self.participants_list = None
        self.adversarial_namelist = None
        self.benign_namelist = None

        self.start_epoch = 1
        self.classes_dict = None
        self.params = params
        self.name = name
        self.target = 0
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        self.tinydata = []
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
        if not self.params.get('environment_name', False):
            self.params['environment_name'] = self.name
        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path

    def load_data(self):
        logger.info('Loading data')
        dataPath = './data'
        if self.params['type'] == 'CIFAR10':
            # data load
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
            self.train_dataset = datasets.CIFAR10(dataPath, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR10(dataPath, train=False, download=True, transform=transform_test)
            self.test_dataset_poisoned = datasets.CIFAR10(dataPath, train=False, download=True,
                                                          transform=transform_test)
        elif self.params['type'] == 'MNIST':
            transform_train = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.train_dataset = datasets.MNIST(dataPath, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.MNIST(dataPath, train=False, download=True, transform=transform_test)
            self.test_dataset_poisoned = datasets.MNIST(dataPath, train=False, download=True, transform=transform_test)
        elif self.params['type'] == 'CIFAR100':
            transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                       std=[0.267, 0.256, 0.276])])
            transform_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                      std=[0.267, 0.256, 0.276])])
            self.train_dataset = datasets.CIFAR100(dataPath, train=True, download=True, transform=transform_train)
            self.test_dataset = datasets.CIFAR100(dataPath, train=False, download=True, transform=transform_test)
            self.test_dataset_poisoned = datasets.CIFAR100(dataPath, train=False, download=True,
                                                           transform=transform_test)
        elif self.params['type'] == 'GTSRB':
            transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                                                                       std=[0.2672, 0.2564, 0.2629])])
            transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                                                                      std=[0.2672, 0.2564, 0.2629])])
            from models.gtsrb import GTSRB
            self.train_dataset = GTSRB(dataPath, train=True, transform=transform_train)
            self.test_dataset = GTSRB(dataPath, train=False, transform=transform_test)
            self.test_dataset_poisoned = GTSRB(dataPath, train=False, transform=transform_test)

        elif self.params['type'] == 'TinyImagenet':
            transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                                  transforms.RandomCrop(32, padding=4),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                                                                       std=[0.2672, 0.2564, 0.2629])])
            transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.3337, 0.3064, 0.3171],
                                                                      std=[0.2672, 0.2564, 0.2629])])
            self.train_dataset = TinyImageNet(dataPath, train=True, transform=transform_train)
            self.test_dataset = TinyImageNet(dataPath, train=False,transform=transform_train)
            self.test_dataset_poisoned = TinyImageNet(dataPath, train=False, transform=transform_test)
        else:
            logger.info('no this type!!!!!')

        logger.info('reading data done')
        self.classes_dict = self.build_classes_dict()
        logger.info('build_classes_dict done')
        if self.params['sampling_dirichlet']:
            indices_per_participant, dict_users = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in indices_per_participant.items()]
            self.clients_data_num = dict_users
        else:
            num_items = int(len(self.train_dataset) / self.params['number_of_total_participants'])
            dict_users, all_idxs = {}, [i for i in range(len(self.train_dataset))]
            for i in range(self.params['number_of_total_participants']):
                dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - dict_users[i])
            for i in range(self.params['number_of_total_participants']):
                dict_users[i] = np.array(list(dict_users[i])).tolist()
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in dict_users.items()]
            self.clients_data_num = [len(indices) for pos, indices in dict_users.items()]

        logger.info('train loaders done')
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.test_un_target_label_data, self.test_target_label_data = self.poison_test_dataset_ind()
        self.participants_list = list(range(self.params['number_of_total_participants']))
        self.adversarial_namelist = random.sample(self.participants_list,
                                                  self.params['poison_client_num'])
        self.benign_namelist = list(set(self.participants_list) - set(self.adversarial_namelist))

    def build_classes_dict(self):
        classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        return classes

    def sample_dirichlet_train_data(self, num_participants, alpha):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in MNIST/CIFAR training set.
            Requires: classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """
        classes = self.classes_dict
        class_size = len(classes[0])
        per_participant_list = defaultdict(list)
        num_classes = len(classes.keys())
        image_nums = []
        for n in range(num_classes):
            image_num = []
            random.shuffle(classes[n])
            if self.params['type'] == 'GTSRB' or self.params['type'] == 'CIFAR-100':
                class_size = len(classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(num_participants * [alpha]))
            for user in range(num_participants):
                num_imgs = int(round(sampled_probabilities[user]))
                sampled_list = classes[n][:min(len(classes[n]), num_imgs)]
                image_num.append(len(sampled_list))
                per_participant_list[user].extend(sampled_list)
                classes[n] = classes[n][min(len(classes[n]), num_imgs):]
            image_nums.append(image_num)
        # self.draw_dirichlet_plot(num_classes, num_participants, image_nums, alpha)
        dict_users = []
        for i in range(num_participants):
            dict_users.append(len(per_participant_list[i]))
        return per_participant_list, dict_users

    '''
    def draw_dirichlet_plot(self, no_classes, no_participants, image_nums, alpha):
        fig = plt.figure(figsize=(10, 5))
        s = np.empty([no_classes, no_participants])
        for i in range(0, len(image_nums)):
            for j in range(0, len(image_nums[0])):
                s[i][j] = image_nums[i][j]
        s = s.transpose()
        left = 0
        y_labels = []
        category_colors = plt.get_cmap('RdYlGn')(
            np.linspace(0.15, 0.85, no_participants))
        for k in range(no_classes):
            y_labels.append('Label ' + str(k))
        vis_par = [0, 10, 20, 30]
        for k in range(no_participants):
            # for k in vis_par:
            color = category_colors[k]
            plt.barh(y_labels, s[k], left=left, label=str(k), color=color)
            widths = s[k]
            xcenters = left + widths / 2
            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            # for y, (x, c) in enumerate(zip(xcenters, widths)):
            #     plt.text(x, y, str(int(c)), ha='center', va='center',
            #              color=text_color,fontsize='small')
            left += s[k]
        plt.legend(ncol=20, loc='lower left', bbox_to_anchor=(0, 1), fontsize=4)  #
        # plt.legend(ncol=len(vis_par), bbox_to_anchor=(0, 1),
        #            loc='lower left', fontsize='small')
        plt.xlabel("Number of Images", fontsize=16)
        # plt.ylabel("Label 0 ~ 199", fontsize=16)
        # plt.yticks([])
        fig.tight_layout(pad=0.1)
        # plt.ylabel("Label",fontsize='small')
        fig.savefig(self.folder_path + '/Num_Img_Dirichlet_Alpha{}.pdf'.format(alpha))
    '''

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                                   pin_memory=True, num_workers=0)
        return train_loader

    def get_test(self):
        test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                  batch_size=self.params['batch_size'],
                                                  shuffle=False)
        return test_loader

    def poison_test_dataset_ind(self):
        logger.info('get poison test loader')
        # delete the test data with target label
        test_classes = {}
        for ind, x in enumerate(self.test_dataset_poisoned):
            _, label = x
            if label in test_classes:
                test_classes[label].append(ind)
            else:
                test_classes[label] = [ind]

        range_num_id = list(range(0, len(self.test_dataset_poisoned)))
        for image_ind in test_classes[self.target]:
            if image_ind in range_num_id:
                range_num_id.remove(image_ind)
        poison_label_inds = test_classes[self.target]
        # self.poison_test_data(self.test_dataset_poisoned)
        return (torch.utils.data.DataLoader(self.test_dataset_poisoned,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(range_num_id)),
                torch.utils.data.DataLoader(self.test_dataset_poisoned,
                                            batch_size=self.params['batch_size'],
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(poison_label_inds)))

    def random_get_clients(self, epoch):
        adversarial_name_keys = []
        if self.params['is_poison'] is True:
            if self.params['poison_type'] == 'single_attack':
                for idx in range(0, len(self.params['adversary_list'])):
                    if epoch in self.params[str(idx) + '_poison_epochs']:
                        if self.params['adversary_list'][idx] not in adversarial_name_keys:
                            adversarial_name_keys.append(self.params['adversary_list'][idx])
                non_attacker = []
                for adv in self.params['adversary_list']:
                    if adv not in adversarial_name_keys:
                        non_attacker.append(copy.deepcopy(adv))
                benign_num = self.params['num_clients'] - len(adversarial_name_keys)
                random_agent_name_keys = random.sample(self.benign_namelist + non_attacker, benign_num)
                agent_name_keys = adversarial_name_keys + random_agent_name_keys
            else:
                # handle.params['poison_type'] == 'multiple_attack':
                if epoch <= self.params['start_poison']:
                    agent_name_keys = random.sample(self.benign_namelist, self.params['num_clients'])
                else:
                    agent_name_keys = random.sample(self.participants_list, self.params['num_clients'])
                for client in agent_name_keys:
                    if client in self.adversarial_namelist:
                        adversarial_name_keys.append(client)
        else:
            agent_name_keys = random.sample(self.benign_namelist, self.params['num_clients'])
        # logger.info(f'Server Epoch:{epoch} choose agents : {agent_name_keys}.')
        return agent_name_keys, adversarial_name_keys

    def create_model(self):
        model = None
        if self.params['type'] == 'CIFAR10':
            model = ResNet8()
        elif self.params['type'] == 'MNIST':
            model = CNNMnist()
        elif self.params['type'] == 'CIFAR100':
            model = ResNet18_cifar10(num_classes=100)
        elif self.params['type'] == 'GTSRB':
            model = ResNet8(num_classes=43)
        elif self.params['type'] == 'TinyImagenet':
            model = ResNet18_cifar10(num_classes=200)
        model = model.to(device)
        self.model = model
