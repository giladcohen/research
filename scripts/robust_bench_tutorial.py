from robustbench.data import load_cifar10
from robustbench.utils import load_model
import foolbox as fb

x_test, y_test = load_cifar10(n_examples=50, data_dir='/data/dataset/cifar10')
model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10', threat_model='Linf',
                   model_dir='/data/models')
fmodel = fb.PyTorchModel(model, bounds=(0, 1))
_, advs, success = fb.attacks.LinfPGD()(fmodel, x_test.to('cuda:0'), y_test.to('cuda:0'), epsilons=[8/255])
print('Robust accuracy: {:.1%}'.format(1 - success.float().mean()))




from autoattack import AutoAttack
model = model.cuda()
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test.to('cuda:0'), y_test.to('cuda:0'))




from robustbench.data import load_cifar10c
from robustbench.utils import clean_accuracy
corruptions = ['fog']
x_test, y_test = load_cifar10c(n_examples=1000, corruptions=corruptions, severity=5, data_dir='/data/dataset/cifar10c')
for model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting',
                   'Carmon2019Unlabeled']:
    model = load_model(model_name, dataset='cifar10', threat_model='Linf', model_dir='/data/models')
    acc = clean_accuracy(model, x_test, y_test)
    print(f'Model: {model_name}, CIFAR-10-C accuracy: {acc:.1%}')

