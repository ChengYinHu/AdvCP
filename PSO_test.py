import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from Color_Projection_Simulation import classify, initiation, function

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = models.resnet50(pretrained=True).eval().to(device)


tag_save = 1
tag_35 = np.zeros((1, 3))
if __name__ == "__main__":

    Omega, c1, r1, c2, r2 = 0.9, 1.6, 0.5, 1.4, 0.5

    seed = 20  # population
    step = 10

    Hyper = 12

    population = np.zeros((seed, Hyper))
    unit = np.zeros((1, Hyper))
    population = initiation(population)
    conf_35 = np.zeros((1, seed))
    label_adv = np.zeros((1, seed))

    P_best = np.zeros((seed, Hyper))
    conf_P = np.ones((1, seed)) * 100
    G_best = np.zeros((1, Hyper))
    conf_G = 100
    V = np.zeros((seed, Hyper))

    print('population = ', population)

    dir_read = "35.jpg"  # Image path

    tag_break = 0
    for steps in range(0, step):
         for seeds in range(0, seed):
             tag_break = 0

             for i in range(0, Hyper):
                 unit[0][i] = population[seeds][i]
                 # print('unit[0][j] = ', unit[0][i])

             print('steps, seeds,  tag_35= ', steps, seeds, tag_35)
             label_adv[0][seeds], conf_35[0][seeds], tag_break = function(dir_read, net, unit, tag_break)
             print('label_adv[0][i], conf_35[0][i]', label_adv[0][seeds], conf_35[0][seeds])
             # print('unit = ', unit)

             if tag_break == 1:
                 img_save = plt.imread('adv.jpg')

                 tag_35[0][0] = tag_35[0][0] + 1
                 tag_35[0][1], tag_35[0][2] = steps, seeds
                 # print('tag_35 = ', tag_35)
                 name_save = 'result/' + str(tag_save) + '.jpg'

                 plt.imsave(name_save, img_save)
                 tag_save = tag_save + 1

                 if tag_35[0][0] == 20:
                     exit()


             if conf_G > conf_35[0][seeds]:
                 for i in range(0, Hyper):
                     G_best[0][i] = population[seeds][i]
                 conf_G = conf_35[0][seeds]

             if conf_P[0][seeds] > conf_35[0][seeds]:
                 for i in range(0, Hyper):
                     P_best[seeds][i] = population[seeds][i]
                 conf_P[0][seeds] = conf_35[0][seeds]


         for seeds in range(0, seed):
             for i in range(0, Hyper):
                 # print('V = ', V)
                 # print('P_best = ', P_best)
                 # print('population = ', population)
                 # print('G_best = ', G_best)
                 V[seeds][i] = Omega*V[seeds][i] + c1*r1*(P_best[seeds][i]-population[seeds][i]) + c2*r2*(G_best[0][i]-population[seeds][i])
                 population[seeds][i] = population[seeds][i] + int(V[seeds][i])

         # print('population = ', population)





    # print('G_best = ', G_best)
    # print('conf_G = ', conf_G)
    # print('P_best = ', P_best)
    # print('conf_P = ', conf_P)





