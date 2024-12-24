# -*- coding: cp1251 -*-
# -*- coding: utf-8 -*-
# '''
# Модулированный гауссов импульс.

# Область моделирования - свободное пространство.
# '''
from json import tool
import numpy as np
import tools
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift

class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return round(x / self.discrete)




class GaussianModPlaneWave:
    # ''' Класс с уравнением плоской волны для модулированного гауссова сигнала в дискретном виде
    # d - определяет задержку сигнала.
    # w - определяет ширину сигнала.
    # Nl - количество ячеек на длину волны.
    # Sc - число Куранта.
    # eps - относительная диэлектрическая проницаемость среды, в которой расположен источник.
    # mu - относительная магнитная проницаемость среды, в которой расположен источник.
    # '''

    def __init__(self, d, w, Nl, Sc=1.0, eps=1.0, mu=1.0):
        self.d = d
        self.w = w
        self.Nl = Nl
        self.Sc = Sc
        self.eps = eps
        self.mu = mu

    def getE(self, m, q):
        # '''
        # Расчет поля E в дискретной точке пространства m
        # в дискретный момент времени q
        # '''
        return (np.sin(2 * np.pi / self.Nl * (q * self.Sc - m * np.sqrt(self.eps * self.mu))) *
                np.exp(-(((q - m * np.sqrt(self.eps * self.mu) / self.Sc) - self.d) / self.w) ** 2))



if __name__ == '__main__':
    
    c = 3e8

    # Волновое сопротивление свободного пространства
    eps_r = 9
    W0 = 120.0 * np.pi
    
    # Расчет длинны волны через частоты сиганал
    f_min = 0.5e9
    f_max = 3.5e9
    f_mid = (f_max+f_min)/2
    lamda = c / f_mid
                   
    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в отсчетах
    maxSize_m = 4
    dx = 5e-3
    dt = dx/c
    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)
    PecX = maxSize - 1

    # Положение источника в отсчетах
    sourcePos_m = 1.5
    sourcePos = sampler_x.sample(sourcePos_m)
    
    # Время расчета в отсчетах
    maxTime_s = 60e-9
    maxTime = sampler_t.sample(maxTime_s)
    # Параметры сигнала
    dg = 250
    wg = 35
    Nl = 30
    
    # Датчики для регистрации поля
    probepos_m = 2.5
    probesPos = [sampler_x.sample(probepos_m)]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    
    eps = np.ones(maxSize)
    eps[:]=eps_r
    
    # Магнитная проницаемость
    mu = np.ones(maxSize - 1)

    Ez = np.zeros(maxSize)
    Hy = np.zeros(maxSize - 1)
    
    source = GaussianModPlaneWave (dg, wg, Nl, Sc, eps[sourcePos], mu[sourcePos])
    
     
    # Коэффициенты для расчета ABC второй степени
    
    # Sc' для правой границы
    Sc1Right = Sc / np.sqrt(mu[-1] * eps[-1])

    k1Right = -1 / (1 / Sc1Right + 2 + Sc1Right)
    k2Right = 1 / Sc1Right - 2 + Sc1Right
    k3Right = 2 * (Sc1Right - 1 / Sc1Right)
    k4Right = 4 * (1 / Sc1Right + Sc1Right)

    # Ez[0: 2] в предыдущий момент времени (q)
    oldEzLeft1 = np.zeros(3)

    # Ez[0: 2] в пред-предыдущий момент времени (q - 1)
    oldEzLeft2 = np.zeros(3)

    # Ez[-3: -1] в предыдущий момент времени (q)
    oldEzRight1 = np.zeros(3)

    # Ez[-3: -1] в пред-предыдущий момент времени (q - 1)
    oldEzRight2 = np.zeros(3)

    


  

    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -2.1
    display_ymax = 2.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(dx, dt, maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= Sc / (W0 * mu[sourcePos - 1]) * source.getE(0, q)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1:-1] + (Hy[1:] - Hy[:-1]) * Sc * W0 / eps[1:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += (Sc / (np.sqrt(eps[sourcePos] * mu[sourcePos])) *
                          source.getE(-0.5, q + 0.5))


        #Лево
        
        Ez[1] = Ez[2]
        Ez[0] = 0
        


        # Граничные условия ABC второй степени (справа)
        Ez[-1] = (k1Right * (k2Right * (Ez[-3] + oldEzRight2[-1]) +
                             k3Right * (oldEzRight1[-1] + oldEzRight1[-3] - Ez[-2] - oldEzRight2[-2]) -
                             k4Right * oldEzRight1[-2]) - oldEzRight2[-3])

        oldEzRight2[:] = oldEzRight1[:]
        oldEzRight1[:] = Ez[-3:]
        
        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 50 == 0:
            display.updateData(display_field, q)
    display.stop()
    
    tools.showProbeSpectrum(probes, dx, dt, -5, 5)
