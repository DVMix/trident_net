#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !apt-get update 
# !pip3 install pybullet
# !pip3 install tensorflow
# !apt-get install -y ffmpeg
# https://colab.research.google.com/drive/1GE2hlm-swF6M51XKMOCDnPnCLdj1mWbb#scrollTo=WBk03tTBoIN8


# In[2]:


import warnings
warnings.filterwarnings('ignore')

import random, numpy, math, time
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import pybullet as pb
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.optimizers import RMSprop


# In[3]:


####################
# Окружение (игра)
####################

MAX_STEPS = 1000        # максимальное количество шагов симуляции
STEPS_AFTER_TARGET = 30 # количество шагов симуляции после достижения цели
TARGET_DELTA = 0.2      # величина приемлемого качения возле цели (абсолютное значение)
FORCE_DELTA = 0.1       # шаг измениния силы (абсолютное значение)
PB_BallMass = 1         # масса шара
PB_BallRadius = 0.2     # радиус шара
PB_HEIGHT = 10          # максимальная высота поднятия шара
MAX_FORCE = 20          # максимальная вертикальная сила пиложенная к шару
MIN_FORCE = 0           # минимальнпая сила пиложенная к шару
MAX_VEL = 14.2          # максимальная вертикальная скорость шара
MIN_VEL = -14.2         # минимальная вертикальная скорость шара


# In[4]:


class Environment:
    def __init__(self):
        # текущее состояние окружения
        self.pb_z = 0        # текущая высота шара
        self.pb_force = 0    # текущая сила приложенная к шару
        self.pb_velocity = 0 # текущая вертикальная скорость шара
        self.z_target = 0    # целевая высота
        self.start_time = 0  # время начала новой игры
        self.steps = 0       # количество шагов после начала симуляции
        self.target_area = 0        # факт достежения цели
        self.steps_after_target = 0 # количество шагов после достежения цели

        # создадим симуляцию
        self.pb_physicsClient = pb.connect(pb.DIRECT)

    def reset(self):
        # случайные высота шара и целевая высота
        z_target = random.uniform(0.01, 0.99)
        self.z_target = PB_BallRadius + z_target*PB_HEIGHT
        z = random.uniform(0.05, 0.95)
        self.pb_z = PB_BallRadius + z*PB_HEIGHT
        
        # сброс параметров окружения
        pb.resetSimulation()
        self.target_area = 0
        self.start_time = time.time()
        self.steps = 0
        self.steps_after_target = 0
        
        # шаг симуляции 1/60 сек.
        pb.setTimeStep(1./60)
        
        # поверхность
        floorColShape = pb.createCollisionShape(pb.GEOM_PLANE)
        # для GEOM_PLANE, visualShape - не отображается, будем использовать GEOM_BOX
        floorVisualShapeId = pb.createVisualShape(pb.GEOM_BOX,halfExtents=[100,100,0.0001], 
                                                  rgbaColor=[1,1,.98,1])
        self.pb_floorId = pb.createMultiBody(0,floorColShape,floorVisualShapeId, [0,0,0], 
                                             [0,0,0,1])# (mass,collisionShape,visualShape)
        
        # шар
        ballPosition = [0,0,self.pb_z]
        ballOrientation=[0,0,0,1]
        ballColShape = pb.createCollisionShape(pb.GEOM_SPHERE,radius=PB_BallRadius)
        ballVisualShapeId = pb.createVisualShape(pb.GEOM_SPHERE,radius=PB_BallRadius, 
                                                 rgbaColor=[0.25, 0.75, 0.25,1])
        self.pb_ballId = pb.createMultiBody(PB_BallMass, ballColShape, ballVisualShapeId, 
                                            ballPosition, ballOrientation) 
        #(mass, collisionShape, visualShape, ballPosition, ballOrientation)
        #pb.changeVisualShape(self.pb_ballId,-1,rgbaColor=[1,0.27,0,1])
        
        # указатель цели (без CollisionShape, только отображение(VisualShape))
        targetPosition = [0,0,self.z_target]
        targetOrientation=[0,0,0,1]
        targetVisualShapeId = pb.createVisualShape(pb.GEOM_BOX,halfExtents=[1,0.025,0.025], 
                                                   rgbaColor=[0,0,0,1])
        self.pb_targetId = pb.createMultiBody(0,-1, targetVisualShapeId, targetPosition, targetOrientation)

        # гравитация
        pb.setGravity(0,0,-10)

        # ограничим движение шара только по вертикальной оси
        pb.createConstraint(self.pb_floorId, -1, self.pb_ballId, -1, pb.JOINT_PRISMATIC, [0,0,1], 
                            [0,0,0], [0,0,0])

        # установим действующую силу на шар, чтобы компенсировать гравитацию
        self.pb_force = 10 * PB_BallMass
        pb.applyExternalForce(self.pb_ballId, -1, [0,0,self.pb_force], [0,0,0], pb.LINK_FRAME)
                
        # return values
        observation = self.getObservation()
        reward, done = self.getReward()
        info = self.getInfo()
        return [observation, reward, done, info]

    # Наблюдения (возвращаются нормализованными)
    def getObservation(self):
        # расстояние до цели
        d_target =  0.5 + (self.pb_z - self.z_target)/(2*PB_HEIGHT)
        # действующая сила
        force = (self.pb_force-MIN_FORCE)/(MAX_FORCE-MIN_FORCE)
        # текущая высота шара
        z = (self.pb_z-PB_BallRadius)/PB_HEIGHT
        # текущая скорость
        z_velocity = (self.pb_velocity-MIN_VEL)/(MAX_VEL-MIN_VEL)
        state = [d_target, force, z_velocity]
        return state

    # вычисление награды за действие
    def getReward(self):
        done = False
        z_reward = 0
        # Факт достижения цели, после чего ждем STEPS_AFTER_TARGET шагов и завершем игру.
        if (TARGET_DELTA >= math.fabs(self.z_target - self.pb_z)):
            self.target_area = 1
            z_reward = 1
        # Выход за пределы зоны
        if (self.pb_z > (PB_HEIGHT + PB_BallRadius) or self.pb_z < PB_BallRadius):
            done = True
        # Завершение игры после достижения цели
        if (self.target_area > 0):
            self.steps_after_target += 1
            if (self.steps_after_target>=STEPS_AFTER_TARGET):
                done = True
        # Завершение игры по таймауту
        if (self.steps >= MAX_STEPS):
            done = True

        return [z_reward, done]
    
    # Дополнительная информация для сбора статистики
    def getInfo(self):
        game_time = time.time() - self.start_time
        if game_time:
            fps = round(self.steps/game_time)
        return {'step': self.steps, 'fps': fps}

    # Запуск шага симуляции согласно переданному действию
    def step(self, action):
        self.steps += 1
        if action == 0:
            # 0 - увеличение приложеной силы
            self.pb_force -= FORCE_DELTA
            if self.pb_force < MIN_FORCE:
                self.pb_force = MIN_FORCE
        else:
            # 1 - уменьшение приложенной силы
            self.pb_force += FORCE_DELTA
            if self.pb_force > MAX_FORCE:
                self.pb_force = MAX_FORCE
        
        # изменим текущую сил и запустим шаг симуляции
        pb.applyExternalForce(self.pb_ballId, -1, [0,0,self.pb_force], [0,0,0], pb.LINK_FRAME)
        pb.stepSimulation()
        
        # обновим парамтры состояния окружения (положение и скорость шара)
        curPos, curOrient = pb.getBasePositionAndOrientation(self.pb_ballId)
        lin_vel, ang_vel= pb.getBaseVelocity(self.pb_ballId)
        self.pb_z = curPos[2]
        self.pb_velocity = lin_vel[2]
        
        # вернем наблюдения, награду, факт окончания игры и доп.информацию
        observation = self.getObservation()
        reward, done = self.getReward()
        info = self.getInfo()
        return [observation, reward, done, info]
    
    # Текущее изображение с камеры
    def render(self):
        camTargetPos = [0,0,5] # расположение цели (фокуса) камеры
        camDistance = 10       # дистанция камеры от цели
        yaw = 0                # угол рыскания относительно цели
        pitch = 0              # наклон камеры относительно цели
        roll=0                 # угол крена камеры относительно цели
        upAxisIndex = 2        # ось вертикали камеры (z)

        fov = 60               # угол зрения камеры
        nearPlane = 0.01       # расстояние до ближней плоскости отсечения
        farPlane = 20          # расстояние до дальной плоскости отсечения
        pixelWidth = 320       # ширина изображения
        pixelHeight = 200      # высота изображения
        aspect = pixelWidth/pixelHeight;  # соотношение сторон изображения
       
        # видовая матрица
        viewMatrix = pb.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, 
                                                          upAxisIndex)
        # проекционная матрица
        projectionMatrix = pb.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        # рендеринг изображения с камеры
        img_arr = pb.getCameraImage(pixelWidth, pixelHeight, viewMatrix, projectionMatrix, shadow=0, 
                                    lightDirection=[0,1,1],renderer=pb.ER_TINY_RENDERER)
        w=img_arr[0] #width of the image, in pixels
        h=img_arr[1] #height of the image, in pixels
        rgb=img_arr[2] #color data RGB
        dep=img_arr[3] #depth data
        
        # вернем rgb матрицу
        return rgb


# In[5]:


#################################
# Память для обучающих примеров
#################################

MEMORY_CAPACITY = 200000
class Memory:
    def __init__(self):
        self.samples = []   # хранятся кортежи типа ( s, a, r, s_ )

    def add(self, sample):
        self.samples.append(sample)
        if len(self.samples) > MEMORY_CAPACITY:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)


# In[6]:


##################
# Нейронная сеть
##################

LAYER_SIZE = 512       # размер слоя
STATE_CNT  = 3         # количество входных пераметров (расстояние до цели + действующая сила + скорость)
ACTION_CNT = 2         # количесво выходов (награда за уменьшение и увеличение силы)
class Brain:
    def __init__(self):
        self.model = self._QNetwork()
        
    def _QNetwork(self):
        # Создадим сеть используя Keras
        model = Sequential()
        model.add(Dense(units=LAYER_SIZE, activation='relu', input_dim=STATE_CNT))
        model.add(Dense(units=LAYER_SIZE, activation='relu'))
        model.add(Dense(units=ACTION_CNT, activation='linear'))
        opt = RMSprop(lr=0.00025)
        model.compile(loss='mse', optimizer=opt)
        return model
    
    # обучение по одному пакету обучающих примеров 
    def train(self, x, y, batch_size=32, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=verbose)

    # предсказания сети по списку начальных состояний
    def predict(self, s):
        return self.model.predict(s)

    #  предсказания сети по одном начальному состоянию
    def predictOne(self, s):
        s = numpy.array(s)
        predictions = self.predict(s.reshape(1, STATE_CNT)).flatten()
        return predictions


# In[7]:


###############
# Агент
###############

GAMMA = 0.98        # фактор дисконтирования
MAX_EPSILON = 0.5   # максимальная вероятность выбора случайного действия
MIN_EPSILON = 0.1   # минимальная вероятность выбора случайного действия
LAMBDA = 0.001      # параметр определяющий скорость уменьшения вероятности выбора случайного действия
BATCH_SIZE = 32     # размер обучающего пакета

class Agent:
    def __init__(self):
        self.brain = Brain()                     # Нейронная сеть для обучения
        self.memory = Memory()                   # Хранилище обучающих примеров
        self.epsilon = MAX_EPSILON               # Определяет вероятность выбора случайного действия

    # выбор действия
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_CNT - 1)        # выбираем случайное действие
        else:
            return numpy.argmax(self.brain.predictOne(s))   # выбираем оптимальное действие
        
    # изменение состояния агента
    def observe(self, sample, game_num):  # sample = (s, a, r, s_)
        self.memory.add(sample)
        self.epsilon = MIN_EPSILON + (MAX_EPSILON-MIN_EPSILON)*math.exp(-LAMBDA*game_num)

    # обучение по случайному пакету обучающих примеров (batch)
    def train(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)  
        if batchLen<BATCH_SIZE: # будем обучаться только если есть достаточное количество примеров в памяти
            return

        # начальные состояния из пакета
        states = numpy.array([ o[0] for o in batch ])
        # начальные состояния из пакета
        states_ = numpy.array([ o[3] for o in batch ])

        # выгоды для начальных состояний
        p = agent.brain.predict(states)
        # выгоды для конечных состояний
        p_ = agent.brain.predict(states_)

        # сформируем пустой обучающий пакет
        x = numpy.zeros((batchLen, STATE_CNT))
        y = numpy.zeros((batchLen, ACTION_CNT))

        # заполним пакет
        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i] # выгоды действий для начального состояния
            # обновим выгоду только для совершенного действия, для неиспользованных действий выгоды останутся 
            # прежними
            t[a] = r + GAMMA * numpy.amax(p_[i]) # вычислим новую выгоду действия используя награду и 
            # максимальную выгоду конечного состояния
            
            # сохраним значения в batch
            x[i] = s
            y[i] = t

        # обучим сеть по данному пакету
        self.brain.train(x, y)


# In[8]:


#######################
# Статистика
#######################

class Stats():
    def __init__(self):
        self.stats={"game_num": [],"rewards": [], "success_steps": [], "fps": [], "steps":[], "epsilon":[]}

    def save_stat(self, R, info, epsilon, game_num):
        self.stats["rewards"].append(R)
        self.stats["success_steps"].append(R/STEPS_AFTER_TARGET)
        self.stats["game_num"].append(game_num)
        self.stats["epsilon"].append(epsilon)
        self.stats["steps"].append(info["step"])
        self.stats["fps"].append(info["fps"])
    def show_stat(self):
        # отобраим процент удачных шагов за опыт
        plt.plot(self.stats["game_num"], self.stats["success_steps"], "b.")
        # отобразим сглаженный график
        x, y = self.fit_data(self.stats["game_num"],  self.stats["success_steps"])
        plt.plot(x, y, "r-")
        # второй вариант сглаживания    
        # plt.plot(numpy.linspace(self.stats["game_num"][0], self.stats["game_num"][-1],50), 
        #          numpy.average(numpy.array_split(self.stats["success_steps"][:-1], 50),1), "g-")
        plt.show()
    #  Полиномиальное сглаживание
    def fit_data(self, x, y):
        z = numpy.polyfit(x, y, 3)
        f = numpy.poly1d(z)
        # новые данные размерностью 50
        x_new = numpy.linspace(x[0], x[-1], 50)
        y_new = f(x_new)
        return [x_new, y_new]


# In[9]:


###########
# MAIN
###########
get_ipython().run_line_magic('matplotlib', 'inline')
MAX_GAMES = 50000   # максимальное количество игр
RENDER_PERIOD = 1000 # период генерации видео с опытом (0 для отключения)

env = Environment()
agent = Agent()
stats = Stats()

for game_num in range(MAX_GAMES):
    print ("Game %d:" % game_num)
    render_imgs = []
    observation, r, done, info = env.reset()
    s = observation
    R = r
    
    if RENDER_PERIOD and (game_num % RENDER_PERIOD == 0):
        plt.subplots()
    
    while True:
        # возьмем оптимальное действие на основе текущего состояния
        a = agent.act(s)
        # запустим шаг симуляции
        observation, r, done, info = env.step(a)
        s_ = observation # новое состояние
        # сохраним состояние агента
        agent.observe((s, a, r, s_), game_num)
        # обучим сеть по случайносу batch-у
        agent.train()
        
        s = s_
        R += r
        
        # сохраним изображение, если необходимо
        if RENDER_PERIOD and game_num % RENDER_PERIOD == 0:
            rgb = env.render()
            render_imgs.append([plt.imshow(rgb, animated=True)])

        if done:
            break
        #time.sleep(1./130)
#     if (game_num%(RENDER_PERIOD*5)==0) and (game_num!=0):
#         print ("Game %d:" % game_num)
    print("Total reward:", R, " FPS:", info['fps'])
    
    # сохраним статистику
    stats.save_stat(R, info, agent.epsilon, game_num)
    
    # сформируем анимацию игры и графики статистики обучения
    if len(render_imgs):
        render_start = time.time()
        ani = animation.ArtistAnimation(plt.gcf(), render_imgs, interval=10, blit=True,repeat_delay=1000)
        plt.close()
        display(HTML(ani.to_html5_video()))
        # статистика
        if game_num != 0:
            plt.subplots(figsize=(10,4))
            stats.show_stat()
            plt.close()
        render_stop = time.time()
        print ("render time: %f sec.\n---\n" % (render_stop - render_start))


# In[10]:


5%2


# In[ ]:




