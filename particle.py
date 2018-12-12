import numpy as np
import random
from copy import copy


class Particle:
    """Particle class for PSO./n
    Paramenters:/n
    position        starting position/n
    speed           starting speed/n
    fitness         fitness function/n
    mass            coefficient of current speed to be kept for next iteration step/n
                    DEFAULT: (1,1)/n
    lrate           tuple of coefficients for global best and personal best respectively/n
                    DEFAULT: (1,1)/n
    randsigma       Variance of random component/n
                    DEFAULT: 0.3/n
    tinterval       time interval for move step/n
                    DEFAULT: 1/n
    value_ranges    array of tuples containing lower and upper limit for the values of each parameter/n
                    DEFAULT: None/n
    wrap            boolean determining whether values wrap around a given range/n
                    DEFAULT: False
    """


    def __init__(self, position, speed, fitness, mass=1, lrate=(1,1),
                randsigma=0.3, tinterval=1, value_ranges=None, wrap=False, **kwargs):
        """First value in coefficients is for gbest, second for pbest."""
        position=np.asarray(position)
        speed=np.asarray(speed)
        assert len(position.shape)==1
        assert len(speed.shape)==1
        assert position.shape==speed.shape

        self.mass=mass
        self.position=position
        self.speed=speed
        self.fitness=fitness
        self.pbest=[copy(self.position)]
        self.lrate=np.asarray(lrate)
        self.randsigma=randsigma
        self.tinterval=tinterval
        self.bounce=1
        self.value_ranges=value_ranges
        self.wrap=wrap
        if wrap:
            if self.value_ranges.any():
                self.bounce=None
            else:
                raise ValueError("Can only use wraparound within a value range.")


    def get(self):
        return copy(self.position)

    def accelerate(self, acceleration):
        self.speed=self.speed*self.mass+acceleration*self.tinterval

    def scatter(self):
        self.position+=+np.random.normal(0, self.randsigma, self.speed.shape)

    def movestep(self):
        self.position+=self.speed*self.tinterval

    def distance(self, positions):
        if len(positions.shape)==1:
            return np.apply_along_axis(lambda x: x**2, 0, self.position-position)
        else:
            return np.apply_along_axis(distance, 0, positions)

    def calcaccel(self, gbest):
        pbest=random.choice(self.pbest)
        bests=np.asarray((gbest, pbest))
        #randomness=np.random.normal(0, self.randsigma, (2))
        randomness=np.random.random((2))
        diffs=bests-np.asarray((self.position, self.position))
        accel=np.dot(np.transpose(diffs), self.lrate*randomness)
        return accel

    def check_range(self):
        for i in range(len(self.position)):
            if self.position[i]<self.value_ranges[i,0]:
                self.position[i]=self.value_ranges[i,0]
                if self.bounce:
                    self.speed[i]=-1*self.bounce*self.speed[i]
            elif self.position[i]>self.value_ranges[i,1]:
                self.position[i]=self.value_ranges[i,1]
                if self.bounce:
                    self.speed[i]=-1*self.bounce*self.speed[i]
            if self.speed[i]<-1:
                self.speed[i]=-1
            elif self.speed[i]>1:
                self.speed[i]=1

    def check_range_wrapping(self):
        for i in range(len(self.position)):
            if self.position[i]<self.value_ranges[i,0] or self.position[i]>self.value_ranges[i,1]:
                self.position[i]=self.position[i]%(self.value_ranges[i,1]-self.value_ranges[i,0])
                self.speed[i]=0
            """if self.speed[i]<-1:
                self.speed[i]=-1
            elif self.speed[i]>1:
                self.speed[i]=1
"""
    def update_pbest(self):
        pbestfit=self.fitness(self.pbest[0])
        score=self.fitness(self.position)
        if score>pbestfit:
            self.pbest=[copy(self.position)]
        elif score==pbestfit:
            self.pbest.append(self.position)
        return score

    def step(self, gbest, gbest_fit):
        self.accelerate(self.calcaccel(gbest))
        self.movestep()

        if not(self.value_ranges is None):
            if self.wrap:
                self.check_range_wrapping()
            else:
                self.check_range()

        score=self.update_pbest()

        if score>gbest_fit:
            return 1
        elif score==gbest_fit:
            return 2
        else:
            return 0
