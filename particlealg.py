"""A Particle Swarm Optimization library. /n
Lucas Nakano 30/10/2018 /n
Version 1.0 - First fully working version."""

from particle import Particle
import numpy as np
from copy import copy
import random

##TO IMPROVE: IMPLEMENT WRAPAROUND,


class ParticleOptimizer:
    """Base optimizer class./n
    Arguments:/n
    n_dims          number of optimizeable paramenters/n
    n_particles     number of particles/n
    fitness         the fitness function/n
    value_ranges    (optional) array of shape (n_dims, 2) with the range of each parameter/n
    for other arguments, check documentation for Particle class"""

    def __init__(self, n_dims, n_particles, fitness, **kwargs):
        self.speedscale=1
        self.fitness=fitness
        self.make_particles(n_dims, n_particles, **kwargs)
        self.gbest=None
        self._uptodate=False

    def make_particles(self, n_dims, n_particles, **kwargs):
        self.particles=[]
        pos = np.random.random((n_particles, n_dims))
        spd = self.speedscale*(np.random.random((n_particles, n_dims))-0.5)
        if "value_ranges" in kwargs:
            assert kwargs["value_ranges"].shape==(n_dims, 2)
            kwargs["value_ranges"]=np.asarray(kwargs["value_ranges"])
            pos = pos*(kwargs["value_ranges"][:,1]-kwargs["value_ranges"][:,0])+kwargs["value_ranges"][:,0]
        else:
            kwargs["value_ranges"]=None
        for i in range(n_particles):
            self.particles.append(Particle(pos[i], spd[i], self.fitness, **kwargs))

    def get(self):
        return random.choice(self.gbest)

    def get_all(self):
        positions=[]
        for particle in self.particles:
            positions.append(particle.get())
        return positions

    def get_speeds(self):
        speeds=[]
        for particle in self.particles:
            speeds.append(particle.speed)
        return speeds

    def _get_best(self):
        best=[self.particles[0].get()]
        for i in range(1, len(self.particles)):
            position=self.particles[i].get()
            fitness=self.fitness(position)
            best_fit=self.fitness(best[0])
            if fitness>best_fit:
                best=[position]
            elif fitness==best_fit:
                best.append(position)
        return best

    def gbestprop():
        doc = "The gbest property."
        def fget(self):
            if self._uptodate:
                return self._gbest
            else:
                current_best=self._get_best()
                if not(self._gbest) or self.fitness(current_best[0])>self.fitness(self._gbest[0]):
                    self._gbest=current_best
                self._uptodate=True
                return self._gbest
        def fset(self, value):
            if value==None:
                self._gbest=None
            else:
                raise Exception("Dont mess with gbest manually")
        def fdel(self):
            del self._gbest
        return locals()
    gbest = property(**gbestprop())

    def step(self):
        gbest_fit=self.fitness(self.gbest[0])
        for particle in self.particles:
            particle.step(random.choice(self.gbest), gbest_fit)
        self._uptodate=False

    def get_fitness(self):
        return self.fitness(self.gbest[0])

    def scatter(self):
        for particle in self.particles:
            particle.scatter()


class PermutationOptimizer(ParticleOptimizer):

    def __init__(self, n_particles, valuearray, mass=1.1, decoding="sort",
                lrate=(1,1), **kwargs):
        """Valuearray should be n_dims by n_dims, with the value at (x,y) representing
            the value of element y if in position x."""
        assert valuearray.shape[0]==valuearray.shape[1]
        if decoding=="ordered":
            self.decode=self.ordereddecode
        elif decoding=="sort":
            self.decode=self.sortdecode
        else:
            raise Exception()

        self.valuearray=valuearray
        self.fitness=self.make_fitness()
        self.speedscale=0.5

        n_dims=self.valuearray.shape[0]
        kwargs["mass"]=mass
        kwargs["value_ranges"]=np.asarray([(0, 0.9999)]*n_dims)
        kwargs["lrate"]=lrate

        if not("n_particles" in kwargs):
            kwargs["n_particles"]=n_dims*3
        if not ("smoothing" in kwargs):
            kwargs["smoothing"]=(3, 0.1)

        self.make_particles(n_dims, **kwargs)
        self.smooth(*kwargs["smoothing"])

        self.gbest=None
        self._uptodate=False

    def get(self):
        return self.decode(self.gbest[0])

    def get_raw(self):
        return self.gbest[0]

    def get_all(self):
        positions=[]
        for particle in self.particles:
            positions.append(self.decode(particle.get()))
        return positions

    def get_all_raw(self):
        positions=[]
        for particle in self.particles:
            positions.append((particle.get()))
        return positions

    def ordereddecode(self, positions, **kwargs):
        slots = list(range(len(self.valuearray)))
        permutation = []
        for i in range(len(self.valuearray)):
            index = int(positions[i]*(len(self.valuearray)-i))
            permutation.append(slots.pop(index))
        return permutation

    def sortdecode(self, positions, **kwargs):
        values=list(zip(positions, range(len(self.valuearray))))
        values.sort(key=lambda x: x[0])
        permutation=list(map(lambda x: x[1], values))
        return permutation

    def evaluate(self, permutation):
        value=0
        for i in range(len(permutation)):
            value+=self.valuearray[i, permutation[i]]
        return value

    def make_fitness(self):
        def fitnessfunc(positions):
            permutation=self.decode(positions)
            return self.evaluate(permutation)
        return fitnessfunc

    def _smooth(self, iterations=1, c=0.1):
        n_dims=self.valuearray.shape[0]
        for i in range(iterations):
            newarray=np.zeros((n_dims, n_dims))
            for row in range(n_dims):
                for column in range(n_dims):
                    toadd=0
                    if row>0:
                        toadd+=self.valuearray[row-1, column]*c
                    if row<n_dims-1:
                        toadd+=self.valuearray[row+1, column]*c
                    newarray[row, column] = self.valuearray[row, column] + toadd
            self.valuearray = newarray

    def smooth(self, iterations=1, c=0.2):
        n_dims=self.valuearray.shape[0]
        newarray=np.zeros((n_dims, n_dims))
        for row in range(n_dims):
            for column in range(n_dims):
                newvalue=0
                for offset in range(-iterations-1, iterations):
                    currow=row+offset
                    if currow>=0 and currow<n_dims:
                        newvalue+=self.valuearray[currow, column]**2*c**abs(offset)
                newarray[row, column] = newvalue
        self.valuearray = newarray


class PermutationOptimizerEX(PermutationOptimizer):

        def __init__(self, n_particles, valuearray, mass=1.1, decoding="sort",
                    lrate=(1,1), **kwargs):
            """Valuearray should be n_dims by n_dims, with the value at (x,y) representing
                the value of element y if in position x."""
            assert valuearray.shape[0]==valuearray.shape[1]
            if decoding=="ordered":
                self.decode=self.ordereddecode
            elif decoding=="sort":
                self.decode=self.sortdecode
            else:
                raise Exception()


            self.true_valuearray=valuearray
            cleared_array=self.prelocate(valuearray)
            self.valuearray=cleared_array

            super().__init__(n_particles, valuearray, mass, decoding, lrate, **kwargs)

        def prelocate(self, valuearray):
            self.instant_matches(valuearray)
            cleared_array=valuearray[self.exclusion_mask]
            cleared_array=cleared_array[:, self.exclusion_mask]
            return cleared_array

        def instant_matches(self, valuearray):
            "included gives positions, excluded gives values"
            rowmax = np.argmax(valuearray, axis=1)
            columnmax = np.argmax(valuearray, axis=0)
            matches=[]
            n_dims=valuearray.shape[0]
            self.exclusion_mask=np.ones((n_dims), dtype="bool")
            self.excluded=[]
            self.unknown_positions=[]
            for dim in range(n_dims):
                if columnmax[rowmax[dim]]==dim:
                    self.exclusion_mask[dim]=False
                    self.excluded.append(rowmax[dim])
                else:
                    self.unknown_positions.append(dim)

        def reinclude(self, array, excluded):
            reconstructed=np.zeros(len(self.exclusion_mask))
            array_index=0
            excluded_index=0
            for i, included in enumerate(self.exclusion_mask):
                if included :
                    reconstructed[i]=array[array_index]
                    array_index+=1
                else:
                    reconstructed[i]=excluded[excluded_index]
                    excluded_index+=1
            assert array_index==len(array)
            assert excluded_index==len(excluded)
            return reconstructed

        def get(self):
            true_positions=[self.unknown_positions[i] for i in self.decode(self.gbest[0])]
            return self.reinclude(true_positions, self.excluded)

        def get_all(self):
            positions=[]
            for particle in self.particles:
                true_positions=[self.unknown_positions[i] for i in self.decode(particle.get())]
                positions.append(self.reinclude(true_positions, self.excluded))
            return positions
