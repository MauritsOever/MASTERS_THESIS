# -*- coding: utf-8 -*-
"""
Created on Tue May 24 09:27:29 2022

@author: MauritsvandenOeverPr
"""
from scipy.optimize import minimize
import numpy as np
from scipy.special import gamma
import torch

class DCC_garch:
    
    def __init__(self, dist = 'norm'):
        if dist == 'norm' or dist == 't':
            self.dist = dist
        else: 
            print("Takes pdf name as param: 'norm' or 't'.")
            
    def garch_fit(self, returns):
        res = minimize( self.garch_loglike, (0.01, 0.01, 0.94), args = returns,
              bounds = ((1e-6, 1), (1e-6, 1), (1e-6, 1)))
        return res.x

    def garch_loglike(self, params, returns):
        T = len(returns)
        var_t = self.garch_var(params, returns)
        LogL = np.sum(-np.log(2*np.pi*var_t)) - np.sum( (returns.A1**2)/(2*var_t))
        return -LogL

    def garch_var(self, params, returns):
        T = len(returns)
        omega = params[0]
        alpha = params[1]
        beta = params[2]
        var_t = np.zeros(T)     
        for i in range(T):
            if i==0:
                var_t[i] = returns[i]**2
            else: 
                var_t[i] = omega + alpha*(returns[i-1]**2) + beta*var_t[i-1]
        return var_t        
        
    def mgarch_loglike(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        Q_bar = np.cov(self.rt.reshape(self.N, self.T))

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        H_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

        loglike = 0
        for i in range(1,self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))   

            loglike = (loglike + self.N*np.log(2*np.pi) + 
                      2*np.log(D_t[i].sum()) + 
                      np.log(np.linalg.det(R_t[i])) + 
                      np.matmul(self.rt[i], (np.matmul( np.linalg.inv(H_t[i]), self.rt[i].T))))
        return loglike

    
    def mgarch_logliket(self, params, D_t):
        # No of assets
        a = params[0]
        b = params[1]
        dof = params[2]
        Q_bar = np.cov(self.rt.reshape(self.N, self.T))

        Q_t = np.zeros((self.T,self.N,self.N))
        R_t = np.zeros((self.T,self.N,self.N))
        H_t = np.zeros((self.T,self.N,self.N))
        
        Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

        loglike = 0
        for i in range(1,self.T):
            dts = np.diag(D_t[i])
            dtinv = np.linalg.inv(dts)
            et = dtinv*self.rt[i].T
            Q_t[i] = (1-a-b)*Q_bar + a*(et*et.T) + b*Q_t[i-1]
            qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

            R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


            H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts))   

            loglike = (loglike + np.log( gamma((self.N+dof)/2.)) - np.log(gamma(dof/2)) 
                      -(self.N/2.)*np.log(np.pi*(dof - 2)) - np.log(np.linalg.det(H_t[i])) 
- ((dof+ self.N)*( ((np.matmul(self.rt[i], (np.matmul( np.linalg.inv(H_t[i]), self.rt[i].T))))/(dof - 2.)) + 1)/2.))


        return -loglike
    
    
    def predict(self, out_of_sampledata=None):
        if not out_of_sampledata:
            Q_bar = np.cov(self.rt.reshape(self.N, self.T))

            Q_t = np.zeros((self.T,self.N,self.N))
            R_t = np.zeros((self.T,self.N,self.N))
            self.H_t = np.zeros((self.T,self.N,self.N))

            Q_t[0] = np.matmul(self.rt[0].T/2, self.rt[0]/2)

            for i in range(1,self.T):
                dts = np.diag(self.D_t[i])
                dtinv = np.linalg.inv(dts)
                et = dtinv*self.rt[i].T
                Q_t[i] = (1-self.a-self.b)*Q_bar + self.a*(et*et.T) + self.b*Q_t[i-1]
                qts = np.linalg.inv(np.sqrt(np.diag(np.diag(Q_t[i]))))

                R_t[i] = np.matmul(qts, np.matmul(Q_t[i], qts))


                self.H_t[i] = np.matmul(dts, np.matmul(R_t[i], dts)) 
            
        else:
            raise NotImplementedError
                
        return
            
            
    def fit(self, returns):
        self.rt = np.matrix(returns)
        
        self.T = self.rt.shape[0]
        self.N = self.rt.shape[1]
        
        if self.N == 1 or self.T == 1:
            return 'Required: 2d-array with columns > 2' 
        self.mean = self.rt.mean(axis = 0)
        self.rt = self.rt - self.mean
        
        D_t = np.zeros((self.T, self.N))
        for i in range(self.N):
            params = self.garch_fit(self.rt[:,i])
            D_t[:,i] = np.sqrt(self.garch_var(params, self.rt[:,i]))
        self.D_t = D_t
        if self.dist == 'norm':
            res = minimize(self.mgarch_loglike, (0.01, 0.94), args = D_t,
            bounds = ((1e-6, 1), (1e-6, 1)), 
            #options = {'maxiter':10000000, 'disp':True},
            )
            self.a = res.x[0]
            self.b = res.x[1]
            
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b} 
        elif self.dist == 't':
            res = minimize(self.mgarch_logliket, (0.01, 0.94, 3), args = D_t,
            bounds = ((1e-6, 1), (1e-6, 1), (3, None)), 
            #options = {'maxiter':10000000, 'disp':True},
            )
            self.a = res.x[0]
            self.b = res.x[1]
            self.dof = res.x[2]
            return {'mu': self.mean, 'alpha': self.a, 'beta': self.b, 'dof': self.dof} 

#%%
#########################################################################################
from scipy.optimize import minimize
import numpy as np
from scipy.special import gamma
import torch

class robust_garch_torch:
    
    def __init__(self, data, dist, output=False):
        self.data = self.force_tensor(data)
        self.n = data.shape[0]
        self.K = data.shape[1]
        # init params
        self.omega = torch.cov(data.T)
        self.dist = dist  #'normal', 't'
        if self.dist == 't':
            self.nu = 6
        
        self.output = output
        
        paramslist = [0.96]
        for dim in range(self.K):
            paramslist += [0]*dim + [0.14]
        
        self.params =  torch.Tensor(paramslist)
        self.params.requires_grad = True
        
    def force_tensor(self, data):
        """
        forces the given object into a float tensor

        Parameters
        ----------
        X : np.array or pd.DataFrame of data

        Returns
        -------
        float tensor of given data

        """
        # write code that forces X to be a tensor
        if type(data) != torch.Tensor:
            return torch.Tensor(data).float()
        else:
            return data.float() # force it to float anyway

        
    def loglik(self, params):
        # construct coefs
        beta, A = self.construct_params(params)                    
        # calc log likelihood
        sigmat = self.omega
        
        if self.dist == 'normal':
            loglik = self.n * self.K*torch.log(torch.Tensor([torch.pi*2]))
            loglik = torch.reshape(loglik, (1,1))
            for row in range(1, self.n):
                obs = torch.reshape(self.data[row,:], (self.K, 1))

                sigmat = (1-beta)*self.omega + A@(obs@obs.T - sigmat)@A.T + beta*sigmat
                loglik += -0.5*(torch.log(torch.linalg.det(sigmat))) + (obs.T@torch.linalg.inv(sigmat)@obs) 

                                
        elif self.dist == 't':
            Gammas_term = torch.Tensor([gamma((self.nu+self.K)/2)/gamma(self.nu/2)])
            loglik = (-2*torch.log(Gammas_term) + self.K*torch.log(torch.Tensor([torch.pi])) + 
                      torch.log(torch.Tensor([self.nu-2.])) + (self.K-1)*torch.log(torch.Tensor([self.nu])))*self.n
            loglik = torch.reshape(loglik, (1,1))
            for row in range(1, self.n):
                obs = torch.reshape(self.data[row,:], (self.K, 1))
                sigmat = (1-beta)*self.omega + A@(obs@obs.T/(1 + obs.T@torch.linalg.inv(sigmat)@obs) - sigmat)@A.T + beta*sigmat
                loglik += torch.log(torch.linalg.det(sigmat)) + (self.nu + self.K)*torch.log(1 + 1/(self.nu-2) * obs.T@torch.linalg.inv(sigmat)@obs)
        
        # print(-1*loglik[0]/self.n)
        return (-1*loglik[0])/self.n
    
    def fit(self, epochs):
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        
        if self.dist == 'normal':
            learning_rate = 0.015
        elif self.dist == 't':
            learning_rate = 0.01
        
        
        optimizer = torch.optim.AdamW([self.params],
                             lr = learning_rate,
                             weight_decay = 1e-3) # specify some hyperparams for the optimizer

        
        logliks = []
        print(f'fitting MGARCH(1,1) for {epochs} epochs...')
        for epoch in tqdm(range(epochs)):
            loss = self.loglik(self.params)
            logliks += [loss.detach().numpy()]
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
          
        if self.output:
            plt.plot(logliks)
            plt.show()
            beta, A = self.construct_params(self.params)
            print(f'beta = {beta.detach().numpy()}')
            print('A    = ')
            print(A.detach().numpy())
            
            print('storing sigmas...')
        self.store_sigmas()
        return
    
    def construct_params(self, params):
        # params = self.params
        beta = params[0]
        params = params[1:]
        
        A = torch.zeros((self.K, self.K))
        for i in range(self.K):
            params = params[i:]
            A[i, 0:i+1] = params[0:i+1]
            
        # reparam
        beta = 1 / (1+torch.exp(-beta))
        
        for i in range(self.K):
            for j in range(self.K):
                if i == j:
                    A[i,j] = 1 / (1+torch.exp(-A[i,j]))
                if i > j:
                    A[i,j] = (1/3)*(-1 + 2/(1 + torch.exp(-A[i,j])))
        
        # print(f'beta = {beta.detach().numpy()}')
        # print('A    = ')
        # print(A.detach().numpy())

        return beta, A
    
    def store_sigmas(self):
        beta, A = self.construct_params(self.params)                    
        self.sigmas = [self.omega]
        for row in range(1, self.n):
            obs = torch.reshape(self.data[row,:], (self.K, 1))
            self.sigmas += [(1-beta)*self.omega + A@(obs@obs.T - self.sigmas[row-1])@A.T + beta*self.sigmas[row-1]]
    
    def estimate_sigmas(self, data):
        beta, A = self.construct_params(self.params)
        sigmas = [torch.cov(data.T)]
        for row in range(1, data.shape[0]):
            obs = torch.reshape(data[row,:], (data.shape[1], 1))
            sigmas += [(1-beta)*sigmas[0] + A@(obs@obs.T - sigmas[row-1])@A.T + beta*sigmas[row-1]]
            
        return sigmas

#%%
# import os
# os.chdir(r'C:\Users\MauritsvandenOeverPr\OneDrive - Probability\Documenten\GitHub\MASTERS_THESIS')

# from data.datafuncs import GetData, GenerateAllDataSets

# data = GetData('returns')[0][:,5:9]
# data = torch.Tensor(data)
# garch = robust_garch_torch(data, 't', output=True)

# garch.fit(epochs=50)
# garch.store_sigmas()

# count = 0
# for i in range(len(garch.sigmas)):
#     try:
#         torch.linalg.cholesky(garch.sigmas[i])
#     except:
#         print(f'WE FOUND HIM, ITS NUMBER {i}')
#         count += 1

# print(f'ratio of non p-def matrices in sigmas = {count / len(garch.sigmas)}')

