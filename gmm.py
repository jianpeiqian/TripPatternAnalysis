import numpy as np
from scipy.stats import norm
    
class GMM2Level():   
    def __init__(self, X, K, H, ini_mu, ini_sigma):
        self.X = X
        
        self.M = len(self.X)
        self.K = K                    #number of passenger cluster
        self.H = H                    #number of gaussian cluster
                
        self.Pi = np.ones(self.K)/self.K
        self.Tau = np.ones((self.K,self.H))/self.H
        
        self.Mu = ini_mu
        self.Sigma  = ini_sigma  
        
        self.Z1 = np.array([[0.0]  * self.K for _ in range(self.M)])     #旅客i在每个k上的责任（隐变量）
        self.t1 = np.array([[0.0]  * self.K for _ in range(self.M)])     #旅客i在每个k上的责任（后验概率）
        
        self.Z2 = []
        self.t2 = []      
    
    #计算第1层隐变量的后验概率，即旅客i的卡类k
    def E1_cal_resp(self): 
        for i in range(self.M):
            Ni = len(self.X[i])
            pdf =  np.array([[[0.0] * self.H ] * self.K for _ in range(Ni)])
            for k in range(self.K):
                for h in range(self.H):
                    pdf[:, k, h] = self.Tau[k, h] * norm.pdf(self.X[i], self.Mu[k, h], self.Sigma[k, h])
                pdfs = pdf.sum(axis = 2)
                self.t1[i, k] = np.log(self.Pi[k] * np.prod(pdfs[:,k]))
        return self.t1
    
    #计算潜变量Z1
    def E1_cal_Z1(self):
        self.Z1 = ((self.t1.T - np.max(self.t1, axis = 1)).T == 0).astype(int)
        return self.Z1
    
    #计算第一层的卡类型权重
    def M1_update_Pi(self):
        self.Pi = self.Z1.sum(axis = 0) / self.Z1.sum()
        return self.Pi
    
    #计算第2层隐变量的后验概率，即旅客i第j次出行(k类卡)的高斯h
    def E2_cal_resp(self):
        for i in range(self.M):
            Ni = len(self.X[i])
            pdf = np.array([[[0.0] * self.H ] * self.K for _ in range(Ni)])
            for k in range(self.K):
                for h in range(self.H):
                    pdf[:, k, h] = self.Tau[k, h] * norm.pdf(self.X[i], self.Mu[k, h], self.Sigma[k, h])
            self.t2.append(pdf/pdf.sum(axis=2).reshape(Ni,self.K,1))
        return self.t2
    
    #def E2_cal_Z2(self):
                
    #计算第二层的高斯子模型权重 
    def M2_update_Tau(self): 
        _tau = np.array([[[0.0] * self.H ] * self.K for _ in range(self.M)])
        _n = []
        for i in range(self.M):
            _n.append(len(self.X[i]) * self.Z1[i])
            _tau[i] = np.einsum('k,jkh->kh',self.Z1[i],self.t2[i])
        self.Tau = np.sum(_tau, axis=0)/np.sum(_n, axis=0).reshape(self.K,1)
        return self.Tau
    
    def M2_update_Mu(self):
        _mu = np.array([[[0.0] * self.H ] * self.K for _ in range(self.M)])
        _tau = np.array([[[0.0] * self.H ] * self.K for _ in range(self.M)])
        for i in range(self.M):
            _mu[i] = np.einsum('kj,jkh->kh',np.einsum('k,j->kj',self.Z1[i],self.X[i]), self.t2[i])
            _tau[i] = np.einsum('k,jkh->kh',self.Z1[i],self.t2[i])
        self.Mu = np.sum(_mu, axis=0)/np.sum(_tau, axis=0)
        return self.Mu
    
    def M2_update_Sigma(self):
        _sigma = np.array([[[0.0] * self.H ] * self.K for _ in range(self.M)])
        _tau = np.array([[[0.0] * self.H ] * self.K for _ in range(self.M)])
        for i in range(self.M):
            _sigma[i] = np.sum(self.t2[i] * self.Z1[i].reshape(1,self.K,1) * np.array([self.X[i][j]-self.Mu for j in range(len(self.X[i]))]) ** 2, axis=0)
            _tau[i] = np.einsum('k,jkh->kh',self.Z1[i],self.t2[i])
        self.Sigma =  np.sqrt(np.sum(_sigma, axis=0)/np.sum(_tau, axis=0))
        return self.Sigma
    
    def fit(self, iter):
        iters = iter
        for i in range(iters):
            self.E1_cal_resp()                      #t1
            self.E1_cal_Z1()
            self.M1_update_Pi()
            #print('round',i,'M1_update_Pi')
            print(self.Pi)
            self.E2_cal_resp()                      #t2
            self.M2_update_Tau()
            #print('round',i,'M2_update_Tau')
            #print(self.Tau)            
            self.M2_update_Mu()
            #print('round',i,'M2_update_Mu')
            print(self.Mu)
            self.M2_update_Sigma()
            #print('round',i,'M2_update_Sigma')
            #print(self.Sigma)

def main():
    import pandas as pd
    import numpy as np
    
    #导入数据
    #data = pd.read_csv('E:/algorithm/paper/03clustering/gmm2L_em/toy.csv') 
    data = pd.read_csv('E:/algorithm/paper/03clustering/gmm2L_em/ttt2+_grouped.csv')
    #空值处理
    X2np = np.array(data.fillna('del_token'))
    X_texts = X2np.tolist()
    stoplist = {'del_token'}
    X_docs = [[word for word in datarow if word not in stoplist] for datarow in X_texts]
    
    #定义簇类数目、用户数、初始化高斯参数
    K = 5
    H = 3
    
    Mu = np.array([[2, 7, 9], [2, 5, 10], [3,10,18], [3,5,8], [10,17,20]])
    Sigma = np.array([[1, 2, 2],[1, 2, 2],[1, 2, 2],[1, 2, 2],[1, 2, 2],[1, 2, 2]])
    
    #模型训练
    import gmm
    gmm_model = gmm.GMM2Level(X_docs, K, H, Mu, Sigma)
    gmm_model.fit(10)   

if __name__ == "__main__":
    main()