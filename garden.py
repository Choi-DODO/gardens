# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 21:24:41 2022

@author: Choi, Dohyeong
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

path = 'C:/Users/user/OneDrive - SNU/바탕 화면/뮤레파/정원/정원면적_huff.xlsx'
file = pd.read_excel(path, sheet_name='Sheet2', engine='openpyxl', header = None)


size = file.iloc[103:150,3:253]
size.columns = file.iloc[2,3:253]
size.reset_index(inplace = True, drop = True)
size = size.apply(pd.to_numeric, errors = 'coerce')

pop = file.iloc[156:406,3:5]
pop.columns = file.iloc[155,3:5]
pop = pop.apply(pd.to_numeric, errors = 'coerce')
pop.reset_index(inplace = True, drop = True)

nation = file.iloc[156:203,12]
nation = nation.apply(pd.to_numeric, errors = 'coerce')
nation.reset_index(inplace = True, drop = True)

visitor = file.iloc[156:203,11]
visitor.columns = file.iloc[155,11]
visitor = visitor.apply(pd.to_numeric, errors = 'coerce')
visitor.reset_index(inplace = True, drop = True)
visitor = visitor.fillna(visitor.mean()) #평균 = 351462명

visitor = np.log(visitor)
plt.hist(visitor)

def minmax_norm(df):
    return (df - df.min()) / ( df.max() - df.min())

def mean_norm(df):
    return df.apply(lambda x: (x-df.mean())/ df.std())


garden_name = pd.Series(file.iloc[156:203,7])
garden_name.columns = ['GDN_NAME']
garden_name.reset_index(inplace = True, drop = True)

dist = file.iloc[3:50,3:253]
dist.columns = file.iloc[2,3:253]
dist.reset_index(inplace = True, drop = True)
dist = dist.apply(pd.to_numeric, errors = 'coerce')

region = file.iloc[53:100,3:253]
region.columns = file.iloc[52,3:253]
region.reset_index(inplace = True, drop = True)
region = region.apply(pd.to_numeric, errors = 'coerce')

name_dist_size = pd.concat([garden_name, size.iloc[:,0], dist], axis = 1)


#%% laboratory
dist_dummy = region.replace(1,1)
dist_dummy = dist_dummy.replace(0,1)

pop_dummy = region.replace(1,0.3)
pop_dummy = pop_dummy.replace(0,0.7)

new_dist = dist * dist_dummy

#######################################
att = size

att_dist = att.div(new_dist, axis = 0)

sum_att_dist =  np.array(att_dist).sum(axis = 0)
    
sum_att_dist_hat = np.diag(sum_att_dist)
sum_att_dist_hat_inv = np.linalg.pinv(sum_att_dist_hat)
    
prob = att_dist @ sum_att_dist_hat_inv

HI = pd.Series(prob.sum(axis = 1))
HI_pop = pd.Series(prob @ pop.iloc[:,1])

HI_pop_weighted = list()
for i in range(len(HI)):

    weighted_pop = np.array(pop_dummy.iloc[i,]).reshape(-1,1) * np.array(pop.iloc[:,1]).reshape(-1,1)
    HIPW = np.array(prob.iloc[i,:]).reshape(1,-1) @ weighted_pop
    
    HI_pop_weighted.extend(HIPW[0])
    
HI_pop_weighted = pd.Series(HI_pop_weighted)

result = pd.concat([garden_name, HI, HI_pop, HI_pop_weighted], axis = 1)
result.columns = ['name', 'HI', 'HI_pop', 'HI_pop_weighted']

########################################
new_att = size.mul(visitor, axis = 0)

new_att_dist = new_att.div(new_dist, axis = 0)

new_sum_att_dist =  np.array(new_att_dist).sum(axis = 0)
    
new_sum_att_dist_hat = np.diag(new_sum_att_dist)
new_sum_att_dist_hat_inv = np.linalg.pinv(new_sum_att_dist_hat)
    
new_prob = new_att_dist @ new_sum_att_dist_hat_inv

new_HI = pd.Series(new_prob.sum(axis = 1))
new_HI_pop = pd.Series(new_prob @ pop.iloc[:,1])

new_HI_pop_weighted = list()
for i in range(len(new_HI)):

    weighted_pop = np.array(pop_dummy.iloc[i,]).reshape(-1,1) * np.array(pop.iloc[:,1]).reshape(-1,1)
    new_HIPW = np.array(new_prob.iloc[i,:]).reshape(1,-1) @ weighted_pop
    
    new_HI_pop_weighted.extend(new_HIPW[0])
    
new_HI_pop_weighted = pd.Series(new_HI_pop_weighted)

result = pd.concat([garden_name, new_HI, new_HI_pop, new_HI_pop_weighted], axis = 1)
result.columns = ['name', 'HI', 'HI_pop', 'HI_pop_weighted']



#%% version5
def GARDEN(garden_code, garden_size, _in, _out, normalize = False):
            
    dist_dummy = region.replace(1,_in)
    dist_dummy = dist_dummy.replace(0,_out)
    
    new_dist = dist * dist_dummy
    
    pop_dummy = region.replace(1,0.3)
    pop_dummy = pop_dummy.replace(0,0.7)
    
    new_size = size.copy()
    new_size.iloc[garden_code,:] = garden_size
    
    new_att = new_size.mul(nation, axis = 0)
    
    new_att = new_att.mul(visitor, axis = 0)

    new_att_dist = new_att.div(new_dist, axis = 0)

    new_sum_att_dist =  np.array(new_att_dist).sum(axis = 0)
        
    new_sum_att_dist_hat = np.diag(new_sum_att_dist)
    new_sum_att_dist_hat_inv = np.linalg.pinv(new_sum_att_dist_hat)
        
    new_prob = new_att_dist @ new_sum_att_dist_hat_inv

    new_HI = pd.Series(new_prob.sum(axis = 1))
    new_HI_pop = pd.Series(new_prob @ pop.iloc[:,1])

    new_HI_pop_weighted = list()
    for i in range(len(new_HI)):

        weighted_pop = np.array(pop_dummy.iloc[i,]).reshape(-1,1) * np.array(pop.iloc[:,1]).reshape(-1,1)
        new_HIPW = np.array(new_prob.iloc[i,:]).reshape(1,-1) @ weighted_pop
        
        new_HI_pop_weighted.extend(new_HIPW[0])
        
    new_HI_pop_weighted = pd.Series(new_HI_pop_weighted)

    result = pd.concat([garden_name, new_HI, new_HI_pop, new_HI_pop_weighted], axis = 1)
    result.columns = ['name', 'HI', 'HI_pop', 'HI_pop_weighted']
    
    if normalize == False:
        return result
    
    elif normalize == 'minmax':
        
        def minmax_norm(df):
            return (df - df.min()) / ( df.max() - df.min())
        
        result.iloc[:,1:] = minmax_norm(result.iloc[:,1:])
        
        return result
    
    elif normalize == 'norm':
        
        def mean_norm(df):
            return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)    
        
        result.iloc[:,1:] = mean_norm(result.iloc[:,1:])
        
        return result


#%% version4
def GARDEN(garden_code, garden_size, _in, _out, normalize = False):
            
    dist_dummy = region.replace(1,_in)
    dist_dummy = dist_dummy.replace(0,_out)
    
    new_dist = dist * dist_dummy
    
    pop_dummy = region.replace(1,0.3)
    pop_dummy = pop_dummy.replace(0,0.7)
    
    new_size = size.copy()
    new_size.iloc[garden_code,:] = garden_size
    
    new_att = new_size.mul(nation, axis = 0)

    new_att_dist = new_att.div(new_dist, axis = 0)

    new_sum_att_dist =  np.array(new_att_dist).sum(axis = 0)
        
    new_sum_att_dist_hat = np.diag(new_sum_att_dist)
    new_sum_att_dist_hat_inv = np.linalg.pinv(new_sum_att_dist_hat)
        
    new_prob = new_att_dist @ new_sum_att_dist_hat_inv

    new_HI = pd.Series(new_prob.sum(axis = 1))
    new_HI_pop = pd.Series(new_prob @ pop.iloc[:,1])

    new_HI_pop_weighted = list()
    for i in range(len(new_HI)):

        weighted_pop = np.array(pop_dummy.iloc[i,]).reshape(-1,1) * np.array(pop.iloc[:,1]).reshape(-1,1)
        new_HIPW = np.array(new_prob.iloc[i,:]).reshape(1,-1) @ weighted_pop
        
        new_HI_pop_weighted.extend(new_HIPW[0])
        
    new_HI_pop_weighted = pd.Series(new_HI_pop_weighted)

    result = pd.concat([garden_name, new_HI, new_HI_pop, new_HI_pop_weighted], axis = 1)
    result.columns = ['name', 'HI', 'HI_pop', 'HI_pop_weighted']
    
    if normalize == False:
        return result
    
    elif normalize == 'minmax':
        
        def minmax_norm(df):
            return (df - df.min()) / ( df.max() - df.min())
        
        result.iloc[:,1:] = minmax_norm(result.iloc[:,1:])
        
        return result
    
    elif normalize == 'norm':
        
        def mean_norm(df):
            return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)    
        
        result.iloc[:,1:] = mean_norm(result.iloc[:,1:])
        
        return result
#%% version3
def GARDEN(garden_code, garden_size, _in, _out, normalize = False):
            
    dist_dummy = region.replace(1,_in)
    dist_dummy = dist_dummy.replace(0,_out)
    
    new_dist = dist * dist_dummy
    
    pop_dummy = region.replace(1,0.3)
    pop_dummy = pop_dummy.replace(0,0.7)
    
    new_size = size.copy()
    new_size.iloc[garden_code,:] = garden_size
    
    attract = new_size.div(new_dist, axis = 0)
    
    sum_atrct =  np.array(attract).sum(axis = 0)
        
    sum_atrct_hat = np.diag(sum_atrct)
    sum_atrct_hat_inv = np.linalg.pinv(sum_atrct_hat)
        
    prob = attract @ sum_atrct_hat_inv
    
    HI = pd.Series(prob.sum(axis = 1))
    HI_pop = pd.Series(prob @ pop.iloc[:,1])
    
    
    HI_pop_weighted = list()
    for i in range(len(HI)):

        weighted_pop = np.array(pop_dummy.iloc[i,]).reshape(-1,1) * np.array(pop.iloc[:,1]).reshape(-1,1)
        HIPW = np.array(prob.iloc[i,:]).reshape(1,-1) @ weighted_pop
        
        HI_pop_weighted.extend(HIPW[0])
        
    HI_pop_weighted = pd.Series(HI_pop_weighted)

    result = pd.concat([garden_name, HI, HI_pop, HI_pop_weighted], axis = 1)
    result.columns = ['name', 'HI', 'HI_pop', 'HI_pop_weighted']
    
    if normalize == False:
        return result
    
    elif normalize == 'minmax':
        
        def minmax_norm(df):
            return (df - df.min()) / ( df.max() - df.min())
        
        result.iloc[:,1:] = minmax_norm(result.iloc[:,1:])
        
        return result
    
    elif normalize == 'norm':
        
        def mean_norm(df):
            return df.apply(lambda x: (x-x.mean())/ x.std(), axis=0)    
        
        result.iloc[:,1:] = mean_norm(result.iloc[:,1:])
        
        return result
        

    

#%% old version
def GARDEN(regional, national, _in, _out):
            
    DF_dummy = DF_region.replace(1,_in)
    DF_dummy = DF_dummy.replace(0,_out)
    
    new_dist = DF_dist * DF_dummy
    
    dummy = DF_region.replace(1,0.7)
    dummy = dummy.replace(0,0.3)
    
       
    region_garden = regional/new_dist.iloc[0:45,:]
    national_garden = national/new_dist.iloc[45:,:]
        
    attract = pd.concat([region_garden, national_garden], axis = 0)
    
    sum_atrct =  np.array(attract).sum(axis = 0)
        
    sum_atrct_hat = np.diag(sum_atrct)
    sum_atrct_hat_inv = np.linalg.pinv(sum_atrct_hat)
        
    prob = attract @ sum_atrct_hat_inv
    
    HI = pd.Series(prob.sum(axis = 1))
    HI_pop = pd.Series(prob @ pop.iloc[:,1])
    
    
    HI_pop_weighted = list()
    for i in range(len(HI)):

        weighted_pop = np.array(dummy.iloc[i,]).reshape(-1,1) * np.array(pop.iloc[:,1]).reshape(-1,1)
        HIPW = np.array(prob.iloc[i,:]).reshape(1,-1) @ weighted_pop
        
        HI_pop_weighted.extend(HIPW[0])
        
    HI_pop_weighted = pd.Series(HI_pop_weighted)

    result = pd.concat([garden_name, HI, HI_pop, HI_pop_weighted], axis = 1)
    result.columns = ['name', 'HI', 'HI_pop', 'HI_pop_weighted']

    return result


#%%

"""
45 = 순천만 92.6ha
14 = 세미원 12.7ha
"""
wow = GARDEN(14, 12.7, 1.5, 1, normalize = 'minmax')
criteria = np.mean(wow['HI'])

def Size(garden_code):
    diff = []; area = []
    for t in range(10, 101, 1):
        diff.append(GARDEN(garden_code, t, 1.5, 1).iloc[garden_code,1] - criteria)
        area.append(t)    
    return np.array(pd.DataFrame([area, diff]))

WOWW = dict()
for i in range(0,47,1):
    WOWW[i] = Size(i)
    
Base = WOWW[0]
for i in range(1, 47, 1):
    Base = np.vstack([Base, WOWW[i][1]])
Base = pd.DataFrame(np.transpose(Base))

A_Base = np.abs(Base)

final_size = np.transpose(A_Base.apply(lambda x : x.index[x == min(x)], axis=0)).iloc[1:,] + 10

#%%
'세미원 12.7ha'
x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(14, t, 1.5, 1, normalize = 'minmax').iloc[14,1]) for t in x]
y2 = [np.array(GARDEN(14, t, 1.5, 1, normalize = 'minmax').iloc[14,2]) for t in x]
y3 = [np.array(GARDEN(14, t, 1.5, 1, normalize = 'minmax').iloc[14,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')



ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')


line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Semiwon')
plt.show()

#%%
'순천만 92.6ha'
x = np.arange(20, 100, 0.5)
y1 = [np.array(GARDEN(45, t, 1.5, 1, normalize = 'minmax').iloc[45,1]) for t in x]
y2 = [np.array(GARDEN(45, t, 1.5, 1, normalize = 'minmax').iloc[45,2]) for t in x]
y3 = [np.array(GARDEN(45, t, 1.5, 1, normalize = 'minmax').iloc[45,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Suncheonman')
plt.show()


#%%
'태화강 83.5'
x = np.arange(20, 100, 0.5)
y1 = [np.array(GARDEN(46, t, 1.5, 1, normalize = 'minmax').iloc[46,1]) for t in x]
y2 = [np.array(GARDEN(46, t, 1.5, 1, normalize = 'minmax').iloc[46,2]) for t in x]
y3 = [np.array(GARDEN(46, t, 1.5, 1, normalize = 'minmax').iloc[46,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Taehwagang')
plt.show()

#%%
'죽녹원 15.7'
x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(31, t, 1.5, 1, normalize = 'minmax').iloc[31,1]) for t in x]
y2 = [np.array(GARDEN(31, t, 1.5, 1, normalize = 'minmax').iloc[31,2]) for t in x]
y3 = [np.array(GARDEN(31, t, 1.5, 1, normalize = 'minmax').iloc[31,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Juknokwon')
plt.show()

#%%
'거창창포원 21.7'
x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(2, t, 1.5, 1, normalize = 'minmax').iloc[2,1]) for t in x]
y2 = [np.array(GARDEN(2, t, 1.5, 1, normalize = 'minmax').iloc[2,2]) for t in x]
y3 = [np.array(GARDEN(2, t, 1.5, 1, normalize = 'minmax').iloc[2,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Changpowon')
plt.show()

#%%
'연당원 10.4'
x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(25, t, 1.5, 1, normalize = 'minmax').iloc[25,1]) for t in x]
y2 = [np.array(GARDEN(25, t, 1.5, 1, normalize = 'minmax').iloc[25,2]) for t in x]
y3 = [np.array(GARDEN(25, t, 1.5, 1, normalize = 'minmax').iloc[25,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Yeondangwon')
plt.show()

#%%
'구절초 38.8'
x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(29, t, 1.5, 1, normalize = 'minmax').iloc[29,1]) for t in x]
y2 = [np.array(GARDEN(29, t, 1.5, 1, normalize = 'minmax').iloc[29,2]) for t in x]
y3 = [np.array(GARDEN(29, t, 1.5, 1, normalize = 'minmax').iloc[29,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Gujeolcho')
plt.show()

#%%
'안양천 43'
x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(23, t, 1.5, 1, normalize = 'minmax').iloc[23,1]) for t in x]
y2 = [np.array(GARDEN(23, t, 1.5, 1, normalize = 'minmax').iloc[23,2]) for t in x]
y3 = [np.array(GARDEN(23, t, 1.5, 1, normalize = 'minmax').iloc[23,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')

ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')

line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Ahnyangcheon')
plt.show()


#%%
'others'
gdn_code = 12

x = np.arange(5, 50, 0.5)
y1 = [np.array(GARDEN(gdn_code, t, 1.5, 1, normalize = 'minmax').iloc[gdn_code,1]) for t in x]
y2 = [np.array(GARDEN(gdn_code, t, 1.5, 1, normalize = 'minmax').iloc[gdn_code,2]) for t in x]
y3 = [np.array(GARDEN(gdn_code, t, 1.5, 1, normalize = 'minmax').iloc[gdn_code,3]) for t in x]

fig = plt.figure(figsize=(7,5))
fig.set_facecolor('white')


ax1 = fig.add_subplot()
ax1.set_xlabel('Area(ha)')
ax1.set_ylabel('Huff Index')


line1 = ax1.plot(x, y1, color = 'green', label = 'HI')
line2 = ax1.plot(x, y2, color = 'deeppink', label = 'HI_pop')
line3 = ax1.plot(x, y3, color = 'blue', label = 'HI_pop_W')

lines = line1 + line2 + line3
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left')

plt.ylim([0, 0.6])
plt.title('Seonsan')
plt.show()
