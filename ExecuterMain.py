
# coding: utf-8

# In[ ]:


# Importing everything
import sys
import math
import copy
import pandas as pd
import collections.abc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import types
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import auc, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_curve
from imblearn.under_sampling import NeighbourhoodCleaningRule
from sklearn.model_selection import train_test_split
from sklearn import svm
sns.set(style='whitegrid')


# In[ ]:


# Getting the files read
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')
bids_df = pd.read_csv('dataset/bids.csv')


# In[ ]:


print('Train Length:%d' %len(train_df))
print('Test Length:%d' %len(test_df))
print('Bids Length:%d' %len(bids_df))


# In[ ]:


#Getting column info
print('Train columns: %s\n' %(train_df.columns.values))
print('Test columns: %s\n' %(test_df.columns.values))
print('Bids columns: %s\n' %(bids_df.columns.values))


# In[ ]:


#Displaying the data difference
sns.countplot(x='outcome', data=train_df)
print(collections.Counter(train_df['outcome']))


# In[ ]:


#Understanding the difference between top 10 bidding nations
top_bidding_nations = collections.Counter(bids_df['country']).most_common(10)
top_df = pd.DataFrame(top_bidding_nations, columns=['country', 'count'])
sns.barplot(x='country', y='count', data=top_df)
print(top_bidding_nations)


# In[ ]:


#Morphing into metadf
meta_train_df = []
meta_test_df = []
cnt = 0
len_bids = len(bids_df)
#Fetching pre-existing metadata files as computation takes a lot of time
try:
    print('Fetching already existing files')
    meta_train_df = pd.read_csv('dataset/meta_train.csv')
    meta_test_df = pd.read_csv('dataset/meta_test.csv')
#If files don't exist already, create and write the data
except FileNotFoundError as fnfe:
    train_bidders = list(train_df['bidder_id'])
    test_bidders = list(test_df['bidder_id'])
    for bidder_bid in bids_df.values:
        if bidder_bid[1] in train_bidders:
            meta_train_df.append(np.concatenate((bidder_bid, (train_df.loc[train_df['bidder_id'] == bidder_bid[1]]).values[0][1:]), axis=0))
        elif bidder_bid[1] in test_bidders:
            meta_test_df.append(np.concatenate((bidder_bid, (test_df.loc[test_df['bidder_id'] == bidder_bid[1]]).values[0][1:]), axis=0))
        cnt+=1
        sys.stdout.write('\rRecords done:%.4f'%(cnt/len_bids))


# In[ ]:


#Converting to dfs
meta_train_df = pd.DataFrame(meta_train_df, columns=np.concatenate((bids_df.columns, train_df.columns[1:]), axis=0))
meta_test_df = pd.DataFrame(meta_test_df, columns=np.concatenate((bids_df.columns, test_df.columns[1:]), axis=0))


# In[ ]:


#Write dfs to file for saving computations if things go south
meta_train_df.to_csv('dataset/meta_train.csv')
meta_test_df.to_csv('dataset/meta_test.csv')


# In[ ]:


#Sorting train data in order to easy readability access
meta_train_df = meta_train_df.sort_values(['bidder_id', 'merchandise', 'time'],  ascending=[1, 1, 1])


# In[ ]:


#Generating bins in order to plot binwise distribution
sns.set(); np.random.seed(32)
x = list((meta_train_df.loc[meta_train_df['outcome'] == 1.0]).groupby('bidder_id').size())
highest_pow = math.ceil(math.log(max(x), 2))
bins = []
[bins.append((2**(k-1), 2**(k)-1)) for k in range(1, highest_pow)]
changed_ds = []
for i in x:
    [changed_ds.append(k) for k in range(len(bins)) if bins[k][0] <= i <= bins[k][1]]
#Plot graph binwise count of bidders
sns.countplot(changed_ds)
print(bins)
print(list(collections.Counter(changed_ds).most_common()))


# In[ ]:


#Get unique train and test data
unique_train_bidder = meta_train_df.bidder_id.unique()
unique_test_bidder = meta_test_df.bidder_id.unique()

print('Considerable training data: %d'%len(unique_train_bidder))
print('Considerable testing data: %d'%len(unique_test_bidder))


# In[ ]:


#Class to store bidder specific features
class BidderData:
    def __init__(self, bidder_id, bid_ids, countries, urls, auctions, merchandise, devices, times, ips, outcome=None):
        self.bidder_id = bidder_id #Bidder_id
        self.bid_ids = list(bid_ids) #Bid_id
        self.countries = list(countries) #countries
        self.urls = list(urls) #URLs
        self.auctions = list(auctions) #auctions
        self.merchandise = list(merchandise) #merchandise
        self.devices = list(devices) #devices
        self.times = list(times) #times
        self.ips = list(ips) #ips
        self.lasttmbids = 0 #max bids placed in 20 minutes
        self.lastthbids = 0 #max bids placed in 2 hours
        if not outcome == None:
            self.outcome = outcome #outcome target lable
        self.__getRepeatedTimes() #Unique timestamps
        self.__getMerchandiseUnique() #Unique merchandise
        self.__getDevicesUnique() #Unique devices
        self.__getURLsUnique() #Unique urls
        self.__getRatioNativeCountry() #Ratio of bids placed from native country
        self.__getRatioURL() #Ratio of bids placed from main url
        self.__getRatioDevice() #Ratio of bids placed from main device
        self.__medConsecutiveBidsTime() #time of placing consecutive bids
        self.__getIPUnique() #Unique Ips
        self.__getAvgBidsPerAuction() #bids placed per auction
        self.__getLastTMBids() 
        self.__getLastTHBids() 
        self.__medConsecutiveBidsTime()
        self.__len_attrib()
    
    def __len_attrib(self):
        self.total_bids = len(list(self.bid_ids))

    def __getRepeatedTimes(self):
        self.repeated_times = len(self.times) - len(pd.unique(self.times))
    def __getMerchandiseUnique(self):
        self.merchandise_unique = len(pd.unique(self.merchandise))
    def __getDevicesUnique(self):
        self.devices_unique = len(pd.unique(self.devices))
    def __getURLsUnique(self):
        self.urls_unique = len(pd.unique(self.urls))
    def __getIPUnique(self):
        self.ips_unique = len(pd.unique(self.ips))
    
    def __getRatioNativeCountry(self):
        if len(self.countries) > 0:
            self.ratioNC = collections.Counter(self.countries).most_common(1)[0][1]/len(self.countries)
        else:
            self.ratioNC = 1
    def __getRatioURL(self):
        if len(self.urls) > 0:
            self.ratioURL = collections.Counter(self.urls).most_common(1)[0][1]/len(self.urls)
        else:
            self.ratioURL = 1
    def __getRatioDevice(self):
        if len(self.devices) > 0:
            self.ratioDevices = collections.Counter(self.urls).most_common(1)[0][1]/len(self.devices)
        else:
            self.ratioDevices = 0
    
    def __getAvgBidsPerAuction(self):
        if len(self.bid_ids) > 0 and len(self.auctions) > 0:
            self.avg_bid_auc = len(pd.unique(self.bid_ids))/len(pd.unique(self.auctions))
        else:
            self.avg_bid_auc = 0
    
    def __medConsecutiveBidsTime(self):
        srt_time_split = sorted(self.times, reverse=True)
        if len(srt_time_split) > 1:
            self.fastest_consec_bid = np.min([(srt_time_split[x]-srt_time_split[x+1]) for x in range(len(srt_time_split[:-1]))])
            self.cons_bids_time = np.median([(srt_time_split[x]-srt_time_split[x+1]) for x in range(len(srt_time_split[:-1]))])
        else:
            self.fastest_consec_bid = sys.maxsize
            self.cons_bids_time = sys.maxsize
    
    def __getLastTMBids(self):
        sorted_time = sorted(self.times, reverse=True)
        max_d = 0
        for i in range(len(sorted_time)):
            d = 0
            for j in range(i, len(sorted_time)):
                if sorted_time[i] - sorted_time[j] <= 1200000:
                    d+=1
                    if d > max_d:
                        max_d = d
                else:
                    break
        self.lasttmbids = max_d
        
    def __getLastTHBids(self):
        sorted_time = sorted(self.times, reverse=True)
        max_d = 0
        for i in range(len(sorted_time)):
            d = 0
            for j in range(i, len(sorted_time)):
                if sorted_time[i] - sorted_time[j] <= 7200000:
                    d+=1
                    if d > max_d:
                        max_d = d
                else:
                    break
        self.lasthbids = max_d
        
    def to_dict(self):
        repr_dict = {}
        for k, v in self.__dict__.items():
            if type([]) != type(v) and type(np.asarray([])) != type(v):
                repr_dict[k] = v
        return repr_dict
    
    def __str__(self):
        return 'Bidder_id: %s, Total bids: %d' %(self.bidder_id, self.total_bids)


# In[ ]:


#Prepare training data
train_collection = {}
cnt = 0
err_cnt = 0
for utb in train_df['bidder_id'].values:
    #If we have bidder data
    if utb in unique_train_bidder:
        temp_bidder_meta = meta_train_df.loc[meta_train_df['bidder_id'] == utb]
        train_collection[utb] = BidderData(utb, temp_bidder_meta['bid_id'], temp_bidder_meta['country'], temp_bidder_meta['url'], temp_bidder_meta['auction'], temp_bidder_meta['merchandise'], temp_bidder_meta['device'], temp_bidder_meta['time'], temp_bidder_meta['ip'], outcome = list(temp_bidder_meta['outcome'])[0])
        cnt+=1
    #If we have no bidder data, initialize
    else:
        train_collection[utb] = BidderData(utb, [], [], [], [], [], [], [], [], outcome = 0)
        err_cnt += 1
    sys.stdout.write('\r%d : %d - %s' %(cnt, err_cnt, train_collection[utb]))


# In[ ]:


#Generating final training set
training_df_final = pd.DataFrame.from_records([dfs.to_dict() for dfs in train_collection.values() if not (dfs.total_bids == 1 and dfs.outcome == 1.0)])


# In[ ]:


print('Training features are: %s'%training_df_final.columns)


# In[ ]:


#Splitting data into train data and labels
training_df_fl = training_df_final[[col for col in training_df_final if col not in ['outcome', 'bidder_id']]]
training_df_flabel = training_df_final[[col for col in training_df_final if col in ['outcome']]]


# In[ ]:


#Applying neighborhood cleaning rule and preparing 1st phase model data
ncr = NeighbourhoodCleaningRule(n_neighbors=15, random_state=32, ratio={0:0.5})
training_df_X, training_df_y = ncr.fit_resample(training_df_fl, training_df_flabel.values.reshape(1, -1)[0])


# In[ ]:


#Creating class for containing different model executions
class ClassifierContainer:
    def __init__(self, model, training_X, training_y, measuring_parameter='auc'):
        self.model = model
        self.training_X = training_X
        self.training_y = training_y
        self.measuring_parameter = measuring_parameter
    
    #Predicting results
    def predict_get_results(self, n_splits):
        result_list = []
        #Kfold cross val
        kf = KFold(n_splits=n_splits, shuffle=True)
        for train_indices, validation_indices in kf.split(self.training_X):
            X_train, X_valid = self.training_X[train_indices], self.training_X[validation_indices]
            y_train, y_valid = self.training_y[train_indices], self.training_y[validation_indices]
            self.model.fit(X_train, y_train)
            pred_y = self.model.predict(X_valid)
            #Measures
            cm = confusion_matrix(y_valid, pred_y)
            acs = accuracy_score(y_valid, pred_y)
            ps = precision_score(y_valid, pred_y)
            rs = recall_score(y_valid, pred_y)
            f1_s = f1_score(y_valid, pred_y)
            fpr, tpr, thresholds = roc_curve(y_valid, pred_y)
            aucs = auc(fpr, tpr)
            result = {'pred_y':pred_y, 'cm':cm, 'acs':acs, 'ps':ps, 'rs':rs, 'f1_s':f1_s, 'auc': aucs, 'model':copy.deepcopy(self.model)}
            result_list.append(result)
        self.results = None
        for result in result_list:
            if not self.measuring_parameter in result:
                self.measuring_parameter = 'auc'
            if self.results == None or self.results[self.measuring_parameter] < result[self.measuring_parameter]:
                self.pred_y = result['pred_y']
                del result['pred_y']
                self.results = result
                self.best_model = result['model']
                
        return self.results
    
    #Get best model
    def get_Best_Instance(self):
        return self.best_model
    
    #Return prediction
    def get_prediction(self):
        return self.pred_y


# In[ ]:


#Gradient Booster
xbg_container = ClassifierContainer(GradientBoostingClassifier(), training_df_X, training_df_y)
print(xbg_container.predict_get_results(n_splits=4))


# In[ ]:


#Adaboost
adb_container = ClassifierContainer(AdaBoostClassifier(n_estimators=1500), training_df_X, training_df_y)
print(adb_container.predict_get_results(n_splits=4))


# In[ ]:


#RandomForest
rfc_container = ClassifierContainer(RandomForestClassifier(n_estimators=1500), training_df_X, training_df_y)
print(rfc_container.predict_get_results(n_splits=4))


# In[ ]:


#RandomForest
svm_container = ClassifierContainer(svm.SVC(gamma='scale'), training_df_X, training_df_y)
print(svm_container.predict_get_results(n_splits=4))


# In[ ]:


#X is located on outcome
x_loc = pd.DataFrame(training_df_flabel.values.reshape(1, -1)[0], columns=['outcome'])
x_loc = x_loc.loc[x_loc['outcome']==0.0]


# In[ ]:


#Purify and prepare data
bi_info = meta_train_df.loc[meta_train_df['bidder_id'] == unique_train_bidder[331]]
country_list = list(bi_info.to_dict()['country'].values())
for cnt in country_list:
    if type(cnt) == type(float) and math.isnan(cnt):
        print(collections.Counter(list(bids_df.loc[bids_df['ip'].str.startswith(str(bi_info['ip'].values[0])[:4]), 'country'])).most_common()[0][0])


# In[ ]:


#Check for null countries and replace them with nearest ips
null_countries = []
for tr_bd in bids_df.values:
    cnt = 0
    for x in tr_bd:
        if pd.isnull(x):
            null_countries.append((tr_bd, cnt))
        cnt+=1


# In[ ]:


print('Number countries with NaN values: %s'%len(null_countries))


# In[ ]:


#Get custom model split
def get_custom_split(training_df_X, training_df_y, n_splits = 5):
    #Train-Validation bot ids
    training_bot_ind = (training_df_y.loc[training_df_y['outcome'] == 1.0]).index.values
    validation_bot_ind = np.random.choice(training_bot_ind, size = int(0.3*len(training_bot_ind)), replace=False)
    training_bot_ind = np.setdiff1d(training_bot_ind, validation_bot_ind)
    training_ind_arr = []
    #Train-Validation human ids
    for i in range(n_splits):
        training_hum_ind = (training_df_y.loc[training_df_y['outcome'] == 0.0]).index.values
        validation_hum_ind = np.random.choice(training_hum_ind, size = int(0.3*len(training_hum_ind)), replace=False)
        training_hum_ind = np.setdiff1d(training_hum_ind, validation_hum_ind)
        training_ind = np.append(np.random.choice(training_hum_ind, size=int(len(training_hum_ind)/1.9)), training_bot_ind)
        np.random.shuffle(training_ind)
        training_ind_arr.append(training_ind)
    validation_ind = np.append(validation_bot_ind, validation_hum_ind)
    np.random.shuffle(validation_ind)
    return [training_df_X.iloc[training_ind_i] for training_ind_i in training_ind_arr], [training_df_y.iloc[training_ind_i] for training_ind_i in training_ind_arr], training_df_X.iloc[validation_ind], training_df_y.iloc[validation_ind]


# In[ ]:


#generate train-validation sets
train_df_X_set, train_df_y_set, validation_df_X, validation_df_y = get_custom_split(training_df_fl, training_df_flabel)


# In[ ]:


#Our model - Ensemble of ensemble
results = {'cm':[], 'acs':[],'ps':[], 'rs':[], 'f1_s':[], 'auc':[]}
for i in range(len(train_df_X_set)):
    train_df_y_transpose = train_df_y_set[i].values.reshape(1,-1)[0]
    ncr = NeighbourhoodCleaningRule(n_neighbors=1, random_state=32)
    train_df_X_i, train_df_y_i = ncr.fit_resample(train_df_X_set[i], train_df_y_transpose)
    rfc1_y = ((RandomForestClassifier(n_estimators=1500)).fit(train_df_X_i, train_df_y_i)).predict(validation_df_X)
    xgb1_y = ((GradientBoostingClassifier(n_estimators=1500)).fit(train_df_X_i, train_df_y_i)).predict(validation_df_X)
    adb1_y = ((AdaBoostClassifier(n_estimators=1500, random_state=42, learning_rate=0.098)).fit(train_df_X_i, train_df_y_i)).predict(validation_df_X)
    svm1_y = ((svm.SVC(gamma='scale')).fit(train_df_X_set[i], train_df_y_set[i])).predict(validation_df_X)
    rfc2_y = ((RandomForestClassifier(n_estimators=1500)).fit(train_df_X_set[i], train_df_y_set[i])).predict(validation_df_X)
    xgb2_y = ((GradientBoostingClassifier(n_estimators=1500)).fit(train_df_X_set[i], train_df_y_set[i])).predict(validation_df_X)
    adb2_y = ((AdaBoostClassifier(n_estimators=1500, random_state=42, learning_rate=0.098)).fit(train_df_X_set[i], train_df_y_set[i])).predict(validation_df_X)
    
    gen_y = []
    for i in range(len(validation_df_y['outcome'])):
        f1 = (0.48*svm1_y[i])+(0.27*rfc1_y[i])+(0.25*adb1_y[i])
        f2 = (0.3*rfc2_y[i])+(0.35*xgb2_y[i])+(0.35*adb2_y[i])
        gen_y.append(round(max(f1, f2)))
    results['cm'].append(confusion_matrix(gen_y, validation_df_y['outcome']))
    results['acs'].append(accuracy_score(gen_y, validation_df_y['outcome']))
    results['ps'].append(precision_score(gen_y, validation_df_y['outcome']))
    results['rs'].append(recall_score(gen_y, validation_df_y['outcome']))
    results['f1_s'].append(f1_score(gen_y, validation_df_y['outcome']))
    fpr, tpr, thresholds = roc_curve(gen_y, validation_df_y['outcome'])
    results['auc'].append(auc(fpr, tpr))
results_custom_pd = pd.DataFrame.from_dict(results)


# In[ ]:


#Outputting report
print(results_custom_pd)


# In[ ]:


#Compare xbgs
results = {}
xbg_container_0 = ClassifierContainer(GradientBoostingClassifier(), training_df_X, training_df_y)
results['Vanilla'] = (xbg_container_0.predict_get_results(n_splits=4))['auc']
xbg_container_1 = ClassifierContainer(GradientBoostingClassifier(max_features=5), training_df_X, training_df_y)
results['m_feat 5'] = (xbg_container_1.predict_get_results(n_splits=4))['auc']
xbg_container_2 = ClassifierContainer(GradientBoostingClassifier(learning_rate=0.01), training_df_X, training_df_y)
results['l_rate 0.01'] = (xbg_container_2.predict_get_results(n_splits=4))['auc']
xbg_container_3 = ClassifierContainer(GradientBoostingClassifier(max_depth=15), training_df_X, training_df_y)
results['m_depth 15'] = (xbg_container_3.predict_get_results(n_splits=4))['auc']
xbg_container_4 = ClassifierContainer(GradientBoostingClassifier(learning_rate=0.15, max_features=10), training_df_X, training_df_y)
results['l_rate 0.15\nm_feat 10'] = (xbg_container_4.predict_get_results(n_splits=4))['auc']

res_xbg_df = pd.DataFrame([(k, v) for k, v in results.items()], columns=['Title', 'AUC'])
print(res_xbg_df)
sns.barplot(x='Title', y='AUC', data=res_xbg_df)


# In[ ]:


#Compare adaboost classifier
results = {}
abd_container_0 = ClassifierContainer(AdaBoostClassifier(), training_df_X, training_df_y)
results['Vanilla'] = (abd_container_0.predict_get_results(n_splits=4))['auc']
abd_container_1 = ClassifierContainer(AdaBoostClassifier(n_estimators=1000), training_df_X, training_df_y)
results['n_estim 5'] = (abd_container_1.predict_get_results(n_splits=4))['auc']
abd_container_2 = ClassifierContainer(AdaBoostClassifier(learning_rate=0.01), training_df_X, training_df_y)
results['l_rate 0.01'] = (abd_container_2.predict_get_results(n_splits=4))['auc']
abd_container_3 = ClassifierContainer(AdaBoostClassifier(random_state=15), training_df_X, training_df_y)
results['m_depth 15'] = (abd_container_3.predict_get_results(n_splits=4))['auc']
abd_container_4 = ClassifierContainer(AdaBoostClassifier(learning_rate=0.15, n_estimators=1500), training_df_X, training_df_y)
results['l_rate 0.15\nn_estim 1500'] = (abd_container_4.predict_get_results(n_splits=4))['auc']

res_abd_df = pd.DataFrame([(k, v) for k, v in results.items()], columns=['Title', 'AUC'])
print(res_abd_df)
sns.barplot(x='Title', y='AUC', data=res_abd_df)


# In[ ]:


#Compare randomforest
results = {}
rfc_container_0 = ClassifierContainer(RandomForestClassifier(), training_df_X, training_df_y)
results['Vanilla'] = (rfc_container_0.predict_get_results(n_splits=4))['auc']
rfc_container_1 = ClassifierContainer(RandomForestClassifier(max_features=5), training_df_X, training_df_y)
results['m_feat 5'] = (rfc_container_1.predict_get_results(n_splits=4))['auc']
rfc_container_2 = ClassifierContainer(RandomForestClassifier(max_depth=10), training_df_X, training_df_y)
results['l_depth 10'] = (rfc_container_2.predict_get_results(n_splits=4))['auc']
rfc_container_3 = ClassifierContainer(RandomForestClassifier(class_weight={0.0:0.2}), training_df_X, training_df_y)
results['c_weight 0.1'] = (rfc_container_3.predict_get_results(n_splits=4))['auc']
rfc_container_4 = ClassifierContainer(RandomForestClassifier(max_features=10, max_depth=10), training_df_X, training_df_y)
results['m_feat 10\nm_depth 10'] = (rfc_container_4.predict_get_results(n_splits=4))['auc']

res_rfc_df = pd.DataFrame([(k, v) for k, v in results.items()], columns=['Title', 'AUC'])
print(res_rfc_df)
sns.barplot(x='Title', y='AUC', data=res_rfc_df)


# In[ ]:


#featwise stats
results = {}
for col in range(training_df_X.shape[1]):
    xbg_container_0 = ClassifierContainer(GradientBoostingClassifier(), training_df_X[:,col].reshape(-1, 1), training_df_y)
    results[(training_df_fl.columns)[col]] = (xbg_container_0.predict_get_results(n_splits=4))['auc']
print(results)


# In[ ]:


#Plot Feature and AUC
results_out = [(k, v) for k,v in results.items()]
results_out.sort(key=lambda x: x[1], reverse=True)
sns.barplot(x='Feature', y='AUC', data=pd.DataFrame(results_out[:4], columns=['Feature', 'AUC']))
print(results_out)


# In[ ]:


#Test prediction
test_collection = {}
cnt = 0
err_cnt = 0
for utb in test_df['bidder_id'].values:
    if utb in unique_test_bidder:
        temp_bidder_meta = meta_test_df.loc[meta_test_df['bidder_id'] == utb]
        test_collection[utb] = BidderData(utb, temp_bidder_meta['bid_id'], temp_bidder_meta['country'], temp_bidder_meta['url'], temp_bidder_meta['auction'], temp_bidder_meta['merchandise'], temp_bidder_meta['device'], temp_bidder_meta['time'], temp_bidder_meta['ip'])
        cnt+=1
    else:
        test_collection[utb] = BidderData(utb, [], [], [], [], [], [], [], [])
        err_cnt += 1
    sys.stdout.write('\r%d : %d - %s' %(cnt, err_cnt, test_collection[utb]))


# In[ ]:


testing_df_final = pd.DataFrame.from_records([dfs.to_dict() for dfs in test_collection.values()])


# In[ ]:


testing_df_final = testing_df_final[[col for col in testing_df_final if col not in ['bidder_id']]]


# In[ ]:


testing_df_final.columns


# In[ ]:


with open('dataset/format.txt', 'w+') as writable_file:
    adb_container = ClassifierContainer(GradientBoostingClassifier(), training_df_X, training_df_y)
    prediction_test = adb_container.predict_get_results(n_splits=4)
    test_preds = prediction_test['model'].predict(testing_df_final)
    for a in test_preds:
        writable_file.write('%d\n'%a)
    writable_file.close()

