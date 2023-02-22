from dataloader import get_loader
from sklearn.decomposition import PCA
import numpy as np

def get_singular(data):
    pca = PCA(svd_solver='full')
    pca.fit(X=data)
    return pca.singular_values_

def get_keep_num(singular,keep_list):
    eigenvalue = np.array(sorted((singular**2),reverse=True))
    print(eigenvalue)
    cumsum_eigen = eigenvalue.cumsum(axis=0)
    print(cumsum_eigen)
    singular_sum = sum(eigenvalue)
    print(singular_sum)
    i = 1

    keep_num_list = []
    for keep in keep_list:
        for i in range(i,singular.shape[0]):
            if cumsum_eigen[i]/singular_sum >= keep:
                keep_num_list.append(i+2)
                break
    return keep_num_list

def get_pca_feat(data,test_data):
    bs,C,H,W = data.shape
    bs2 = test_data.shape[0]
    data = data.cpu().view(bs,-1).contiguous().numpy()
    test_data = test_data.cpu().view(bs2,-1).contiguous().numpy()
    singular = get_singular(data)

    keep_list = [0.4,0.6,0.8]
    keep_num_list = get_keep_num(singular,keep_list)
    # print(keep_num_list)

    reducted_data = []
    test_data_list = []
    for num in keep_num_list:
        pca = PCA(n_components=num).fit(X=data)
        data_new = pca.transform(data)
        data_new = np.concatenate((data_new,np.ones((data_new.shape[0],1))),axis=1)
        test_data_new = pca.transform(test_data)
        test_data_new = np.concatenate((test_data_new,np.ones((test_data_new.shape[0],1))),axis=1)
        reducted_data.append(data_new)
        test_data_list.append(test_data_new)
    reducted_data.append(np.concatenate((data,np.ones((data.shape[0],1))),axis=1))
    test_data_list.append(np.concatenate((test_data,np.ones((test_data.shape[0],1))),axis=1))
    keep_num_list.append(C*H*W+1)

    print([data.shape for data in reducted_data])
    print([test_data.shape for test_data in test_data_list])
    return reducted_data,test_data_list,keep_num_list

def main():
    trainloader, testloader = get_loader(normalize_tag=False,bs='full')
    # trainloader, testloader = get_loader(normalize_tag=False,bs=32)

    data,label = next(iter(trainloader))
    test_data,test_label = next(iter(testloader))
    label = label.cpu().numpy()
    test_label = test_label.cpu().numpy()
    return get_pca_feat(data,test_data),label,test_label

    


if __name__ == "__main__":
    main()
        
