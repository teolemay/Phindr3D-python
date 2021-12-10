"""
Teophile Lemay, 2021

Clustering functions for Phindr3d translated into python
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import time

eps = np.finfo(np.float64).eps
realmin = np.finfo(np.float64).tiny
realmax = np.finfo(np.float64).max

def apcluster(s, p, sparse=False, maxits=500, convits=50, dampfact=0.5, plot=True, details=False, nonoise=False):
    """in third party/clustering"""
    """
    s = similarities
    p = preferences

    % APCLUSTER uses affinity propagation (Frey and Dueck, Science,
    % 2007) to identify data clusters, using a set of real-valued
    % pair-wise data point similarities as input. Each cluster is
    % represented by a data point called a cluster center, and the
    % method searches for clusters so as to maximize a fitness
    % function called net similarity. The method is iterative and
    % stops after maxits iterations (default of 500 - see below for
    % how to change this value) or when the cluster centers stay
    % constant for convits iterations (default of 50). The command
    % apcluster(s,p,'plot') can be used to plot the net similarity
    % during operation of the algorithm.
    %
    % For N data points, there may be as many as N^2-N pair-wise
    % similarities (note that the similarity of data point i to k
    % need not be equal to the similarity of data point k to i).
    % These may be passed to APCLUSTER in an NxN matrix s, where
    % s(i,k) is the similarity of point i to point k. In fact, only
    % a smaller number of relevant similarities are needed for
    % APCLUSTER to work. If only M similarity values are known,
    % where M < N^2-N, they can be passed to APCLUSTER in an Mx3
    % matrix s, where each row of s contains a pair of data point
    % indices and a corresponding similarity value: s(j,3) is the
    % similarity of data point s(j,1) to data point s(j,2).
    %
    % APCLUSTER automatically determines the number of clusters,
    % based on the input p, which is an Nx1 matrix of real numbers
    % called preferences. p(i) indicates the preference that data
    % point i be chosen as a cluster center. A good choice is to 
    % set all preference values to the median of the similarity
    % values. The number of identified clusters can be increased or
    % decreased  by changing this value accordingly. If p is a
    % scalar, APCLUSTER assumes all preferences are equal to p.
    %
    % The fitness function (net similarity) used to search for
    % solutions equals the sum of the preferences of the the data
    % centers plus the sum of the similarities of the other data
    % points to their data centers.
    %
    % The identified cluster centers and the assignments of other
    % data points to these centers are returned in idx. idx(j) is
    % the index of the data point that is the cluster center for
    % data point j. If idx(j) equals j, then point j is itself a
    % cluster center. The sum of the similarities of the data
    % points to their cluster centers is returned in dpsim, the
    % sum of the preferences of the identified cluster centers is
    % returned in expref and the net similarity (sum of the data
    % point similarities and preferences) is returned in netsim.
    %
    % EXAMPLE
    %
    % N=100; x=rand(N,2); % Create N, 2-D data points
    % M=N*N-N; s=zeros(M,3); % Make ALL N^2-N similarities
    % j=1;
    % for i=1:N
    %   for k=[1:i-1,i+1:N]
    %     s(j,1)=i; s(j,2)=k; s(j,3)=-sum((x(i,:)-x(k,:)).^2);
    %     j=j+1;
    %   end;
    % end;
    % p=median(s(:,3)); % Set preference to median similarity
    % [idx,netsim,dpsim,expref]=apcluster(s,p,'plot');
    % fprintf('Number of clusters: %d\n',length(unique(idx)));
    % fprintf('Fitness (net similarity): %f\n',netsim);
    % figure; % Make a figures showing the data and the clusters
    % for i=unique(idx)'
    %   ii=find(idx==i); h=plot(x(ii,1),x(ii,2),'o'); hold on;
    %   col=rand(1,3); set(h,'Color',col,'MarkerFaceColor',col);
    %   xi1=x(i,1)*ones(size(ii)); xi2=x(i,2)*ones(size(ii)); 
    %   line([x(ii,1),xi1]',[x(ii,2),xi2]','Color',col);
    % end;
    % axis equal tight;
    %
    % PARAMETERS
    % 
    % [idx,netsim,dpsim,expref]=apcluster(s,p,'NAME',VALUE,...)
    % 
    % The following parameters can be set by providing name-value
    % pairs, eg, apcluster(s,p,'maxits',1000):
    %
    %   Parameter    Value
    %   'sparse'     No value needed. Use when the number of data
    %                points is large (eg, >3000). Normally,
    %                APCLUSTER passes messages between every pair
    %                of data points. This flag causes APCLUSTER
    %                to pass messages between pairs of points only
    %                if their input similarity is provided and
    %                is not equal to -Inf.
    %   'maxits'     Any positive integer. This specifies the
    %                maximum number of iterations performed by
    %                affinity propagation. Default: 500.
    %   'convits'    Any positive integer. APCLUSTER decides that
    %                the algorithm has converged if the estimated
    %                cluster centers stay fixed for convits
    %                iterations. Increase this value to apply a
    %                more stringent convergence test. Default: 50.
    %   'dampfact'   A real number that is less than 1 and
    %                greater than or equal to 0.5. This sets the
    %                damping level of the message-passing method,
    %                where values close to 1 correspond to heavy
    %                damping which may be needed if oscillations
    %                occur.
    %   'plot'       No value needed. This creates a figure that
    %                plots the net similarity after each iteration
    %                of the method. If the net similarity fails to
    %                converge, consider increasing the values of
    %                dampfact and maxits.
    %   'details'    No value needed. This causes idx, netsim,
    %                dpsim and expref to be stored after each
    %                iteration.
    %   'nonoise'    No value needed. Degenerate input similarities
    %                (eg, where the similarity of i to k equals the
    %                similarity of k to i) can prevent convergence.
    %                To avoid this, APCLUSTER adds a small amount
    %                of noise to the input similarities. This flag
    %                turns off the addition of noise.
    %
    % Copyright (c) Brendan J. Frey and Delbert Dueck (2006). This
    % software may be freely used and distributed for
    % non-commercial purposes.
    """
    ##Global R, A, E, tmpidx, tmpnetsim, S #These get define later down, BUT maybe only behind an if statement.
    if sparse == True:
        return apcluster_sparse(s, p, maxits=maxits, convits=convits, dampfact=dampfact, plot=plot, details=details, nonoise=nonoise)
    maxits = int(maxits)
    if maxits <= 0:
        print('maxits must be positve integer')
        return None
    convits = int(convits)
    if convits <= 0:
        print('convits must be positive integer')
        return None
    lam = dampfact
    if (lam < 0.5) or (lam>=1):
        print('dampfact must be in the range [0.5, 1)')
        return None
    if lam > 0.9:
        print('\nLarge damping factor selected, plotting is recommended, Algorithm will also change decisions slowly so large convits should be set as well.\n')
    if len(s.shape) != 2:
        print('s should be 2d matrix')
        return None
    elif len(p.shape) > 2:
        print('p should be vector or scalar')
    elif s.shape[1] == 3:
        tmp = np.maximum(np.max(s[:, 0]), np.max(s[:, 1]))
        if len(p) == 1:
            N=tmp
        else:
            N=len(p)
        if tmp > N:
            print('Error, data point index exceeds number of datapoints')
            return None 
        elif np.minimum(np.min(s[:, 0]), np.min(s[:, 1])) < 0:
            print('Error, indices must be >= 0')
            return None
    else:
        print('Error, s must have 3 columns or be square.')
        return None 
    #construct similarity matrix
    if N > 3000:
        print('\nLarge memory request, consider setting sparse=True\n')
    if s.shape[1] == 3:
        S = np.full((N, N), -np.inf)
        for j in range(0, s.shape[0]):
            S[s[j, 0], s[j, 1]] = s[j, 3]
    else:
        S = s
    #in case user did not remove degeneracies from input similarities, avoid degenerate solutions by adding small noise to input similarities
    if not nonoise:
        rns = np.random.get_state()
        np.random.seed(0)
        S = S + (eps*S + realmin*100)*np.random.random((N,N))
        np.random.set_state(rns)
    #place preference on diagonal of S
    if len(p)==1:
        for i in range(N):
            S[i,i] = p
    else:
        for i in range(N):
            S[i, i] = p[i]
    #place for messages, etc:
    dS = np.diag(S)
    A = np.zeros((N, N))
    R = np.zeros((N, N))
    t = 1
    if plot:
        netsim=np.zeros((1, maxits+1))
    if details:
        idx = np.zeros((N, maxits+1))
        netsim = np.zeros((N, maxits+1))
        dpsim = np.zeros((N, maxits+1))
        expref = np.zeros((N, maxits+1))
    
    #execute parallel affinity propagation updates!
    e = np.zeros((N, convits))
    dn=False
    i=0
    while not dn:
        i += 1
        #compute responsibilities
        Rold=R
        AS = A + S
        Y = np.amax(AS, axis=1)
        I = np.argmax(AS, axis=1) #have to remember that I is along axis 1
        for k in range(N):
            AS[k, I][k] = -realmax
        Y2 = np.amax(AS, axis=1)
        I2 = np.argmax(AS, axis=2)
        R = S-np.tile(Y, [1, N])
        for k in range(N):
            R[k, I(k)] = S[k, I[k]] - Y2[k]
        #damping
        R = (1-lam)*R + lam*Rold

        #compute availabilities
        Aold = A
        Rp = np.maximum(R, 0)
        for k in range(N):
            Rp[k, k]=R[k, k]
        A = np.tile(np.sum(Rp, axis=0), [N, 1]) - Rp
        dA = np.diag(A)
        A = np.minimum(A, 0)
        for k in range(N):
            A[k,k] = dA[k]
        #damping
        A = (1-lam)*A + lam*Aold  

        #check for convergence
        E = ((np.diag(A) + np.diag(R)) > 0)
        e[:, (i+1)%convits+1]=E
        K=np.sum(E) #I think E is a vector so sum should work properly
        if i>=convits or i>=maxits:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se==convits)+(se==0))!= N)
            if (not unconverged and (K>0)) or (i == maxits):
                dn=True
        if plot or details:
            if K == 0:
                tmpnetsim=np.nan
                tmpdpsim=np.nan
                tmpexpref=np.nan
                tmpidx=np.nan
            else:
                I = np.nonzero(E)
                tmp = np.amax(S[:, I], axis=1)
                c = np.argmax(S[:, I], axis=1)
                c[I] = np.arange(0, K)
                tmpidx = I[c]
                tmpnetsim = np.sum(S[(tmpidx-1)*N + np.arange[0, N].T], axis=0) #might not need axis here
                tmpexpref = np.sum(dS[I])
                tmpdpsim = tmpnetsim - tmpexpref
        if details:
            netsim[i] = tmpnetsim
            dpsim[i] = tmpdpsim
            expref[i] = tmpexpref
            idx[:, i] = tmpidx
        if plot:
            netsim[i] = tmpnetsim
            plt.figure()
            tmp = np.arange(0,i)
            tmpi=np.nonzero(np.isfinite(netsim[:i]))
            plt.plot(tmp[tmpi], netsim[tmpi], 'r--')
            plt.xlabel('Number of iterations')
            plt.ylabel('Fitness (net similarity) of quantized intermediate solution')
            plt.show()
            time.sleep(2)
            plt.close()
    #identify exemplars
    I = np.nonzero(np.diag(A+R)>0)
    K = len(I)
    if K > 0:
        #identify clusters
        tmp = np.amax(S[:, I], axis=1)
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(0, K)
        #refine the final set of exemplars and clusters and return results
        for k in range(K):
            ii = np.nonzero(c==k)
            y = np.amax(S[ii, ii], axis=0)
            j = np.argmax(S[ii, ii], axis=0)
            I[k] = ii[j[0]]
        tmp = np.amax(S[:, I], axis=1)
        c = np.argmax(S[:, I], axis=1)
        c[I] = np.arange(0, K)
        tmpidx = I[c]
    else:
        tmpidx = np.full((N, 1), np.nan)
        tmpnetsim = np.nan
        tmpexpref = np.nan
    if details:
        netsim[i+1] = tmpnetsim
        netsim = netsim[:i+1]
        dpsim[i+1] = tmpnetsim - tmpexpref
        dpsim = dpsim[:i+1]
        expref[i+1] = tmpexpref
        expref = expref[:i+1]
        idx[:, i+1] = tmpidx
        idx = idx[:, :i+1]
    else:
        netsim = tmpnetsim
        dpsim = tmpnetsim - tmpexpref
        expref = tmpexpref
        idx = tmpidx
    if plot or details:
        print(f'number of identified clusters: {K}')
        print(f'Fitness (net similarity): {tmpnetsim}')
        print(f'\t similarities of data points to exemplars: {dpsim[-1]}')
        print(f'\t preferences of selected exemplars: {tmpexpref}')
        print(f'number of itereations: {i}\n')
    if unconverged:
        print('algorithm did not converge, similarities may contain degeneracies - add noise to similarities to remove degeneracies.')
        print('To monitor net similarity, activate plotting. Also consider increasing maxits and if necessary, dampfact.')
    return idx, netsim, dpsim, expref, unconverged

def preferenceRange(sim):
    """called in clsIn"""
    """in third party/clustering folder"""
    return None

def apclusterK(s, kk, prc=10):
    """called in computeClustering"""
    """in third party/clustering folder"""
    """
    % Finds approximately k clusters using affinity propagation (BJ Frey and
    % D Dueck, Science 2007), by searching for an appropriate preference value
    % using a bisection method. By default, the method stops refining the
    % number of clusters when it is within 10%% of the value k provided by the
    % user. To change this percentage, use apclusterK(s,k,prc) -- eg, setting
    % prc to 0 causes the method to search for exactly k clusters. In any case
    % the method terminates after 20 bisections are attempted.
    %
    """
    # Construct similarity matrix and add a tiny amount of noise
    if s.shape[1] == 3:
        N = np.maximum(np.max(s[:, 0]), np.max(s[:, 1]))
        S = np.full((N, N), -np.inf)
        for j in range(s.shape[0]):
            S[s[j, 0]] = s[j,2]
            S[s[j, 1]] = s[j,2]
    else:
        N = s.shape[0]
        S = s
    rns = np.random.get_state()
    np.random.seed(0)
    S = S + (eps*S + realmin*100)*np.random.random((N,N))
    np.random.set_state(rns)
    #assigning base, S, S
    for k in range(N):
        S[k,k] = 0
    #find limits
    dpsim1 = np.max(np.sum(S, axis=0))
    k11 = np.unravel_index(np.argmax(np.sum(S, axis=0)), shape=np.sum(S, axis=0).shape)
    if dpsim1 == -np.inf:
        print('error, could not find pmin')
        return None
    elif N>1000:
        for k in range(N):
            S[k,k] = -np.inf
        m = np.amax(S, axis=1)
        tmp = np.sum(m)
        yy = np.amin(m, axis=0)
        ii = np.argmin(m, axis=0)
        tmp = tmp - yy - np.min(m[:ii[1]+1, ii[1]+1:N])
        pmin = dpsim1-tmp
    else:
        dpsim2 = -np.inf
        for j21 in range(N-1):
            for j22 in range(j21, N):
                tmp = np.sum(np.max(S[:, j21, j22], axis=2)) #this indexing is very awkckward to do in python
                if tmp > dpsim2:
                    dpsim2 = tmp 
                    k21 = j21 
                    k22 = j22 
        pmin = dpsim1 - dpsim2
    for k in range(N):
        S[k,k] = -np.inf
    pmax = np.max(S)
    highpref = pmax
    highk = N
    lowpref = pmin
    lowk=0
    for k in range(N):
        s[k,k] = 0
    #run AP several times to find lower bound:
    i=-4
    dn=False
    while not dn:
        tmppref = highpref - (10**i) * (highpref-lowpref)
        idx, netsim, dpsim, expref, unconverged = apcluster(S, tmppref, dampfact=0.9, convits=50, maxits=1000)
        tmpk = len(np.unique(idx))
        if tmpk <= kk:
            dn=True
        elif i == 1:
            tmpk = lowk
            tmppref = lowpref
            dn=True
        else:
            i += 1
    #use bisection method to find k
    if np.abs(tmpk-kk)/kk*100 > prc:
        print(f'applyng bisection method')
        lowk = tmpk
        lowpref=tmppref
        ntries = 0
        while (np.abs(tmpk-kk)/kk*100 > prc) and (ntries < 20):
            tmppref = 0.5*highpref + 0.5*lowpref
            idx, netsim, dpsim, expref, unconverged = apcluster(S, tmppref, dampfact=0.9, convits=50, maxits=1000)
            tmpk = len(np.unique(idx))
            if kk > tmpk:
                lowpref = tmppref
                lowk = tmpk
            else:
                highpref = tmppref
                highk = tmpk
            ntries += 1
    pref = tmppref
    print(f'Found {tmpk} clusters using a preference of {pref}')
    return idx, netsim, dpsim, expref, pref

def apcluster_sparse(s, p, maxits=500, convits=50, dampfact=0.5, plot=False, details=False, nonoise=False):
    """
    %
    % APCLUSTER uses affinity propagation (Frey and Dueck, Science,
    % 2007) to identify data clusters, using a set of real-valued
    % pair-wise data point similarities as input. Each cluster is
    % represented by a data point called a cluster center, and the
    % method searches for clusters so as to maximize a fitness
    % function called net similarity. The method is iterative and
    % stops after maxits iterations (default of 500 - see below for
    % how to change this value) or when the cluster centers stay
    % constant for convits iterations (default of 50). The command
    % apcluster(s,p,'plot') can be used to plot the net similarity
    % during operation of the algorithm.
    %
    % For N data points, there may be as many as N^2-N pair-wise
    % similarities (note that the similarity of data point i to k
    % need not be equal to the similarity of data point k to i).
    % These may be passed to APCLUSTER in an NxN matrix s, where
    % s(i,k) is the similarity of point i to point k. In fact, only
    % a smaller number of relevant similarities are needed for
    % APCLUSTER to work. If only M similarity values are known,
    % where M < N^2-N, they can be passed to APCLUSTER in an Mx3
    % matrix s, where each row of s contains a pair of data point
    % indices and a corresponding similarity value: s(j,3) is the
    % similarity of data point s(j,1) to data point s(j,2).
    %
    % APCLUSTER automatically determines the number of clusters,
    % based on the input p, which is an Nx1 matrix of real numbers
    % called preferences. p(i) indicates the preference that data
    % point i be chosen as a cluster center. A good choice is to 
    % set all preference values to the median of the similarity
    % values. The number of identified clusters can be increased or
    % decreased  by changing this value accordingly. If p is a
    % scalar, APCLUSTER assumes all preferences are equal to p.
    %
    % The fitness function (net similarity) used to search for
    % solutions equals the sum of the preferences of the the data
    % centers plus the sum of the similarities of the other data
    % points to their data centers.
    %
    % The identified cluster centers and the assignments of other
    % data points to these centers are returned in idx. idx(j) is
    % the index of the data point that is the cluster center for
    % data point j. If idx(j) equals j, then point j is itself a
    % cluster center. The sum of the similarities of the data
    % points to their cluster centers is returned in dpsim, the
    % sum of the preferences of the identified cluster centers is
    % returned in expref and the net similarity (sum of the data
    % point similarities and preferences) is returned in netsim.
    %
    % EXAMPLE
    %
    % N=100; x=rand(N,2); % Create N, 2-D data points
    % M=N*N-N; s=zeros(M,3); % Make ALL N^2-N similarities
    % j=1;
    % for i=1:N
    %   for k=[1:i-1,i+1:N]
    %     s(j,1)=i; s(j,2)=k; s(j,3)=-sum((x(i,:)-x(k,:)).^2);
    %     j=j+1;
    %   end;
    % end;
    % p=median(s(:,3)); % Set preference to median similarity
    % [idx,netsim,dpsim,expref]=apcluster(s,p,'plot');
    % fprintf('Number of clusters: %d\n',length(unique(idx)));
    % fprintf('Fitness (net similarity): %f\n',netsim);
    % figure; % Make a figures showing the data and the clusters
    % for i=unique(idx)'
    %   ii=find(idx==i); h=plot(x(ii,1),x(ii,2),'o'); hold on;
    %   col=rand(1,3); set(h,'Color',col,'MarkerFaceColor',col);
    %   xi1=x(i,1)*ones(size(ii)); xi2=x(i,2)*ones(size(ii)); 
    %   line([x(ii,1),xi1]',[x(ii,2),xi2]','Color',col);
    % end;
    % axis equal tight;
    %
    % PARAMETERS
    % 
    % [idx,netsim,dpsim,expref]=apcluster(s,p,'NAME',VALUE,...)
    % 
    % The following parameters can be set by providing name-value
    % pairs, eg, apcluster(s,p,'maxits',1000):
    %
    %   Parameter    Value
    %   'sparse'     No value needed. Use when the number of data
    %                points is large (eg, >3000). Normally,
    %                APCLUSTER passes messages between every pair
    %                of data points. This flag causes APCLUSTER
    %                to pass messages between pairs of points only
    %                if their input similarity is provided and
    %                is not equal to -Inf.
    %   'maxits'     Any positive integer. This specifies the
    %                maximum number of iterations performed by
    %                affinity propagation. Default: 500.
    %   'convits'    Any positive integer. APCLUSTER decides that
    %                the algorithm has converged if the estimated
    %                cluster centers stay fixed for convits
    %                iterations. Increase this value to apply a
    %                more stringent convergence test. Default: 50.
    %   'dampfact'   A real number that is less than 1 and
    %                greater than or equal to 0.5. This sets the
    %                damping level of the message-passing method,
    %                where values close to 1 correspond to heavy
    %                damping which may be needed if oscillations
    %                occur.
    %   'plot'       No value needed. This creates a figure that
    %                plots the net similarity after each iteration
    %                of the method. If the net similarity fails to
    %                converge, consider increasing the values of
    %                dampfact and maxits.
    %   'details'    No value needed. This causes idx, netsim,
    %                dpsim and expref to be stored after each
    %                iteration.
    %   'nonoise'    No value needed. Degenerate input similarities
    %                (eg, where the similarity of i to k equals the
    %                similarity of k to i) can prevent convergence.
    %                To avoid this, APCLUSTER adds a small amount
    %                of noise to the input similarities. This flag
    %                turns off the addition of noise.
    %
    % Copyright (c) Brendan J. Frey and Delbert Dueck (2006). This
    % software may be freely used and distributed for
    % non-commercial purposes.
    """
    maxits = int(maxits)
    if maxits <= 0:
        print('maxits must be positve integer')
        return None
    convits = int(convits)
    if convits <= 0:
        print('convits must be positive integer')
        return None
    lam = dampfact
    if (lam < 0.5) or (lam>=1):
        print('dampfact must be in the range [0.5, 1)')
        return None
    if lam > 0.9:
        print('\nLarge damping factor selected, plotting is recommended, Algorithm will also change decisions slowly so large convits should be set as well.\n')
    if len(s.shape) != 2:
        print('s should be 2d matrix')
        return None
    elif len(p.shape) > 2:
        print('p should be vector or scalar')
    elif s.shape[1] == 3:
        tmp = np.maximum(np.max(s[:, 0]), np.max(s[:, 1]))
        if len(p) == 1:
            N=tmp
        else:
            N=len(p)
        if tmp > N:
            print('Error, data point index exceeds number of datapoints')
            return None 
        elif np.minimum(np.min(s[:, 0]), np.min(s[:, 1])) < 0:
            print('Error, indices must be >= 0')
            return None
    else:
        print('Error, s must have 3 columns or be square.')
        return None 
    
    #make vector of preferences:
    if len(p) == 1:
        p = p*np.ones(N, 1)
    #append any self-similarities (preferences) to s-matrix
    tmps = np.concatenate((np.tile(np.arange(0, N).T, [0, 1]), p))
    s = np.concatenate((s, tmps))
    M =  s.shape[0]
    if not nonoise:
        rns = np.random.get_state()
        np.random.seed(0)
        s[:, 2] = s[:, 2] + (eps*s[:, 2] + realmin*100)*np.random.random((M,1))
        np.random.set_state(rns)
    #construct indices of neighbors:
    ind1e = np.zeros((N, 1))
    for j in range(M):
        k = s[j, 0]
        ind1e[k]=ind1e[k]+1
    ind1e = np.sum(ind1e)
    ind1s = np.concatenate((1, ind1e[:-1]+1))
    ind1 = np.zeros(M, 1)
    for j in range(M):
        k = s[j, 0]
        ind1[ind1s[k]]=j
        ind1s[k] = ind1s[k] + 1
    ind1s = np.concatenate((1, ind1e[:-1]+1))
    ind2e = np.zeros(N, 1)
    for j in range(M):
        k=s[j,1]
        ind2e[k]=ind2e[k]+1
    ind2e=np.sum(ind2e)
    ind2s=np.concatenate((1, ind2e[:-1]+1))
    ind2 = np.zeros((M, 1))
    for j in range(M):
        k=s[j,1]
        ind2[ind2s[k]]=j
        ind2s[k]=ind2s[k]+1
    ind2s = np.concatenate((1, ind2e[:-1]+1))
    #allocate space for messages, etc:
    A = np.zeros((M, 1))
    R = np.zeros((M, 1))
    t=1
    if plot:
        netsim=np.zeros((1, maxits+1))
    if details:
        idx=np.zeros((N, maxits+1))
        netsim=np.zeros((N, maxits+1))
        dpsim=np.zeros((N, maxits+1))
        expref=np.zeros((N, maxits+1))
    #execute parallel affinity propagation updates:
    e = np.zeros((N, convits))
    dn=False
    i=0
    while not dn:
        i += 1 
        #compute responsibilities:
        for j in range(N):
            ss = s[ind1s[j]:ind1e[j], 2]
            As = A[ind1s[j]:ind1e[j]] + ss
            Y = np.amax(As, axis=0)
            I = np.argmin(As, axis=0)
            As[I] = -realmax
            Y2 = np.amax(As, axis=0)
            I2 = np.argmax(As, axis=0)
            r = ss-Y
            r[I] = ss[I] - Y2
            R[ind1[ind1s[j]:ind1e[j]]] = (1-lam)*r + lam*R[ind1[ind1s[j]:ind1e[j]]]
        #compute availabilities:
        for j in range(N):
            rp = R[ind2[ind2s[j]:ind2e[j]]]
            rp[:-1] = np.maximum(rp[:-1])
            a = np.sum(rp, axis=0) - rp
            a[:-1] = np.minimum(a[:-1], 0)
            A[ind2[ind2s[j]:ind2e[j]]] = (1-lam)*a + lam*A[ind2[ind2s[j]:ind2e[j]]]
        #check for convergence:
        E = ((A[M-N:M]+R[M-N:M]) >0 )
        e[:, ((i-1)%convits)+1] = E
        K = np.sum(E, axis=0)
        if i >= convits or i >= maxits:
            se = np.sum(e, axis=1)
            unconverged = (np.sum((se==convits)+(se==0)) != N)
            if (not unconverged and (K>0)) or (i==maxits):
                dn=True
    



    return idx, netsim, dpsim, expref
































